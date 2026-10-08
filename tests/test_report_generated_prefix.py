import copy,hashlib,json,unittest
from iam_tools.report_generated_prefix import verify_log,study_module,verify_pairing
from iam_tools import generated_prefix_study,continuous_prefix_study
from iam_tools.writer_expansion import training_schedule
class OwnPrefixReportTests(unittest.TestCase):
    def test_shared_objective_policy_optimizer_ratio_and_selection_guards(self):
        ids=list('abcdefgh');cfg=dict(max_updates=1000,batch=8,schedule_seed=64142,lr=1e-5,pen_weight=.02,teacher_anchor_weight=.01,parent_initial_optimizer_digest='opt',auxiliary_calibration=dict(weights=dict(xy=.001,pen=.002)));data=dict(training_ids=ids);schedule=list(training_schedule(ids,1000,8,64142));h=dict(step=1,capacity_gate=dict(passed=True),free=dict(correct=dict(cer=.2)),teacher=dict(offset_mse=.5))
        result=dict(arm='continuous_xy',last_step=1,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),initial_optimizer_tensor_digest='opt',history=[h],best_step=1,best_train_score=[.2,.5]);row=dict(step=1,parent_body_step=4000,parent_head_updates=6000,sample_ids=schedule[0],lr=1e-5,pen_weight=.02,teacher_anchor_weight=.01,own_xy_weight=.001,own_pen_weight=.002,teacher_offset=.5,teacher_pen=1.,teacher_xy=.1,base_loss=.521,own_xy=.2,own_pen=.3,loss=.5218,continuous_feedback_gradient=True,raw_grad_norm=1.,mean_teacher_advance=.5,mean_own_advance=.6,gradient_check=dict(base_norm=2.,own_xy_norm=20.,own_pen_norm=5.,weighted_xy_ratio=.01,weighted_pen_ratio=.005))
        verify_log(cfg,data,result,[row],continuous_prefix_study)
        for key,value in [('own_xy_weight',1.),('base_loss',2.),('loss',1.),('continuous_feedback_gradient',False)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):verify_log(cfg,data,result,[bad],continuous_prefix_study)
        cfg['recovery']=dict(arms=dict(continuous_xy=dict(step=1)));result['resume_step']=1;result['recovery']=copy.deepcopy(cfg['recovery']);verify_log(cfg,data,result,[row],continuous_prefix_study)
        result['resume_step']=0
        with self.assertRaises(ValueError):verify_log(cfg,data,result,[row],continuous_prefix_study)
        result['resume_step']=1
        bad=copy.deepcopy(row);bad['gradient_check']['weighted_pen_ratio']=.5
        with self.assertRaises(ValueError):verify_log(cfg,data,result,[bad],continuous_prefix_study)
        result['initial_optimizer_tensor_digest']='reset'
        with self.assertRaises(ValueError):verify_log(cfg,data,result,[row],continuous_prefix_study)
    def test_generic_pairing_checks_no_teacher_name_or_draws_required(self):
        cfg=dict(parent_initial_model_digest='model',parent_initial_optimizer_digest='opt')
        for module in (generated_prefix_study,continuous_prefix_study):
            results={a:dict(initial_state_sha256='model',initial_optimizer_tensor_digest='opt',schedule_sha256='schedule') for a in module.ARMS}
            logs={a:[dict(sample_ids=list('abcdefgh'),teacher_offset=.5,teacher_pen=.1,teacher_xy=.01,own_xy=.2,own_pen=.3)] for a in module.ARMS}
            verify_pairing(cfg,results,logs,module)
            for key in ('initial_state_sha256','initial_optimizer_tensor_digest','schedule_sha256'):
                bad=copy.deepcopy(results);bad[module.ARMS[-1]][key]='changed'
                with self.assertRaises(ValueError):verify_pairing(cfg,bad,logs,module)
            for key,value in [('sample_ids',list('hgfedcba')),('own_xy',.4)]:
                bad=copy.deepcopy(logs);bad[module.ARMS[-1]][0][key]=value
                with self.assertRaises(ValueError):verify_pairing(cfg,results,bad,module)
            logs[module.ARMS[-1]]=[]
            with self.assertRaises(ValueError):verify_pairing(cfg,results,logs,module)

    def test_unknown_study_scope_rejected(self):
        self.assertIs(study_module('checkpoints/iam_generated_prefix/run'),generated_prefix_study)
        self.assertIs(study_module('checkpoints/iam_continuous_prefix/run'),continuous_prefix_study)
        with self.assertRaises(ValueError):study_module('checkpoints/blind/test')
if __name__=='__main__':unittest.main()
