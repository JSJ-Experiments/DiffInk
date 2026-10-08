import copy,hashlib,json,unittest
from iam_tools.report_own_delta import verify_log,verify_pairing
from iam_tools.own_delta_study import ARMS,PARENT_SHA,arm_weight
from iam_tools.writer_expansion import training_schedule
from tests.test_own_delta_study import OwnDeltaStudyTests
class ReportOwnDeltaTests(unittest.TestCase):
    def fixture(self,arm='ink_delta'):
        cfg,data=OwnDeltaStudyTests().fixture();cfg['parent_initial_optimizer_digest']='adam';cfg['parent_initial_model_digest']='model'
        schedule=list(training_schedule(data['training_ids'],cfg['max_updates'],cfg['batch'],cfg['schedule_seed']));wd=arm_weight(arm,cfg)
        r=dict(step=1,sample_ids=schedule[0],teacher_offset=.2,teacher_pen=.3,teacher_xy=.4,own_xy=.5,own_pen=.6,own_ink_delta=.7,own_all_delta=.8,own_offset_diagnostic=.8,lr=cfg['lr'],pen_weight=cfg['pen_weight'],teacher_anchor_weight=cfg['teacher_anchor_weight'],own_xy_weight=cfg['own_xy_weight'],own_pen_weight=cfg['own_pen_weight'],parent_joint_step=1000,delta_weight=wd,continuous_feedback_gradient=True,raw_grad_norm=6.,clipped=True,mean_teacher_advance=.5,mean_own_advance=.6,gradient_check=dict(complete_norm=2,delta_norm=3,weighted_delta_ratio=wd*3/2))
        r['base_loss']=r['teacher_offset']+cfg['pen_weight']*r['teacher_pen']+cfg['teacher_anchor_weight']*r['teacher_xy'];r['complete_loss']=r['base_loss']+cfg['own_xy_weight']*.5+cfg['own_pen_weight']*.6;r['loss']=r['complete_loss']+wd*(.7 if arm=='ink_delta' else .8)
        h=dict(step=1,capacity_gate=dict(passed=True),free=dict(correct=dict(cer=.2)),teacher=dict(offset_mse=.3))
        result=dict(arm=arm,last_step=1,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),initial_optimizer_tensor_digest='adam',initial_state_sha256='model',parent_checkpoint_sha256=PARENT_SHA,parent_joint_step=1000,reader_unchanged=True,not_promoted=True,stop='wall_limit',clip_fraction=1.,history=[h],best_step=1,best_train_score=[.2,.3])
        return cfg,data,result,[r]
    def test_complete_scalar_parent_plus_one_delta_and_actual_clipping(self):
        for arm in ARMS:
            cfg,data,result,rows=self.fixture(arm);verify_log(cfg,data,result,rows)
            for key,val in [('complete_loss',0.),('loss',0.),('own_all_delta',.9),('clipped',False),('delta_weight',1.),('own_xy_weight',0.)]:
                bad=copy.deepcopy(rows);bad[0][key]=val
                with self.assertRaises(ValueError):verify_log(cfg,data,result,bad)
            for key,val in [('best_step',0),('parent_joint_step',2000),('clip_fraction',0.),('stop','budget_completed'),('reader_unchanged',False)]:
                bad=copy.deepcopy(result);bad[key]=val
                with self.assertRaises(ValueError):verify_log(cfg,data,bad,rows)
    def test_gradients_and_actual_order_not_fabricated_random_ledgers(self):
        cfg,data,result,rows=self.fixture();bad=copy.deepcopy(rows);bad[0]['gradient_check']['weighted_delta_ratio']=1.
        with self.assertRaises(ValueError):verify_log(cfg,data,result,bad)
        bad=copy.deepcopy(rows);bad[0]['sample_ids'].reverse()
        with self.assertRaises(ValueError):verify_log(cfg,data,result,bad)
        results={a:self.fixture(a)[2] for a in ARMS};logs={a:self.fixture(a)[3] for a in ARMS}
        from iam_tools import own_delta_study
        verify_pairing(cfg,results,logs,own_delta_study)
        logs['all_delta'][0]['own_xy']+=1
        with self.assertRaises(ValueError):verify_pairing(cfg,results,logs,own_delta_study)
if __name__=='__main__':unittest.main()
