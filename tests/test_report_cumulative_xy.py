import copy,hashlib,json,unittest
from iam_tools.report_cumulative_xy import verify_log
from iam_tools.writer_expansion import training_schedule
class CumulativeReportTests(unittest.TestCase):
    def test_anchor_policy_scale_objective_and_source_fail_closed(self):
        cfg=dict(max_updates=1000,batch=8,schedule_seed=62142,rollin_max_probability=.2,rollin_ramp_steps=500,lr=1e-5,pen_weight=.02,anchor_calibration=dict(weight=.01));ids=list('abcdefgh');schedule=list(training_schedule(ids,1000,8,62142));data=dict(training_ids=ids)
        h=dict(step=1,capacity_gate=dict(passed=True),free=dict(correct=dict(cer=.2)),teacher=dict(offset_mse=.5))
        result=dict(arm='rollin_anchor',last_step=1,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),history=[h],best_step=1,best_train_score=[.2,.5])
        row=dict(step=1,parent_body_step=3000,parent_head_updates=6000,sample_ids=schedule[0],rollin_probability=.2/500,lr=1e-5,pen_weight=.02,anchor_weight=.01,offset_mse=.5,pen_loss=1.,cumulative_xy_mse=.1,loss=.521,selected_transitions=0,eligible_transitions=20,selected_fraction=0.,draws_sha256='a'*64,selected_sha256='b'*64,raw_grad_norm=1.,mean_advance=.5,mean_sigma=1.)
        verify_log(cfg,data,result,[row])
        for k,v in [('anchor_weight',1.),('loss',1.),('parent_head_updates',2000),('cumulative_xy_mse',100.),('rollin_probability',.5)]:
            bad=copy.deepcopy(row);bad[k]=v
            with self.assertRaises(ValueError):verify_log(cfg,data,result,[bad])
        for arm in ['teacher','teacher_anchor']:
            r=copy.deepcopy(result);r['arm']=arm;bad=copy.deepcopy(row);bad['rollin_probability']=0.;bad['anchor_weight']=0. if arm=='teacher' else .01;bad['loss']=.52+bad['anchor_weight']*.1;verify_log(cfg,data,r,[bad])
            bad['selected_transitions']=1;bad['selected_fraction']=.05
            with self.assertRaises(ValueError):verify_log(cfg,data,r,[bad])
if __name__=='__main__':unittest.main()
