import copy
import hashlib
import json
import unittest
from iam_tools.report_history_rollin import verify_log,verify_selection
from iam_tools.writer_expansion import training_schedule

class HistoryRollinReportTests(unittest.TestCase):
    def test_capacity_rejection_overrides_free_cer(self):
        def h(step,cer,passed):return dict(step=step,capacity_gate=dict(passed=passed),free=dict(correct=dict(cer=cer)),teacher=dict(offset_mse=.01))
        result=dict(history=[h(0,.46,True),h(250,0.,False),h(1000,.2,True)],best_step=1000,best_train_score=[.2,.01])
        verify_selection(result);result['best_step']=250
        with self.assertRaises(ValueError):verify_selection(result)
    def test_objective_probability_order_applied_draws_and_selection(self):
        cfg=dict(max_updates=1000,batch=8,schedule_seed=60142,rollin_max_probability=.2,rollin_ramp_steps=500,lr=1e-5,pen_weight=.02);ids=[str(j) for j in range(8)];data=dict(training_ids=ids);schedule=list(training_schedule(ids,1000,8,60142))
        h=dict(step=1,capacity_gate=dict(passed=True),free=dict(correct=dict(cer=.2)),teacher=dict(offset_mse=.01));r=dict(arm='rollin',last_step=1,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),history=[h],best_step=1,best_train_score=[.2,.01])
        row=dict(step=1,total_step=3001,sample_ids=schedule[0],rollin_probability=.2/500,lr=1e-5,pen_weight=.02,offset_mse=.5,pen_loss=1.,loss=.52,selected_transitions=0,eligible_transitions=20,selected_fraction=0.,draws_sha256='a'*64,selected_sha256='b'*64,raw_grad_norm=1.,mean_advance=.6,mean_sigma=.5)
        verify_log(cfg,data,r,[row])
        for key,value in [('rollin_probability',.5),('total_step',1),('loss',1.),('sample_ids',['blind']),('selected_fraction',1.)]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(ValueError):verify_log(cfg,data,r,[bad])
        teacher=copy.deepcopy(r);teacher['arm']='teacher';bad=copy.deepcopy(row);bad['rollin_probability']=0.;bad['selected_transitions']=1;bad['selected_fraction']=.05
        with self.assertRaises(ValueError):verify_log(cfg,data,teacher,[bad])

if __name__=='__main__':unittest.main()
