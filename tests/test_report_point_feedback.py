import copy
import hashlib
import json
import unittest
from iam_tools.report_point_feedback import verify_log
from iam_tools.writer_expansion import training_schedule

class PointFeedbackReportTests(unittest.TestCase):
    def test_selection_objective_and_actual_scope_fail_closed(self):
        cfg=dict(max_updates=3000,warmup_steps=100,lr=3e-4,lr_final=5e-5,lr_drop_step=2400,batch=8,schedule_seed=59143,calibration_batches=1,pen_gradient_fraction=.25);ids=[str(j) for j in range(8)];schedule=list(training_schedule(ids,3000,8,59143))
        r=dict(last_step=1,schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),calibration=dict(weight=.5,state_rng_unchanged=True,rows=[dict(offset_norm=2.,pen_norm=1.)]),history=[dict(step=0,teacher=dict(offset_mse=2.,min_pen_f1=0.)),dict(step=1,teacher=dict(offset_mse=1.,min_pen_f1=.3))],best_step=1,best_train_score=[1.,-.3])
        rows=[dict(step=1,sample_ids=schedule[0],lr=3e-4*(.2+.8/100),pen_weight=.5,loss=1.5,offset_mse=1.,pen_loss=1.,raw_grad_norm=1.,mean_sigma=1.,mean_advance=.5)]
        verify_log(cfg,r,rows,ids)
        for mode in ['selection','sample','loss','lr','weight']:
            rr=copy.deepcopy(r);q=copy.deepcopy(rows)
            if mode=='selection':rr['best_step']=0
            if mode=='sample':q[0]['sample_ids'][0]='blind'
            if mode=='loss':q[0]['loss']=0.
            if mode=='lr':q[0]['lr']=1.
            if mode=='weight':q[0]['pen_weight']=1.
            with self.assertRaises(ValueError):verify_log(cfg,rr,q,ids)

if __name__=='__main__':unittest.main()
