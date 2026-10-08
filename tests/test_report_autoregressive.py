import copy
import unittest
import numpy as np
from iam_tools.report_autoregressive_study import verify_log,verify_selection,svg

class AutoregressiveReportTests(unittest.TestCase):
    def fixture(self):
        cfg=dict(max_updates=8000,lr=2e-4,lr_final=5e-5,warmup_steps=200,lr_drop_step=6000,calibration_batches=8,pen_gradient_fraction=.25)
        c=dict(weight=.5,state_rng_unchanged=True,rows=[dict(offset_norm=2.,pen_norm=1.)]*8)
        result=dict(last_step=1,calibration=c)
        row=dict(step=1,pen_weight=.5,lr=2e-4*(.2+.8/200),loss=1.5,offset_mse=1.,pen_loss=1.,mean_advance=.3,mean_sigma=1.,raw_grad_norm=2.)
        return cfg,result,[row]
    def test_fail_closed_log_objective_scale_and_step_order(self):
        cfg,r,rows=self.fixture();verify_log(cfg,r,rows)
        for key,value in [('loss',1.),('pen_weight',1.),('lr',1e-4),('mean_advance',0.),('step',2)]:
            bad=copy.deepcopy(rows);bad[0][key]=value
            with self.assertRaises(ValueError):verify_log(cfg,r,bad)
    def test_selection_ignores_development(self):
        a=dict(step=0,free={'fixed_train8':{'correct':{'cer':1.}},'exposed_dev8':{'correct':{'cer':0.}}},teacher={'offset_mse':1.})
        b=dict(step=1000,free={'fixed_train8':{'correct':{'cer':.5}},'exposed_dev8':{'correct':{'cer':3.}}},teacher={'offset_mse':.3})
        r=dict(history=[a,b],best_step=1000,best_train_score=[.5,.3]);verify_selection(r);r['best_step']=0
        with self.assertRaises(ValueError):verify_selection(r)
    def test_marker_free_svg_stroke_breaks_no_smoothing_or_width_fit(self):
        points=np.c_[np.array([[0.,0.],[1.,1.],[2.,0.],[3.,1.]]),np.eye(3)[[0,1,0,2]]]
        s=svg(points);self.assertEqual(s.count('<path'),2);self.assertNotIn('<circle',s);self.assertNotIn(' C ',s);self.assertIn('width="312.000"',s)
        translated=points.copy();translated[:,:2]+=15.;self.assertEqual(s,svg(translated))
        with self.assertRaises(ValueError):svg(points[:0])

if __name__=='__main__':unittest.main()
