import copy,unittest
from iam_tools.report_generation_weak_alignment import verify_auxiliary_log

class WeakReportTests(unittest.TestCase):
    def fixture(self,arm):
        cfg=dict(auxiliary_weights=dict(xy=.2,first_difference=.3))
        cal=dict(coefficient=.02,state_rng_unchanged=True,step=1000,batches=[dict(base_norm=2.,auxiliary_norm=10.) for _ in range(8)])
        r=dict(calibration=cal,alignment_weight=.02 if arm=='weak_alignment' else 0.)
        rows=[]
        for step in [1,1000,1001,48000]:
            weight=.02 if arm=='weak_alignment' and step>=1001 else 0.
            rows.append(dict(step=step,alignment_weight=weight,alignment_cross_entropy=2.,base_mse=.1,physical_xy_mse=.2,segment_mse=.3,loss=.1+.2*.2+.3*.3+weight*2.))
        return cfg,r,rows

    def test_exact_auxiliary_activation_and_objective(self):
        for arm in ['control','weak_alignment']:
            cfg,r,rows=self.fixture(arm);verify_auxiliary_log(cfg,r,rows,arm)
            bad=copy.deepcopy(rows);bad[0]['alignment_weight']=.02
            with self.assertRaises(ValueError):verify_auxiliary_log(cfg,r,bad,arm)
            bad=copy.deepcopy(rows);bad[-1]['loss']+=.01
            with self.assertRaises(ValueError):verify_auxiliary_log(cfg,r,bad,arm)

    def test_calibration_not_arbitrary_weight_or_giant_loss(self):
        cfg,r,rows=self.fixture('weak_alignment')
        bad=copy.deepcopy(r);bad['calibration']['coefficient']=.2
        with self.assertRaises(ValueError):verify_auxiliary_log(cfg,bad,rows,'weak_alignment')
        bad=copy.deepcopy(r);bad['calibration']['step']=0
        with self.assertRaises(ValueError):verify_auxiliary_log(cfg,bad,rows,'weak_alignment')
