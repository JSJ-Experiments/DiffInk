import unittest
import numpy as np
from iam_tools.offset_drift_audit import cumulative_drift

class OffsetDriftTests(unittest.TestCase):
    def test_pure_per_move_bias_accumulates_and_oracle_cancels(self):
        target=np.array([[0.,.2],[.1,.3],[.2,.1],[.5,.2]])
        drift=np.arange(1,5)[:,None]*np.array([.01,-.02]);r=cumulative_drift(target,target+drift)
        np.testing.assert_allclose(r['mean_index_displacement_error'],[.01,-.02],atol=1e-12)
        np.testing.assert_allclose(r['oracle_endpoint_debiased_axis_rmse'],[0.,0.],atol=1e-12)
        self.assertTrue(r['diagnostic_only']);self.assertTrue(r['not_generation'])
    def test_zero_endpoint_jitter_is_not_smoothed_away(self):
        target=np.zeros((4,2));error=np.array([[.1,0],[-.1,0],[.1,0],[0,0]])
        r=cumulative_drift(target,target+error)
        self.assertEqual(r['raw_axis_rmse'],r['oracle_endpoint_debiased_axis_rmse'])
    def test_point_origin_and_bad_inputs(self):
        r=cumulative_drift([[2.,3.]],[[2.1,3.2]])
        np.testing.assert_allclose(r['endpoint_xy_error'],[.1,.2],atol=1e-12)
        for a,b in [([],[]),([[1.,2.]],[[1.,2.],[3.,4.]]),([[0.,np.nan]],[[0.,0.]])]:
            with self.assertRaises(ValueError):cumulative_drift(a,b)

if __name__=='__main__':unittest.main()
