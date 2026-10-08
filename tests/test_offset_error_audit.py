import unittest
import numpy as np
from iam_tools.offset_error_audit import offset_error_groups,aggregate_groups

class OffsetErrorAuditTests(unittest.TestCase):
    def target(self):return np.c_[np.array([[0.,1.],[1.,2.],[10.,3.],[11.,4.]]),np.eye(3)[[0,1,0,2]]]
    def test_topology_uses_previous_not_current_pen_state(self):
        q=self.target();r=offset_error_groups(q,np.zeros((4,2)),dict(mean=[0.,0.],std=[1.,1.]))
        self.assertEqual(r['origin']['points'],1);self.assertEqual(r['within_stroke']['points'],2);self.assertEqual(r['pen_up_jump']['points'],1)
        self.assertEqual(r['pen_up_jump']['energy'],82.) # 9X/1Y jump follows point1 pen-up
        a=aggregate_groups([r]);self.assertAlmostEqual(a['normalized_per_axis_mse'],87/8)
        self.assertAlmostEqual(sum(a['groups'][k]['fraction_of_total_energy'] for k in ['origin','within_stroke','pen_up_jump']),1.)
    def test_normalized_zero_baseline_not_zero_real_displacement(self):
        q=self.target();s=dict(mean=[1.,1.],std=[2.,2.]);r=offset_error_groups(q,None,s)
        baseline=np.cumsum(np.tile(s['mean'],(4,1)),axis=0)
        self.assertEqual(r,offset_error_groups(q,baseline,s))
    def test_zero_error_and_no_jump_do_not_nan(self):
        q=self.target();q[:,2:]=np.eye(3)[[0,0,0,2]];a=aggregate_groups([offset_error_groups(q,q[:,:2],dict(mean=[0.,0.],std=[1.,1.]))])
        self.assertEqual(a['total_energy'],0.);self.assertIsNone(a['groups']['pen_up_jump']['per_axis_mse']);self.assertEqual(a['groups']['origin']['fraction_of_total_energy'],0.)
    def test_malformed_targets_and_normalization_rejected(self):
        q=self.target();s=dict(mean=[0.,0.],std=[1.,1.])
        with self.assertRaises(ValueError):offset_error_groups(q,np.zeros((3,2)),s)
        with self.assertRaises(ValueError):offset_error_groups(q,None,dict(mean=[0.,0.],std=[0.,1.]))
        q[1,2:]=[0,0,1]
        with self.assertRaises(ValueError):offset_error_groups(q,None,s)

if __name__=='__main__':unittest.main()
