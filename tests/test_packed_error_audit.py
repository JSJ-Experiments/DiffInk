import unittest
import numpy as np
from iam_tools.packed_error_audit import block_error_statistics
class PackedErrorTests(unittest.TestCase):
 def test_constant_block_bias_seam_and_orthogonal_decomposition(self):
  true=np.c_[np.arange(24.),np.zeros(24)];pred=true.copy();pred[8:16,1]=1;states=np.zeros(24,int);states[-1]=2
  r=block_error_statistics([(pred,true,states)]);self.assertEqual(r['point_block_mean_energy_fraction'],1.)
  self.assertEqual(r['links']['seam']['segment_error_energy'],1.);self.assertEqual(r['links']['within']['segment_error_energy'],0.)
  self.assertEqual(r['links']['seam']['shape_difference_energy'],0.)
 def test_signed_cross_term_partial_blocks_and_stroke_boundaries(self):
  rng=np.random.default_rng(1);true=rng.normal(size=(19,2));pred=true+rng.normal(size=(19,2));states=np.zeros(19,int);states[7]=1;states[-1]=2
  r=block_error_statistics([(pred,true,states)]);e=r['point_energy'];self.assertAlmostEqual(e['error'],e['block_mean']+e['shape'])
  for row in r['links'].values():self.assertAlmostEqual(row['segment_error_energy'],row['block_mean_difference_energy']+row['shape_difference_energy']+row['twice_cross_term'])
  self.assertEqual(r['links']['seam']['links'],1)
 def test_perfect_short_and_invalid(self):
  q=np.zeros((1,2));r=block_error_statistics([(q,q,[2])]);self.assertIsNone(r['point_block_mean_energy_fraction']);self.assertEqual(r['links']['seam']['links'],0)
  with self.assertRaises(ValueError):block_error_statistics([(q,q,[9])])
  with self.assertRaises(ValueError):block_error_statistics([(q,q,[2])],0)
if __name__=='__main__':unittest.main()
