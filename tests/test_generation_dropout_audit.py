import unittest
from iam_tools.generation_dropout_audit import summarize_rows
class DropReplayTests(unittest.TestCase):
 def test_groups_do_not_confuse_conditioned_with_null(self):
  rows=[dict(step=i,base_mse=float(i),physical_xy_mse=2.*i,segment_mse=3.*i,gradient_norm=4.*i) for i in [1,2,3]]
  r=summarize_rows(rows,{1:0,2:1,3:2});self.assertEqual(r['no_null']['updates'],1);self.assertEqual(r['has_null']['updates'],2);self.assertEqual(r['has_null']['base_mse'],2.5)
 def test_empty_group_is_unknown_not_zero(self):
  r=summarize_rows([],{});self.assertIsNone(r['no_null']['base_mse']);self.assertEqual(r['has_null']['updates'],0)
if __name__=='__main__':unittest.main()
