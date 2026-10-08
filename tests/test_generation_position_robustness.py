import unittest,torch
from iam_tools.generation_position_robustness import jitter_lengths
class JitterTests(unittest.TestCase):
 def test_reproducible_bounded_no_input_mutation(self):
  n=torch.tensor([20,50,80]*20);old=n.clone();a=jitter_lengths(n,torch.Generator().manual_seed(77));b=jitter_lengths(n,torch.Generator().manual_seed(77));self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(n,old));self.assertTrue(((a-n).abs()<=torch.ceil(n*.2)).all());self.assertTrue((a==n).any());self.assertTrue((a!=n).any())
 def test_zero_probability_preserves_original(self):
  n=torch.tensor([20,50]);self.assertTrue(torch.equal(n,jitter_lengths(n,torch.Generator().manual_seed(1),probability=0)))
 def test_bad_inputs_rejected(self):
  for n in [torch.tensor([1,2]),torch.tensor([2.,3.]),torch.tensor([[2,3]])]:
   with self.assertRaises(ValueError):jitter_lengths(n,torch.Generator())
if __name__=='__main__':unittest.main()

class RobustnessGuardTests(unittest.TestCase):
 def test_exact_two_arm_guard(self):
  from iam_tools.report_generation_position_robustness import verify
  with self.assertRaisesRegex(ValueError,'exact control/jitter'):verify('.',{}, {}, {})
 def test_models_and_training_scope_must_match(self):
  from iam_tools.report_generation_position_robustness import verify
  r={a:{} for a in ['control','pe_jitter']};d={'training_ids':{'control':['x'],'pe_jitter':['x']}};c={'models':{'control':{'depth':1},'pe_jitter':{'depth':2}}}
  with self.assertRaisesRegex(ValueError,'identical model'):verify('.',c,r,d)
 def test_only_predeclared_bounded_protocol(self):
  from iam_tools.report_generation_position_robustness import verify
  arms=['control','pe_jitter'];r={a:{} for a in arms};d={'training_ids':{a:['x'] for a in arms}};c={'models':{a:{} for a in arms},'parent_step':48000,'max_updates':8000,'lr':1e-5,'jitter_probability':1.,'jitter_fraction':.2}
  with self.assertRaisesRegex(ValueError,'bounded pinned protocol'):verify('.',c,r,d)
