import unittest,torch
from iam_tools.generation_alignment import AlignedWriterDenoiser
from iam_tools.generation_timing import TimingProbe,intervention_lengths
class TimingTests(unittest.TestCase):
 def fixture(self,aligned):
  torch.manual_seed(72);m=TimingProbe(alignment=aligned,writer_count=2,vocab_size=5,width=16,depth=2,heads=4,blocks_per_character=1.2);torch.nn.init.normal_(m.final[-1].weight,std=.02);m.eval()
  x=torch.zeros(2,6,384);t=torch.ones(2);text=torch.tensor([[0,1,2],[1,2,-1]]);mask=torch.tensor([[1]*6,[1,1,1,1,0,0]],dtype=torch.bool);w=torch.tensor([0,1]);return m,(x,t,text,mask),w
 def test_override_equal_native_counts_matches_original_both_arms(self):
  for aligned in [False,True]:
   with self.subTest(aligned=aligned):
    m,args,w=self.fixture(aligned);base=AlignedWriterDenoiser(**m.config);base.load_state_dict(m.state_dict());base.eval();a=base(*args,writer_ids=w);b=m(*args,writer_ids=w,position_lengths=args[3].sum(1));self.assertTrue(torch.allclose(a,b,atol=1e-7,rtol=1e-6));self.assertTrue(torch.equal(a,m(*args,writer_ids=w)))
 def test_pe_override_changes_output_not_mask(self):
  m,args,w=self.fixture(True);a=m(*args,writer_ids=w);b=m(*args,writer_ids=w,position_lengths=args[3].sum(1)+1);self.assertGreater(float((a-b).detach().abs().max()),1e-5);self.assertTrue((b[1,4:]==0).all());self.assertTrue(torch.isfinite(b).all())
 def test_separate_interventions_and_invalid_lengths(self):
  self.assertEqual(intervention_lengths(50,1,'pe_only'),(50,51));self.assertEqual(intervention_lengths(50,-1,'mask_only'),(49,50));self.assertEqual(intervention_lengths(50,1,'both'),(51,51));self.assertEqual(intervention_lengths(50,-1,'baseline'),(50,50))
  for args in [(1,1,'pe_only'),(50,3,'both'),(50,1,'unknown')]:
   with self.assertRaises(ValueError):intervention_lengths(*args)
 def test_invalid_override_rejected(self):
  m,args,w=self.fixture(True)
  for lengths in [torch.tensor([0,4]),torch.tensor([5.,4.]),torch.tensor([5])]:
   with self.assertRaises(ValueError):m(*args,writer_ids=w,position_lengths=lengths)
if __name__=='__main__':unittest.main()
