import unittest,torch
from iam_tools.generation_text_anchor import local_text_hint,AnchoredWriterDenoiser
from iam_tools.generation_coverage import WriterTextDenoiser
class TextAnchorTests(unittest.TestCase):
 def test_interpolation_uses_all_chars_and_omits_padding(self):
  e=torch.nn.Embedding(8,1);e.weight.data[:,0]=torch.arange(8)
  hint=local_text_hint(e,torch.tensor([[0,1,2,3,4,-1]]),torch.tensor([[True,True,False]]))
  self.assertEqual(hint[0,:,0].tolist(),[2.,6.,0.])
  h=local_text_hint(e,torch.tensor([[0,1]]),torch.ones(1,3,dtype=torch.bool));self.assertEqual(h[0,:,0].tolist(),[2.,2.5,3.])
 def test_zero_scale_exact_identity_and_same_state_keys(self):
  c=dict(writer_count=2,channels=8,vocab_size=4,width=16,depth=1,heads=2);base=WriterTextDenoiser(**c).eval();torch.nn.init.normal_(base.final[-1].weight);m=AnchoredWriterDenoiser(local_text_scale=0.,**c).eval();m.load_state_dict(base.state_dict())
  args=(torch.zeros(1,3,8),torch.ones(1),torch.tensor([[0,1]]),torch.ones(1,3,dtype=torch.bool));kw=dict(writer_ids=torch.tensor([0]))
  with torch.no_grad():self.assertTrue(torch.equal(base(*args,**kw),m(*args,**kw)))
 def test_null_text_never_leaks_via_local_hint(self):
  m=AnchoredWriterDenoiser(local_text_scale=.1,writer_count=1,channels=8,vocab_size=4,width=16,depth=1,heads=2).eval();torch.nn.init.normal_(m.final[-1].weight)
  args=(torch.zeros(1,3,8),torch.ones(1));mask=torch.ones(1,3,dtype=torch.bool);kw=dict(drop_text=torch.ones(1,dtype=torch.bool),writer_ids=torch.tensor([0]))
  with torch.no_grad():a=m(*args,torch.tensor([[0,1]]),mask,**kw);b=m(*args,torch.tensor([[2,3]]),mask,**kw)
  self.assertTrue(torch.equal(a,b))
 def test_guard_and_nonempty(self):
  with self.assertRaises(ValueError):AnchoredWriterDenoiser(local_text_scale=1.,writer_count=1)
  with self.assertRaises(ValueError):local_text_hint(torch.nn.Embedding(5,2),torch.tensor([[-1,-1]]),torch.ones(1,3,dtype=torch.bool))
if __name__=='__main__':unittest.main()
