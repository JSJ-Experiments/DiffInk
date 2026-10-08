import unittest,torch,numpy as np
from iam_tools.generation_attention import capture,describe
from iam_tools.generation_coverage import WriterTextDenoiser
class AttentionTests(unittest.TestCase):
 def test_observer_preserves_model_and_removes_hooks(self):
  torch.manual_seed(2);m=WriterTextDenoiser(2,channels=8,vocab_size=4,width=16,depth=2,heads=2).eval();torch.nn.init.normal_(m.final[-1].weight)
  x=torch.zeros(1,4,8);t=torch.ones(1);text=torch.tensor([[1,2,-1]]);mask=torch.ones(1,4,dtype=torch.bool);wi=torch.tensor([0]);before={k:v.clone() for k,v in m.state_dict().items()}
  with torch.no_grad():original=m(x,t,text,mask,writer_ids=wi)
  observed,w=capture(m,x,t,text,mask,writer_ids=wi);self.assertTrue(torch.allclose(original,observed,atol=2e-5,rtol=2e-5));self.assertEqual(len(w),2)
  for module in m.modules():self.assertFalse(module._forward_hooks);self.assertFalse(module._forward_pre_hooks)
  for k,v in m.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
  for v in w.values():self.assertEqual(v.shape,(1,2,4,4));self.assertTrue(np.all(v[...,3]==0))
 def test_train_observation_forbidden(self):
  with self.assertRaises(ValueError):capture(torch.nn.Linear(2,2),torch.ones(1,2))
 def test_monotone_and_uniform_descriptions(self):
  w=np.zeros((1,4,5));w[0,np.arange(4),np.arange(4)+1]=1.;d=describe(w,4,4)['heads'][0];self.assertAlmostEqual(d['time_token_barycenter_correlation'],1.);self.assertEqual(d['normalized_char_attention_entropy'],0.)
  w[:]=1/5;d=describe(w,4,4)['heads'][0];self.assertIsNone(d['time_token_barycenter_correlation']);self.assertAlmostEqual(d['normalized_char_attention_entropy'],1.)
 def test_invalid_bounds(self):
  with self.assertRaises(ValueError):describe(np.ones((1,3,3)),4,2)
if __name__=='__main__':unittest.main()
