import unittest,torch
from iam_tools.generation_anchor_refresh import calibrate
from iam_tools.generation_coverage import WriterTextDenoiser
from iam_tools.generation_cache import CachedLatentPool
from iam_tools.ocr_context_study import tensor_digest
class RefreshTests(unittest.TestCase):
 def fixture(self):
  torch.manual_seed(3);model=WriterTextDenoiser(2,channels=384,vocab_size=3,width=16,depth=1,heads=2);torch.nn.init.normal_(model.final[-1].weight,std=.02)
  latents={};records={}
  for i in ['a','b']:
   fields=torch.zeros(16,5);fields[:,0]=torch.linspace(0,4,16);fields[:,1]=torch.linspace(0,4,16).sin();fields[:,2]=1.;fields[-1,2]=0.;fields[-1,4]=1.
   z=torch.zeros(2,384);z[:,:40]=fields.reshape(2,40);latents[i]=z;records[i]=dict(points=16,text='ab',writer_id='1' if i=='a' else '2')
  stats=dict(mean=torch.zeros(384),std=torch.ones(384));pool=CachedLatentPool(latents,records,['a','b',' '],['a','b'],stats,device='cpu')
  return model,pool,records,['1','2'],stats
 def test_calibration_preserves_state_and_backward_buffers(self):
  model,pool,records,writers,stats=self.fixture();digest=tensor_digest(model.state_dict());r=calibrate(model,pool,records,writers,stats,[['a'],['b']],['a','b'])
  self.assertEqual(tensor_digest(model.state_dict()),digest);self.assertTrue(model.training);self.assertTrue(all(p.grad is None for p in model.parameters()))
  self.assertAlmostEqual(r['weights']['xy']*r['aggregate_gradient_norms']['xy']/r['aggregate_gradient_norms']['base'],.25)
  self.assertAlmostEqual(r['weights']['first_difference']*r['aggregate_gradient_norms']['first_difference']/r['aggregate_gradient_norms']['base'],.10)
 def test_no_held_calibration_or_empty_scope(self):
  model,pool,records,writers,stats=self.fixture()
  for batches in [[],[['b']],[[]]]:
   with self.assertRaises(ValueError):calibrate(model,pool,records,writers,stats,batches,['a'])
 def test_invalid_fractions_restore_mode(self):
  model,pool,records,writers,stats=self.fixture();model.eval()
  with self.assertRaises(ValueError):calibrate(model,pool,records,writers,stats,[['a']],['a'],fractions=(5.,.1))
  self.assertFalse(model.training)
 def test_degenerate_fresh_zero_head_rejected(self):
  model,pool,records,writers,stats=self.fixture();torch.nn.init.zeros_(model.final[-1].weight)
  with self.assertRaises(ValueError):calibrate(model,pool,records,writers,stats,[['a']],['a'])
if __name__=='__main__':unittest.main()
