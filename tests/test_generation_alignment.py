import copy,unittest
import numpy as np,torch
from iam_tools.generation_alignment import AlignedWriterDenoiser,gaussian_bias
from iam_tools.generation_coverage import WriterTextDenoiser
from iam_tools.generation_composition import fit_duration,predict_duration,seal_confirmation,learning_rate
from iam_tools.generation_duration_eval import inputs
class AlignmentTests(unittest.TestCase):
 def fixture(self):
  torch.manual_seed(12);m=AlignedWriterDenoiser(writer_count=2,vocab_size=5,width=16,depth=2,heads=4,alignment=True,blocks_per_character=.8)
  torch.nn.init.normal_(m.final[-1].weight,std=.02);m.eval()
  x=torch.zeros(2,5,384);time=torch.ones(2);text=torch.tensor([[0,1,2,3],[1,2,-1,-1]]);mask=torch.tensor([[1]*5,[1,1,1,0,0]],dtype=torch.bool);wi=torch.tensor([0,1])
  return m,(x,time,text,mask),wi
 def test_global_matches_original_and_state_compatible(self):
  m,args,wi=self.fixture();base=WriterTextDenoiser(writer_count=2,vocab_size=5,width=16,depth=2,heads=4);base.load_state_dict({k:v for k,v in m.state_dict().items() if '_pe_scale' not in k});base.eval();m.alignment=False
  self.assertTrue(torch.equal(base(*args,writer_ids=wi),m(*args,writer_ids=wi)))
 def test_gaussian_bias_padding_global_head_delayed_support_null(self):
  mask=torch.tensor([[1,1,1,1,0],[1,0,0,0,0]],dtype=torch.bool);b=gaussian_bias(mask,8,4,1.).reshape(2,4,8,5)
  self.assertTrue(torch.isneginf(b[0,:,:,4]).all());self.assertTrue((b[0,3,:,:4]==0).all());self.assertTrue(torch.isfinite(b[0,:3,:,:4]).all());self.assertEqual(b[0,0,0,1:].argmax().item(),0);self.assertEqual(b[0,0,7,1:].argmax().item(),2)
  self.assertTrue(torch.isfinite(b[1,:,:,0]).all());self.assertTrue(torch.isneginf(b[1,:,:,1:]).all())
 def test_output_finite_masked_batch_and_null(self):
  m,args,wi=self.fixture();out=m(*args,writer_ids=wi);self.assertTrue(torch.isfinite(out).all());self.assertTrue((out[1,3:]==0).all())
  null=m(*args,writer_ids=wi,drop_text=torch.ones(2,dtype=torch.bool));self.assertTrue(torch.isfinite(null).all())
  one=m(args[0][1:,:3],args[1][1:],args[2][1:,:2],args[3][1:,:3],writer_ids=wi[1:]);self.assertTrue(torch.allclose(one,out[1:,:3],atol=2e-6,rtol=1e-5))
 def test_both_pe_amplitudes_and_attention_receive_gradients(self):
  m,args,wi=self.fixture();m.train();m(*args,writer_ids=wi).square().mean().backward()
  for p in [m.text_pe_scale,m.ink_pe_scale,m.blocks[0].cross_attention.in_proj_weight]:self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum(),0)
 def test_inputs_target_free_no_id_or_points(self):
  x,mask,text=inputs(['ab','a'],[3,1],['a','b'],'cpu');self.assertEqual(x.shape,(2,3,384));self.assertTrue((x==0).all());self.assertEqual(mask.sum(1).tolist(),[3,1]);self.assertEqual(text.tolist(),[[0,1],[0,-1]])
 def test_bad_prior_settings(self):
  mask=torch.ones(1,2,dtype=torch.bool)
  for r in [0,float('nan')]:
   with self.assertRaises(ValueError):gaussian_bias(mask,4,4,r)
class CompositionTests(unittest.TestCase):
 def test_duration_train_only_and_inference_text_writer_only(self):
  r={str(i):dict(text='abc '*i,writer_id='a' if i%2 else 'b',points=20*i) for i in range(1,11)}
  ids=list(r);m=fit_duration(r,ids,['a','b']);n=predict_duration(m,'abc abc abc','a');self.assertTrue(1<=n<=256)
  withheld=dict(text='Z'*100,points=9999,writer_id='c');r['held']=withheld;m2=fit_duration(r,ids,['a','b']);self.assertEqual(m,m2)
  self.assertTrue(1<=predict_duration(m,'abc','unknown')<=256)
 def test_seal_excludes_prior_sources_forms_texts_reserved_oov(self):
  records={};train=[]
  for j in range(8):
   w=str(j);i='t'+w;train.append(i);records[i]=dict(writer_id=w,text='abc 0123456789',points=200,prompt_family='old'+w)
   for k in range(2):records[f'h{j}_{k}']=dict(writer_id=w,text=f'abc {j}{k}',points=200,prompt_family='new'+w)
  records['oov']=dict(writer_id='0',text='Q',points=200,prompt_family='novel');history={i:records[i] for i in train};m=dict(records=records,splits=dict(large_train=list(records)),dev_writers=[],test_writers=[])
  seal=seal_confirmation(m,history,train,[str(j) for j in range(8)]);self.assertEqual(len(seal['ids']),16);self.assertNotIn('oov',seal['ids']);self.assertTrue(not set(seal['ids'])&set(train));self.assertEqual(seal,seal_confirmation(m,history,train,[str(j) for j in range(8)]))
  m['test_writers']=['0']
  with self.assertRaises(ValueError):seal_confirmation(m,history,train,[str(j) for j in range(8)])
 def test_lr_schedule_shared_warmup_and_polish(self):
  self.assertAlmostEqual(learning_rate(1000,24000),5e-5);self.assertAlmostEqual(learning_rate(16000,24000),5e-5);self.assertAlmostEqual(learning_rate(24000,24000),1e-5)
  with self.assertRaises(ValueError):learning_rate(0,24000)
if __name__=='__main__':unittest.main()
