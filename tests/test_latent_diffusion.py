"""Text-only pure-noise generation, NULL/padding semantics, frozen reader field order."""
from pathlib import Path
import sys,unittest,copy
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from iam_tools.latent_diffusion import TextLatentDenoiser,fit_whitening,transform,masked_mse,cosine_schedule,forward_noise,ddim_sample
from iam_tools.generation_study import select_scope,validate_budget,collate,noise_for,read_sequence,decode_sample,summarize
from iam_tools.ocr_convergence import POOL_SHA

class DiffusionTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def model(self):
  torch.manual_seed(42);m=TextLatentDenoiser(channels=4,vocab_size=8,width=16,depth=2,heads=2).eval()
  with torch.no_grad():m.final[-1].weight.normal_(std=.1)
  return m
 def test_full_text_not_truncated_to_latent_length(self):
  m=self.model();x=torch.randn(1,2,4);mask=torch.ones(1,2,dtype=torch.bool);t=torch.tensor([.5]);a=torch.tensor([[0,1,2,3,4]]);b=a.clone();b[0,-1]=5
  self.assertGreater(float((m(x,t,a,mask)-m(x,t,b,mask)).abs().max()),1e-5)
  m(x,t,a,mask).square().sum().backward();self.assertGreater(float(m.text.weight.grad[6].abs().sum()),0.)
 def test_null_is_independent_of_every_character_and_pad_length(self):
  m=self.model();x=torch.randn(1,3,4);mask=torch.ones(1,3,dtype=torch.bool);t=torch.tensor([.5]);drop=torch.ones(1,dtype=torch.bool)
  a=m(x,t,torch.tensor([[0,1]]),mask,drop);b=m(x,t,torch.tensor([[6,7,3,-1,-1]]),mask,drop)
  torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)
 def test_padding_is_ignored_for_text_and_latents(self):
  m=self.model();x=torch.randn(1,3,4);mask=torch.ones(1,3,dtype=torch.bool);t=torch.tensor([.5]);text=torch.tensor([[1,2]])
  a=m(x,t,text,mask);padded=torch.cat([x,torch.full((1,4,4),float('nan'))],1);pm=torch.cat([mask,torch.zeros(1,4,dtype=torch.bool)],1)
  b=m(padded,t,torch.tensor([[1,2,-1,-1,-1]]),pm)
  torch.testing.assert_close(a,b[:,:3],atol=2e-6,rtol=2e-6);self.assertTrue(torch.isfinite(b).all());self.assertEqual(float(b[:,3:].abs().sum()),0.)
 def test_train_stats_do_not_include_unseen_and_roundtrip_all_channels(self):
  z={'train':torch.tensor([[0.,1.,2.],[2.,3.,4.]]),'unseen':torch.full((4,3),10000.)};s=fit_whitening(z,['train']);torch.testing.assert_close(s['mean'],torch.tensor([1.,2.,3.]));self.assertEqual(s['train_ids'],['train'])
  torch.testing.assert_close(transform(transform(z['unseen'],s),s,True),z['unseen'])
  z['train'][:,2]=7;s=fit_whitening(z,['train']);self.assertAlmostEqual(float(s['std'][2]),.1,places=6)
 def test_invalid_train_stats_and_budget_fail_before_gpu(self):
  validate_budget(8000,POOL_SHA)
  for n,sha in [(999,POOL_SHA),(12001,POOL_SHA),(True,POOL_SHA),(8000,'')]:
   with self.assertRaises(ValueError):validate_budget(n,sha)
  for ids in [[],['a','a']]:
   with self.assertRaises(ValueError):fit_whitening({'a':torch.zeros(2,4)},ids)
 def test_diffusion_schedule_and_forward_noise_formula(self):
  alpha=cosine_schedule();self.assertTrue((alpha[1:]<alpha[:-1]).all());self.assertGreater(float(alpha[0]),.99);self.assertLess(float(alpha[-1]),1e-7)
  clean=torch.full((2,3,4),2.);epsilon=torch.full_like(clean,3.);t=torch.tensor([0,999]);mask=torch.tensor([[True,True,False],[True,True,True]])
  x=forward_noise(clean,epsilon,t,alpha,mask);expected=alpha[t,None,None].sqrt()*clean+(1-alpha[t,None,None]).sqrt()*epsilon
  torch.testing.assert_close(x[mask],expected[mask]);self.assertEqual(float(x[0,2].abs().sum()),0.)
 def test_masked_loss_nan_padding_has_zero_gradient(self):
  x=torch.randn(1,3,4,requires_grad=True);target=torch.zeros_like(x);target[0,2]=float('nan');mask=torch.tensor([[True,True,False]])
  loss=masked_mse(x,target,mask);loss.backward();self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[0,2].abs().sum()),0.)
 def test_pure_noise_sampler_has_no_trajectory_argument_and_real_text_cfg(self):
  class Constant(torch.nn.Module):
   def __init__(self):super().__init__();self.calls=[];self.eval()
   def forward(self,x,t,text,mask,drop):
    self.calls.append(drop.clone());return torch.where(drop[:,None,None],0.,torch.ones_like(x)*2)
  m=Constant();noise=torch.randn(1,4,3);original=noise.clone();mask=torch.ones(1,4,dtype=torch.bool);text=torch.tensor([[0,1]])
  out=ddim_sample(m,noise,text,mask,cosine_schedule(),steps=5,guidance=3.)
  torch.testing.assert_close(out,torch.full_like(out,6.));torch.testing.assert_close(noise,original);self.assertEqual(sum(bool(v.all()) for v in m.calls),5)
  torch.testing.assert_close(ddim_sample(m,noise,text,mask,cosine_schedule(),steps=5,drop_text=True),torch.zeros_like(out))
 def test_noise_invariant_to_batch_collation_and_id_order(self):
  z={'a':torch.zeros(3,384),'b':torch.zeros(6,384)}
  a=noise_for(['a'],z,9142,'cpu');b=noise_for(['b','a'],z,9142,'cpu')
  torch.testing.assert_close(a[0],b[1,:3]);self.assertEqual(float(b[1,3:].abs().sum()),0.)
  self.assertFalse(torch.equal(a,noise_for(['a'],z,9143,'cpu')))
 def test_mask_guards_and_eval_requirement(self):
  m=self.model();mask=torch.ones(1,3,dtype=torch.bool);x=torch.randn(1,3,4);text=torch.tensor([[0]])
  m.train()
  with self.assertRaises(ValueError):ddim_sample(m,x,text,mask,cosine_schedule())
  m.eval()
  with self.assertRaises(ValueError):ddim_sample(m,x,text,mask*False,cosine_schedule())
 def scope(self):
  ids=[f't{i}' for i in range(41)]+[f'h{i}' for i in range(8)]
  records={i:dict(writer_id='10160',prompt_family='g09-301' if i.startswith('h') else f'form{int(i[1:])//7}',text='text '+i) for i in ids}
  return dict(records=records,splits=dict(large_train=ids),test_writers=['test'],dev_writers=['dev'])
 def test_generator_form_and_text_holdout_are_disjoint(self):
  m=self.scope();s=select_scope(m);self.assertEqual(len(s['train']),32);self.assertEqual(len(s['unseen_prompt']),8);self.assertFalse(set(s['train'])&set(s['unseen_prompt']))
  self.assertEqual(s,select_scope(m))
  m['records']['h0']['text']=m['records'][s['train'][0]]['text']
  with self.assertRaisesRegex(ValueError,'form/text leakage'):select_scope(m)
 def test_reserved_writer_rejected(self):
  m=self.scope();m['test_writers']=['10160']
  with self.assertRaisesRegex(ValueError,'reserved writer'):select_scope(m)
 def test_reader_receives_exact_phase_order_and_actual_stop_length(self):
  class Reader(torch.nn.Module):
   def forward(self,z,padding_mask,point_mask):
    self.z=z.clone();self.pm=point_mask.clone();self.lm=~padding_mask
    logits=torch.zeros(6,1,3);logits[:,0,1]=1;return logits
  reader=Reader();p=torch.zeros(17,5);p[:,:2]=torch.arange(34).reshape(17,2);p[:,2]=1;p[-1,2]=0;p[-1,4]=1
  self.assertEqual(read_sequence(reader,p,['a','b']),'a');self.assertEqual(int(reader.pm.sum()),17)
  torch.testing.assert_close(reader.z[0,:40,:2].T,p[:16].reshape(2,40));self.assertEqual(float(reader.z[:,40:].abs().sum()),0.)
 def test_generation_metrics_keep_free_stop_separate_from_oracle_window(self):
  rows=[dict(sample_id='a',policy='correct',free_errors=3,window_errors=0,characters=3,generated_points_at_stop=1,first_eoc_point=1)]
  s=summarize(rows,{'train':['a']})['train']['correct'];self.assertEqual(s['free_cer'],1);self.assertEqual(s['oracle_window_cer'],0)

if __name__=='__main__':unittest.main()
