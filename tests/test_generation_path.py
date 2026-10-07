import unittest
import torch
from iam_tools.generation_path import training_input,generate,native_train_score
from iam_tools.latent_diffusion import cosine_schedule,forward_noise,TextLatentDenoiser

class Echo(torch.nn.Module):
 def forward(self,x,time,text,mask,drop):
  self.input=x;self.time=time;self.drop=drop;return x

class GenerationPathTests(unittest.TestCase):
 def setUp(self):
  self.clean=torch.randn(2,4,8);self.eps=torch.randn_like(self.clean);self.mask=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool);self.t=torch.tensor([0,999]);self.a=cosine_schedule()
 def test_uniform_exact(self):
  x,t=training_input('uniform',self.clean,self.eps,self.t,self.a,self.mask)
  self.assertTrue(torch.equal(x,forward_noise(self.clean,self.eps,self.t,self.a,self.mask)));self.assertTrue(torch.equal(t,self.t.float()/999))
 def test_terminal_has_no_target_signal(self):
  x,t=training_input('terminal',self.clean,self.eps,self.t,self.a,self.mask)
  other,_=training_input('terminal',self.clean*1000,self.eps,self.t,self.a,self.mask)
  self.assertTrue(torch.equal(x,other));self.assertTrue(torch.equal(x,self.eps.masked_fill(~self.mask[...,None],0)));self.assertTrue(torch.equal(t,torch.ones(2)))
 def test_direct_no_noise_or_target_signal(self):
  x,t=training_input('direct',self.clean,self.eps,self.t,self.a,self.mask);self.assertEqual(float(x.abs().sum()),0)
  other,_=training_input('direct',self.clean*100,self.eps*100,self.t,self.a,self.mask);self.assertTrue(torch.equal(x,other))
 def test_sampler_terminal(self):
  m=Echo().eval();q=generate(m,self.eps,torch.ones(2,3,dtype=torch.long),self.mask,self.a,'terminal',True)
  self.assertTrue(torch.equal(q,self.eps.masked_fill(~self.mask[...,None],0)));self.assertTrue(m.drop.all());self.assertTrue(torch.equal(m.time,torch.ones(2)))
 def test_direct_sampling_noise_independent(self):
  m=Echo().eval();text=torch.ones(2,3,dtype=torch.long);x=generate(m,self.eps,text,self.mask,self.a,'zero');y=generate(m,self.eps*10,text,self.mask,self.a,'zero');self.assertTrue(torch.equal(x,y))
 def test_eval_guard(self):
  with self.assertRaises(ValueError):generate(Echo(),self.eps,torch.ones(2,3,dtype=torch.long),self.mask,self.a,'zero')
 def test_unknown_arm(self):
  with self.assertRaises(ValueError):training_input('unknown',self.clean,self.eps,self.t,self.a,self.mask)
 def test_train_only_selector(self):
  e={'aggregate':{'train':{'zero_correct':{'x_rmse':.1,'y_rmse':.2,'segment_vector_rmse':.4,'pen_f1_min':.8}},'unseen_prompt':{'zero_correct':{'x_rmse':999}}}}
  self.assertAlmostEqual(native_train_score(e,'direct'),.42)
 def test_zeros_model_remains_text_conditional(self):
  torch.manual_seed(4);m=TextLatentDenoiser(channels=8,vocab_size=4,width=16,depth=1,heads=2).eval();torch.nn.init.normal_(m.final[-1].weight)
  t1=torch.tensor([[0,1],[1,2]]);t2=torch.tensor([[2,3],[3,0]])
  x=generate(m,self.eps,t1,self.mask,self.a,'zero');y=generate(m,self.eps,t2,self.mask,self.a,'zero')
  self.assertGreater(float((x-y).abs().max()),1e-5)
if __name__=='__main__':unittest.main()
