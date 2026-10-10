import copy,unittest
from types import SimpleNamespace
import torch
from iam_tools.wsl_conditioning import models,matched_batch,validate,ARMS,PARENT
class NativeConditioningTests(unittest.TestCase):
 def config(self):
  return dict(arms=list(ARMS),model=dict(dim=32,latent_dim=40,num_text_embedding=12,text_dim=16,text_mask_padding=True,conv_layers=1,dim_head=8,depth=2,heads=4,ff_mult=2,dropout=0.,long_skip_connection=False),parent_relative=PARENT,prefix_keep_probability=0.,text_drop_probability=.1,open_confirmations=False,use_amp=False,max_updates=12000,max_train_wall_seconds=1800,batch=32,eval_steps=[0,1000,3000,6000,12000],reader_backend='aten_gpu_no_miopen',backend='AMD ROCm WSL2; fresh controls, NOT T4 numerical equivalence',quality_failures=['unreadable text','severe underlifting','no reliable content sensitivity'],lr=5e-5,min_lr=1e-6,warmup_updates=600,clip=1.,sampling_steps=50,eval_noise_seeds=[73142,73143],eval_guidance=[1.,2.],no_codec_training=True,no_kl=True,no_ctc_training=True,no_style_training=True,not_promoted=True)
 def pool(self):
  return SimpleNamespace(mu=torch.arange(2*5*384).reshape(2,5,384).float()/100,std=torch.full((2,5,384),.1),index={'a':0,'b':1},lengths={'a':3,'b':5},chars={'a':2,'b':3},text=torch.tensor([[1,2,-1],[3,4,5]]),mask=torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool),prefix=torch.tensor([1,2]))
 def test_analogue_initialization_is_bitwise_shared(self):
  a,b,identity=models(self.config()['model'],812);old=a.state_dict();new=b.state_dict()
  torch.testing.assert_close(old['input_embed.proj.weight'][:,:40],new['input_embed.proj.weight'],rtol=0,atol=0)
  torch.testing.assert_close(old['input_embed.proj.weight'][:,40:],new['text_proj.weight'],rtol=0,atol=0)
  torch.testing.assert_close(old['transformer_blocks.0.attn_norm.linear.weight'],new['transformer_blocks.0.attn_norm_x.linear.weight'],rtol=0,atol=0)
  torch.testing.assert_close(old['transformer_blocks.0.ff.ff.0.0.weight'],new['transformer_blocks.0.ff_x.ff.0.0.weight'],rtol=0,atol=0)
  self.assertGreater(identity['joint_parameters'],identity['concat_parameters']);self.assertTrue(identity['not_same_architecture_or_parameter_count'])
 def test_all_shared_embedding_rows_and_output_initial_zeros(self):
  a,b,_=models(self.config()['model'],812)
  torch.testing.assert_close(a.text_embed.text_embed.weight,b.text_embed.text_embed.weight[:len(a.text_embed.text_embed.weight)],rtol=0,atol=0)
  x=torch.randn(2,5,40);text=torch.tensor([[1,2],[3,4]])
  for m in [a,b]:self.assertEqual(float(m(x,x,text,torch.tensor([.1,.9])).abs().max().detach()),0.)
 def test_data_noise_and_cfg_independent_of_model_rng_consumption(self):
  p=self.pool();one=matched_batch(p,['a','b'],1,17,1000);torch.randn(25000);two=matched_batch(p,['a','b'],1,17,1000)
  for x,y in zip(one[:-1],two[:-1]):torch.testing.assert_close(x,y,rtol=0,atol=0)
  self.assertEqual(one[-1],two[-1]);self.assertEqual(one[0].shape,(2,5,384));self.assertEqual(one[5].shape,(2,5,384))
 def test_step_and_id_order_change_actual_draws(self):
  p=self.pool();a=matched_batch(p,['a','b'],1,17,1000);b=matched_batch(p,['a','b'],2,17,1000)
  self.assertFalse(torch.equal(a[5],b[5]));self.assertFalse(torch.equal(a[0],b[0]));self.assertTrue((a[4]<1000).all())
  c=matched_batch(p,['b','a'],1,17,1000);torch.testing.assert_close(c[1][0],p.text[1]);torch.testing.assert_close(c[2][0],p.mask[1])
  with self.assertRaises(ValueError):matched_batch(p,['a'],0,17,1000)
 def test_bounded_native_protocol(self):validate(self.config())
 def test_no_prefix_kl_ctc_amp_backend_or_extra_budget(self):
  for k,v in [('prefix_keep_probability',.7),('use_amp',True),('open_confirmations',True),('max_updates',24000),('max_train_wall_seconds',3600),('backend','T4'),('no_kl',False),('no_ctc_training',False),('batch',64),('clip',10.),('lr',.001),('eval_guidance',[2.])]:
   c=self.config();c[k]=v
   with self.assertRaises(ValueError):validate(c)
 def test_severe_underlifting_not_an_optional_metric(self):
  c=self.config();c['quality_failures'].remove('severe underlifting')
  with self.assertRaisesRegex(ValueError,'quality'):validate(c)
if __name__=='__main__':unittest.main()
