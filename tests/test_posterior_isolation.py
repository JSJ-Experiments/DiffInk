"""Real autograd/Adam isolation, including restored stale gradients and moments."""
from pathlib import Path
import sys,unittest,copy,tempfile
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from iam_tools.posterior_isolation import configure_arm,verify_frozen,assert_output_isolation,ARMS,gradient_diagnostics
from iam_tools.fast_geometry import fast_terms
from iam_tools.codec_kl_study import protect_pen_variance_weights
from iam_tools.kl_tradeoff import source_scales,evaluate
from iam_tools.ocr_pool_study import single_batch

class IsolationTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):torch.set_num_threads(2)
 def fixture(self):
  from test_vae_ctc import small_config
  from model.vae import VAE
  from iam_tools.ocr_joint_adapter import PolyphaseGRUOCR
  from iam_tools.ocr_recurrent import make_head
  cfg=small_config();cfg.latent_dim=48;cfg.hidden_dims=[8,16,48];cfg.decoder_dims=[48,16,128];cfg.trans_dropout=0.;cfg.model_input_scale=.01;cfg.use_decoder_padding_mask=True
  torch.manual_seed(42);m=VAE(cfg).eval()
  m.ocr_model=PolyphaseGRUOCR(make_head(dict(latent_dim=48,ocr_hidden_dim=16),4,None),None).eval().requires_grad_(False)
  m.style_classifier.requires_grad_(False);m.transformer_decoder.requires_grad_(False)
  body=[p for n,p in m.named_parameters() if p.requires_grad and not n.startswith('conv_logvar.')]
  opt=torch.optim.AdamW([dict(params=body,name='body',lr=1e-7),dict(params=list(m.conv_logvar.parameters()),name='posterior',lr=1e-3)],betas=(.9,.99),weight_decay=0)
  for g in opt.param_groups:
   for p in g['params']:p.grad=torch.ones_like(p)
  opt.step() # Nonzero restored momentum plus stale gradients.
  points=np.zeros((25,5),np.float32);points[:,0]=np.linspace(0,100,25);points[:,1]=np.sin(np.linspace(0,3,25))*10;points[:,2]=1;points[-1,2]=0;points[-1,4]=1
  batch=single_batch(points,'ab',['a','b','c'])
  return m,opt,batch
 def test_arm_coefficients_isolate_freeze_not_kl(self):self.assertEqual(ARMS,{'joint':1e-6,'posterior_only':1e-6})
 def test_freeze_clears_stale_grad_and_preserves_moments_steps(self):
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only')
  self.assertEqual([n for n,p in m.named_parameters() if p.requires_grad],['conv_logvar.weight','conv_logvar.bias'])
  for _ in range(3):
   opt.zero_grad(set_to_none=False);loss=fast_terms(m,*b[:2],torch.randn(1,48,4),include_kl=True)
   (loss*torch.tensor([1.,.1,.02,1e-6])).sum().backward();opt.step();verify_frozen(m,opt,snap)
  self.assertTrue(all(p.grad is None for n,p in m.named_parameters() if not n.startswith('conv_logvar.')))
  self.assertEqual(int(opt.state[m.conv_logvar.weight]['step']),4)
 def test_frozen_decoder_preserves_input_backward(self):
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only');opt.zero_grad(set_to_none=False)
  terms=fast_terms(m,*b[:2],torch.randn(1,48,4),include_kl=True)
  self.assertFalse(terms[0].detach().requires_grad)
  terms[1].backward();self.assertGreater(float(m.conv_logvar.weight.grad.abs().sum()),0.)
  self.assertGreater(float(m.conv_logvar.bias.grad.abs().sum()),0.)
  verify_frozen(m,opt,snap)
 def test_real_mean_and_fixed_sigma_outputs_unchanged_after_updates(self):
  m,opt,b=self.fixture();protect=protect_pen_variance_weights(m,opt);snap=configure_arm(m,opt,'posterior_only')
  batches={'a':b};splits={k:['a'] for k in ('train_probe','dev','held_out','named')};records={'a':dict(text='ab',writer_id='1')};std=source_scales(m,batches,['a'])
  with tempfile.TemporaryDirectory() as tmp:
   old=evaluate(m,batches,records,['a','b','c'],splits,std,tmp,0,draws=2,save_trajectories=True)
   for _ in range(3):
    opt.zero_grad(set_to_none=False);terms=fast_terms(m,*b[:2],torch.randn(1,48,4),include_kl=True)
    (terms*torch.tensor([1.,.1,.02,1e-6])).sum().backward();opt.step()
   new=evaluate(m,batches,records,['a','b','c'],splits,std,tmp,3,draws=2,save_trajectories=True)
   self.assertTrue(assert_output_isolation(new,old));verify_frozen(m,opt,snap)
   for name in ('mu','z-0','z-1'):
    np.testing.assert_array_equal(np.load(Path(tmp)/'fixed_source_sigma/step-0/a'/f'{name}.npy'),np.load(Path(tmp)/'fixed_source_sigma/step-3/a'/f'{name}.npy'))
   self.assertNotEqual(new['own']['lines'],old['own']['lines'])
   self.assertTrue(torch.equal(m.conv_logvar.weight[protect[0]],protect[1]))
  protect[2].remove()
 def test_frozen_weight_and_moment_mutations_detected(self):
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only');p=next(m.encoder.parameters())
  with torch.no_grad():p.add_(1)
  with self.assertRaisesRegex(AssertionError,'weight changed'):verify_frozen(m,opt,snap)
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only');p=next(m.encoder.parameters());opt.state[p]['step'].add_(1)
  with self.assertRaisesRegex(AssertionError,'moment/step'):verify_frozen(m,opt,snap)
 def test_stale_zero_grad_is_not_safe_guard_detects(self):
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only');next(m.encoder.parameters()).grad=torch.zeros_like(next(m.encoder.parameters()))
  with self.assertRaisesRegex(AssertionError,'gradient present'):verify_frozen(m,opt,snap)
 def test_isolation_guard_rejects_mean_or_fixed_change(self):
  r={'own':{'lines':[{'sample_id':'a','mu':{'x':1.},'sampled':[]} ]},'fixed_source_sigma':{'lines':[]}}
  q=copy.deepcopy(r);q['own']['lines'][0]['mu']['x']=2.
  with self.assertRaisesRegex(AssertionError,'mean output'):assert_output_isolation(q,r)
  q=copy.deepcopy(r);q['fixed_source_sigma']['lines']=[1]
  with self.assertRaisesRegex(AssertionError,'fixed SOURCE'):assert_output_isolation(q,r)
 def test_posterior_diagnostics_accept_empty_body_group(self):
  m,opt,b=self.fixture();snap=configure_arm(m,opt,'posterior_only');before=torch.get_rng_state().clone()
  r=gradient_diagnostics(m,{'a':b},['a'])
  self.assertEqual(r['groups']['body']['parameter_count'],0)
  self.assertEqual(r['groups']['body']['norms'],{'geometry':0.,'kl':0.})
  self.assertEqual(r['groups']['posterior']['parameter_count'],2)
  self.assertGreater(r['groups']['posterior']['norms']['geometry'],0.)
  self.assertTrue(torch.equal(before,torch.get_rng_state()));verify_frozen(m,opt,snap)
 def test_report_rejects_freeze_contract_or_rng_mismatch(self):
  from test_kl_tradeoff import TradeoffReportTests
  from iam_tools.report_posterior_isolation import pairing
  c,r,l=TradeoffReportTests().fixture()
  c={a:dict(c['kl1e-6']) for a in ARMS};r={a:dict(r['kl1e-6']) for a in ARMS};l={a:copy.deepcopy(l['kl1e-6']) for a in ARMS}
  for a in ARMS:
   c[a]['freeze_policy']='entire codec except conv_logvar' if a=='posterior_only' else 'source body+posterior groups'
   r[a]['frozen_state_invariance']={'frozen_weights_unchanged':True,'frozen_moments_steps_unchanged':True,'frozen_parameters':3}
   r[a]['posterior_output_isolation']=a=='posterior_only'
  self.assertTrue(all(pairing(c,r,l).values()))
  c['posterior_only']['freeze_policy']='transformer only'
  with self.assertRaisesRegex(ValueError,'freeze/isolation'):pairing(c,r,l)
  c['posterior_only']['freeze_policy']='entire codec except conv_logvar';l['posterior_only'][0]['noise_sha256']='different'
  with self.assertRaisesRegex(ValueError,'paired/immutable'):pairing(c,r,l)
 def test_saved_array_invariance_detects_change_not_hidden_by_metrics(self):
  from iam_tools.report_posterior_isolation import trajectory_invariance
  with tempfile.TemporaryDirectory() as t:
   for step in (0,1):
    for policy,kinds in [('own',['mu']),('fixed_source_sigma',['mu','z-0'])]:
     for k in kinds:
      q=Path(t)/policy/f'step-{step}'/'a';q.mkdir(parents=True,exist_ok=True);np.save(q/(k+'.npy'),np.zeros((3,5),np.float32))
   self.assertEqual(trajectory_invariance(t,[0,1],['a'],draws=1)['compared_trajectory_pairs'],3)
   np.save(Path(t)/'fixed_source_sigma/step-1/a/z-0.npy',np.ones((3,5),np.float32))
   with self.assertRaisesRegex(AssertionError,'trajectory changed'):trajectory_invariance(t,[0,1],['a'],draws=1)
 def test_serialized_checkpoint_isolation_checks_body_moments(self):
  from iam_tools.report_posterior_isolation import checkpoint_isolation
  source={'model_state_dict':{'encoder.weight':torch.ones(2),'conv_logvar.weight':torch.zeros(2)},'optimizer_state_dict':{'param_groups':[dict(name='body',params=[0],lr=1e-7)],'state':{0:{'step':torch.tensor(300.),'exp_avg':torch.ones(2)}}}}
  new=copy.deepcopy(source);new['model_state_dict']['conv_logvar.weight'].add_(1)
  self.assertTrue(checkpoint_isolation(source,new)['serialized_geometry_states_equal'])
  new['optimizer_state_dict']['state'][0]['step'].add_(1)
  with self.assertRaisesRegex(AssertionError,'moments/steps'):checkpoint_isolation(source,new)
  new=copy.deepcopy(source);new['model_state_dict']['encoder.weight'].add_(1)
  with self.assertRaisesRegex(AssertionError,'codec state changed'):checkpoint_isolation(source,new)
 def test_unknown_arm_rejected(self):
  m,opt,b=self.fixture()
  with self.assertRaises(ValueError):configure_arm(m,opt,'anything')

if __name__=='__main__':unittest.main()
