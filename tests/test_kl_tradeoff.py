from pathlib import Path
import sys,unittest,tempfile
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from iam_tools.kl_tradeoff import paired_latents,validate_budget,eligibility,selection,evaluation_scope,ARMS
from iam_tools.ocr_convergence import POOL_SHA

class TradeoffTests(unittest.TestCase):
 def test_budget_guards_before_gpu(self):
  validate_budget(200,POOL_SHA)
  for step,sha in [(True,POOL_SHA),(49,POOL_SHA),(301,POOL_SHA),(200,'')]:
   with self.assertRaises(ValueError):validate_budget(step,sha)
 def test_arms_small_and_known_prior_failure_not_repeated(self):
  self.assertEqual(ARMS,{'kl0':0.,'kl1e-7':1e-7,'kl1e-6':1e-6})
 def test_sigma_control_keeps_source_uncertainty_but_uses_current_mean(self):
  mu=torch.zeros(1,48,2);eps=[torch.ones_like(mu),-torch.ones_like(mu)];lv=torch.full_like(mu,2*np.log(.2));source=torch.full_like(mu,.5)
  own,fixed=paired_latents(mu,lv,source,eps)
  torch.testing.assert_close(own[1],mu+.2);torch.testing.assert_close(fixed[1],mu+.5)
  _,shifted=paired_latents(mu+7,lv-3,source,eps)
  torch.testing.assert_close(shifted[1]-7,fixed[1]);torch.testing.assert_close(shifted[2]-7,fixed[2])
  torch.testing.assert_close(own[0],fixed[0]);torch.testing.assert_close(fixed[0],mu)
 def test_invalid_sigma_or_epsilon_rejected(self):
  mu=torch.zeros(1,48,2)
  for std in [torch.zeros_like(mu),torch.full_like(mu,float('nan')),torch.ones(1,48,1),torch.ones_like(mu,dtype=torch.float64)]:
   with self.assertRaises(ValueError):paired_latents(mu,mu,std,[mu])
  with self.assertRaises(ValueError):paired_latents(mu,mu,torch.ones_like(mu),[mu[:,:,:1]])
 def test_all8_always_in_scope_and_not_added_to_trainprobe(self):
  from iam_tools.ocr_joint_study import LEGACY_EIGHT
  s=evaluation_scope({'splits':{'dev':['dev'],'held_out':['report']}},[str(i) for i in range(192)])
  self.assertEqual(s['named'],list(LEGACY_EIGHT));self.assertEqual(len(s['train_probe']),32)
  self.assertFalse(set(s['named'])&set(s['train_probe']))
 def test_both_policies_must_pass_not_just_own_sigma(self):
  from test_ocr_joint import JointStudyGuardTests
  # Reuse the exact public gate fixture, including per-draw pens/angle metrics.
  row=JointStudyGuardTests().fixture();import copy
  current={'own':copy.deepcopy(row),'fixed_source_sigma':copy.deepcopy(row)};ref=copy.deepcopy(current)
  ids=[r['sample_id'] for r in row['lines']]
  self.assertTrue(eligibility(current,ref,ids)['passed'])
  current['fixed_source_sigma']['lines'][0]['sampled'][0]['geometry']['x_rmse']=1.
  self.assertFalse(eligibility(current,ref,ids)['passed'])
 def test_selector_is_train_only_no_dev_or_report(self):
  stat={'groups':{'train_probe':{'kl_per_element':1.1},'dev':{'kl_per_element':-100}}}
  row={'own':{'groups':{'train_probe':{'sampled':{'x_rmse':.01,'y_rmse':.02}},'dev':{'sampled':{'x_rmse':0,'y_rmse':0}}}}}
  self.assertEqual(selection(stat,row),(1.1,.03))
 def test_real_cpu_evaluator_draws_and_pen_fields_match_same_sigma(self):
  from test_vae_ctc import small_config
  from iam_tools.identity_geometry_probe import initialize_identity_geometry
  from iam_tools.ocr_joint_adapter import PolyphaseGRUOCR
  from iam_tools.ocr_recurrent import make_head
  from iam_tools.ocr_pool_study import single_batch
  from iam_tools.kl_tradeoff import evaluate,source_scales
  from model.vae import VAE
  torch.set_num_threads(2);cfg=small_config();cfg.hidden_dims=[48,48,48];cfg.latent_dim=48;cfg.decoder_dims=[48,48,128];cfg.trans_hidden_dim=16;cfg.style_classifier_dim=48;cfg.model_input_scale=.01;cfg.trans_dropout=0.;cfg.use_decoder_padding_mask=True
  m=VAE(cfg).eval();initialize_identity_geometry(m);m.ocr_model=PolyphaseGRUOCR(make_head(dict(latent_dim=48,ocr_hidden_dim=16),4,None),None)
  p=np.zeros((25,5),dtype=np.float32);p[:,0]=np.linspace(0,100,25);p[:,1]=np.sin(np.linspace(0,3,25))*10;p[:,2]=1;p[-1,2]=0;p[-1,4]=1
  batches={'a':single_batch(p,'ab',['a','b','c'])};records={'a':{'text':'ab','writer_id':'1'}};splits={k:['a'] for k in ['train_probe','dev','held_out','named']}
  scales=source_scales(m,batches,['a']);before=torch.get_rng_state().clone()
  with tempfile.TemporaryDirectory() as tmp:
   r=evaluate(m,batches,records,['a','b','c'],splits,scales,tmp,0,draws=2,save_trajectories=True)
   self.assertEqual(r['own']['lines'],r['fixed_source_sigma']['lines'])
   for name in ['mu','z-0','z-1']:
    a=np.load(Path(tmp)/'own/step-0/a'/f'{name}.npy');b=np.load(Path(tmp)/'fixed_source_sigma/step-0/a'/f'{name}.npy');np.testing.assert_array_equal(a,b)
   self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(all(v.grad is None for v in m.parameters()))

class TradeoffReportTests(unittest.TestCase):
 def test_row_table_records_both_policies_and_all_scopes(self):
  from iam_tools.report_kl_tradeoff import aggregate_row_table
  groups={p:{k:{'mu':{'x_rmse':1.},'sampled':{'x_rmse':2.}} for k in ['train_probe','dev','held_out','named']} for p in ['own','fixed_source_sigma']}
  stats={k:{'kl_per_element':1.,'xy':{'std_mean':.1}} for k in ['train_probe','dev','held_out','named']}
  r=aggregate_row_table({'a':{'history':[{'step':100,'groups':groups,'posterior':stats}]}})
  self.assertEqual(len(r),16);self.assertEqual({v['policy'] for v in r},{'own','fixed_source_sigma'})
  self.assertTrue(all(v['step']==100 for v in r))
 def fixture(self):
  import copy
  from iam_tools.report_kl_tradeoff import pairing
  keys=('source_sha256','reader_sha256','pool_manifest_sha256','source_std_sha256','cfg','splits','train_ids','schedule_seed','schedule_sha256','noise_seed','optimizer_groups','max_updates','physical_batch','gradient_accumulation','mean_geometry_weight','sampled_geometry_weight','pen_weight','ctc_weight','style_weight','gmm_weight','dropout','target_delta_weight')
  cfg={k:0 for k in keys};cfg['max_updates']=2
  configs={a:dict(cfg,kl_weight=w) for a,w in ARMS.items()}
  results={a:dict(kl_weight=w,last_step=2,final_rng_cpu_sha256='cpu',final_rng_cuda_sha256='cuda',reader_weights_unchanged=True,readout_style_pen_variance_protected=True,source_unchanged=True) for a,w in ARMS.items()}
  logs={a:[dict(sample_ids=['a'],noise_sha256='eps'),dict(sample_ids=['b'],noise_sha256='eps2')] for a in ARMS}
  return configs,results,logs
 def test_report_requires_matching_noise_and_exact_arm_weights(self):
  from iam_tools.report_kl_tradeoff import pairing
  c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
  l['kl1e-6'][1]['noise_sha256']='different'
  with self.assertRaises(ValueError):pairing(c,r,l)
  c,r,l=self.fixture();c['kl1e-7']['kl_weight']=1e-5
  with self.assertRaises(ValueError):pairing(c,r,l)
 def test_early_stop_can_be_reported_but_not_called_full_paired_budget(self):
  from iam_tools.report_kl_tradeoff import pairing
  c,r,l=self.fixture();r['kl1e-6']['last_step']=1;l['kl1e-6']=l['kl1e-6'][:1]
  q=pairing(c,r,l);self.assertTrue(q['matched_prefix_batches_noise']);self.assertFalse(q['full_matched_budget'])
 def test_hdf5_test_writer_rejected_even_if_point_hash_correct(self):
  from unittest.mock import patch
  import json,hashlib,h5py
  from iam_tools.kl_tradeoff import load_batches
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);legacy=root/'diffink/iam_overfit';legacy.mkdir(parents=True);(legacy/'manifest.json').write_text('{}');pool=root/'pool';pool.mkdir()
   p=np.zeros((8,5),dtype=np.float32);p[:,2]=1;p[-1,2]=0;p[-1,4]=1
   with h5py.File(pool/'lines.h5','w') as hf:hf.create_group('a').create_dataset('point_seq',data=p)
   with h5py.File(legacy/'tiny_train.h5','w'):pass
   m={'records':{'a':{'points_sha256':hashlib.sha256(p.tobytes()).hexdigest(),'text':'a','writer_id':'test'}},'test_writers':['test']}
   with patch('iam_tools.kl_tradeoff.MANIFEST_SHA',hashlib.sha256((legacy/'manifest.json').read_bytes()).hexdigest()):
    with self.assertRaisesRegex(ValueError,'test writer leakage'):load_batches(root,pool,m,['a'],['a'],'cpu')

 def test_dedicated_cpu_rebuild_preserves_checkpoint_and_contract(self):
  from unittest.mock import patch
  from types import SimpleNamespace
  from test_vae_ctc import small_config
  from model.vae import VAE
  from iam_tools.report_kl_tradeoff import rebuild
  from iam_tools.kl_tradeoff import SOURCE,SHA
  from iam_tools.ocr_joint_adapter import READER_SHA,CONTRACT,PolyphaseGRUOCR
  from iam_tools.ocr_recurrent import make_head
  cfg=small_config();m=VAE(cfg);codec={'latent_dim':cfg.latent_dim,'ocr_hidden_dim':16}
  head=PolyphaseGRUOCR(make_head(codec,4,None),None)
  saved=dict(model_state_dict=m.state_dict(),source_sha256=SHA,frozen_reader_sha256=READER_SHA,continuation_updates=100,config=dict(cfg=codec,research_ocr_contract=CONTRACT,model_input_scale=.01,trans_dropout=0.,use_decoder_padding_mask=True))
  class Dataset:
   def __init__(self):self.hf=SimpleNamespace(close=lambda:None)
  with tempfile.TemporaryDirectory() as t:
   torch.save(saved,Path(t)/'checkpoint-last.pt')
   with patch('iam_tools.report_kl_tradeoff.setup',return_value=(m,Dataset(),Dataset(),{},Path(t))),patch('iam_tools.report_kl_tradeoff.load_reader',return_value=(head,{})):
    actual,state=rebuild(t,'checkpoint-last.pt',t,t,{'cfg':codec})
    self.assertEqual(actual.config.model_input_scale,.01);self.assertFalse(actual.training)
    self.assertTrue(all(not p.requires_grad for p in actual.parameters()));self.assertIs(actual.ocr_model,head)
    self.assertEqual(state['continuation_updates'],100)
    with self.assertRaisesRegex(ValueError,'contract drift'):rebuild(t,'checkpoint-last.pt',t,t,{'cfg':{}})

if __name__=='__main__':unittest.main()
