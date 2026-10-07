import sys,copy,unittest
from pathlib import Path
import torch
root=Path(__file__).resolve().parents[1];repo=root/'third_party/DiffInk' if (root/'third_party/DiffInk').is_dir() else root
sys.path.insert(0,str(repo))
from iam_tools.codec_kl_study import restore_optimizer,posterior_stats,packed_std_to_points,protect_pen_variance_weights,candidate_valid
from iam_tools.initialization_study import optimizer_groups
from iam_tools.writer_expansion import training_schedule

class CodecKLTests(unittest.TestCase):
 def model(self):
  class Tiny(torch.nn.Module):
   def __init__(self):
    super().__init__();self.encoder=torch.nn.Linear(2,2);self.conv_logvar=torch.nn.Linear(2,2)
    self.transformer_decoder=torch.nn.Linear(2,2);self.ocr_model=torch.nn.Linear(2,2);self.style_classifier=torch.nn.Linear(2,2)
  return Tiny()
 def test_restore_exact_adam_moments_steps_groups_and_next_update(self):
  torch.manual_seed(54);a=self.model();o=torch.optim.AdamW(optimizer_groups(a,'protected_noise'),betas=(.9,.99),weight_decay=0)
  for p in a.parameters():
   if p.requires_grad:p.grad=torch.randn_like(p)
  o.step();b=copy.deepcopy(a);resumed=restore_optimizer(b,{'optimizer_state_dict':o.state_dict()})
  for ga,gb in zip(o.param_groups,resumed.param_groups):
   self.assertEqual(ga['name'],gb['name']);self.assertEqual(ga['lr'],gb['lr'])
   for pa,pb in zip(ga['params'],gb['params']):
    for key in ('step','exp_avg','exp_avg_sq'):torch.testing.assert_close(o.state[pa][key],resumed.state[pb][key],rtol=0,atol=0)
    pa.grad=torch.randn_like(pa);pb.grad=pa.grad.clone()
  o.step();resumed.step()
  for pa,pb in zip(a.parameters(),b.parameters()):torch.testing.assert_close(pa,pb,atol=0,rtol=0)
 def test_reordered_optimizer_groups_rejected(self):
  a=self.model();o=torch.optim.AdamW(optimizer_groups(a,'protected_noise'),betas=(.9,.99),weight_decay=0)
  s=o.state_dict();s['param_groups'].reverse()
  with self.assertRaises(ValueError):restore_optimizer(a,{'optimizer_state_dict':s})
 def test_posterior_diagnostics_separate_active_unused_channels_and_preserve_rng(self):
  from unittest.mock import patch
  mu=torch.zeros(1,48,2);mu[:,:40]=2.;lv=torch.zeros_like(mu);lv[:,:40]=-4.
  lm=torch.ones(1,2,dtype=torch.bool);raw=torch.zeros(1,5,16);mask=torch.ones(1,16,dtype=torch.bool)
  from model.vae import VAE
  class Diagnostic:
   def kl_divergence_new(self,*args):return VAE.kl_divergence_new(self,*args)
  before=torch.get_rng_state()
  with patch('iam_tools.codec_kl_study.encoded',return_value=(None,mu,lv,lm)):
   row=posterior_stats(Diagnostic(),{'line':(raw,mask,None)},{'train':['line']})
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
  line=row['lines']['line'];group=row['groups']['train']
  self.assertAlmostEqual(sum(line[k]['kl_contribution_per_total_element'] for k in ('xy','pen','unused')),line['kl_per_element'],places=6)
  self.assertEqual(line['unused']['std_median'],1.)
  self.assertEqual(line['unused']['mu_rms'],0.)
  self.assertEqual(group['kl_per_element'],line['kl_per_element'])
  self.assertAlmostEqual(line['xy']['std_median'],.13533528)
 def test_polyphase_std_inspection_preserves_point_phase_field_order(self):
  points=torch.arange(24*5).reshape(24,5).float()
  packed=points.reshape(3,40).T[None]
  expanded=torch.cat((packed,torch.ones(1,8,3)),dim=1)
  torch.testing.assert_close(packed_std_to_points(expanded),points)
  with self.assertRaises(ValueError):packed_std_to_points(torch.ones(2,48,3))
 def test_pen_weight_freeze_blocks_restored_adam_momentum_not_just_gradient(self):
  from types import SimpleNamespace
  layer=torch.nn.Linear(48,48);model=SimpleNamespace(conv_logvar=layer)
  opt=torch.optim.AdamW(layer.parameters(),lr=.01,weight_decay=0)
  for p in layer.parameters():p.grad=torch.ones_like(p)
  opt.step();original=layer.weight.detach().clone();bias=layer.bias.detach().clone()
  rows,before,hook,reset=protect_pen_variance_weights(model,opt)
  self.assertGreater(reset['exp_avg'],0.)
  for _ in range(2):
   opt.zero_grad();layer(torch.ones(1,48)).sum().backward();opt.step()
  torch.testing.assert_close(layer.weight[rows],before,atol=0,rtol=0)
  self.assertTrue((layer.weight[~rows]!=original[~rows]).any());self.assertTrue((layer.bias!=bias).any());hook.remove()
  with self.assertRaises(ValueError):protect_pen_variance_weights(model,torch.optim.AdamW(layer.parameters(),weight_decay=.01))
 def test_selection_rejects_false_eoc_even_with_perfect_pen_up_f1(self):
  g=dict(all_final_eoc_correct=True,false_internal_eoc=0,mu_min_pen_f1=1.,sampled_min_pen_f1=1.)
  self.assertTrue(candidate_valid({'groups':{'train':g,'held_out':dict(false_internal_eoc=10)}}))
  for change in [dict(false_internal_eoc=1),dict(all_final_eoc_correct=False),dict(sampled_min_pen_f1=.99)]:
   self.assertFalse(candidate_valid({'groups':{'train':dict(g,**change)}}))
 def test_schedule_prefix_skip_preserves_continuation(self):
  ids=list('abcdefghijkl');whole=list(training_schedule(ids,203));before=list(training_schedule(ids,200))
  self.assertEqual(whole[:200],before);self.assertEqual(len(whole[200:]),3)
  self.assertNotEqual(whole[200:],whole[:3])

if __name__=='__main__':unittest.main()
