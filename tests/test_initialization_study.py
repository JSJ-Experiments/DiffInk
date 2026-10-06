import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch,yaml
root=Path(__file__).resolve().parents[1];repo=root/'third_party/DiffInk' if (root/'third_party/DiffInk').is_dir() else root
sys.path.insert(0,str(repo))
from model.vae import VAE
from iam_tools.initialization_study import optimizer_groups

class InitializationStudyTests(unittest.TestCase):
 def model(self):
  cfg=yaml.safe_load((repo/'configs/engineering_english.yaml').read_text())
  cfg.update(hidden_dims=[16,24,48],latent_dim=48,decoder_dims=[48,32,128],trans_hidden_dim=32,
             trans_num_layers=1,ocr_hidden_dim=16,ocr_num_heads=4,ocr_num_layers=1,
             num_text_embedding=5,num_writer=8,style_classifier_dim=48)
  return VAE(SimpleNamespace(**cfg))
 def test_groups_are_complete_unique_and_auxiliaries_frozen(self):
  m=self.model()
  for mode in ['uniform','scaled_readout','frozen_readout','protected_noise']:
   groups=optimizer_groups(m,mode);included=[p for g in groups for p in g['params']]
   self.assertEqual(len(included),len(set(map(id,included))))
   self.assertEqual(set(map(id,included)),{id(p) for p in m.parameters() if p.requires_grad})
   self.assertFalse(any(p.requires_grad for p in m.ocr_model.parameters()))
   self.assertFalse(any(p.requires_grad for p in m.style_classifier.parameters()))
   self.assertEqual(any(p.requires_grad for p in m.transformer_decoder.parameters()),mode not in ['frozen_readout','protected_noise'])
   rates={g['name']:g['lr'] for g in groups}
   if mode=='scaled_readout':self.assertAlmostEqual(rates['readout'],5e-5/512)
   if mode=='protected_noise':
    self.assertAlmostEqual(rates['body'],1e-7);self.assertAlmostEqual(rates['posterior'],1e-3)
 def test_invalid_modes_and_rates_do_not_mutate(self):
  m=self.model();before=[p.requires_grad for p in m.parameters()]
  for kw in [dict(mode='no'),dict(mode='uniform',lr=0),dict(mode='uniform',scale=float('nan')),dict(mode='uniform',scale=float('inf'))]:
   with self.assertRaises(ValueError):optimizer_groups(m,**kw)
  self.assertEqual(before,[p.requires_grad for p in m.parameters()])
 def test_real_optimizer_displacement_uses_group_lr(self):
  m=self.model();groups=optimizer_groups(m,'scaled_readout');opt=torch.optim.AdamW(groups,betas=(.9,.99),weight_decay=0)
  before=[p.detach().clone() for g in groups for p in g['params']]
  for g in groups:
   for p in g['params']:p.grad=torch.ones_like(p)
  opt.step();index=0
  for g in groups:
   for p in g['params']:
    torch.testing.assert_close(before[index]-p,torch.full_like(p,g['lr']),atol=3e-8,rtol=.05);index+=1

if __name__=='__main__':unittest.main()
