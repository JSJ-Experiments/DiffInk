from copy import deepcopy
from pathlib import Path
import unittest
import json
import tempfile
import torch
import yaml
from iam_tools.autopsy import check_geometry_config
from iam_tools.autopsy_resume import restore_optimizer,validate_resume,clipping_summary

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT

class ResumeTests(unittest.TestCase):
    def test_resume_only_changes_lr_and_metadata(self):
        cfg=yaml.safe_load((REPO/'configs/vae_iam_autopsy_resume.yaml').read_text());check_geometry_config(cfg)
        original=yaml.safe_load((REPO/'configs/vae_iam_autopsy_geometry.yaml').read_text())
        source={'step':1000,'config':original,'sample_sha256':'sample',
                'model_state_dict':{},'optimizer_state_dict':{'state':{0:{'step':torch.tensor(1000.)}}}}
        digest=cfg['resume_checkpoint_sha256'];validate_resume(source,cfg,digest,'sample')
        for key,value in [('model_input_scale',.02),('grad_clip',20),('weight_decay',0),('pen_weight',1),('sample_id','other')]:
            changed=deepcopy(cfg);changed[key]=value
            with self.assertRaises(ValueError):validate_resume(source,changed,digest,'sample')
        for bad_digest,bad_sample in [('other','sample'),(digest,'other')]:
            with self.assertRaises(ValueError):validate_resume(source,cfg,bad_digest,bad_sample)
        bad=deepcopy(source);del bad['optimizer_state_dict']
        with self.assertRaises((ValueError,KeyError)):validate_resume(bad,cfg,digest,'sample')
        bad=deepcopy(source);bad['optimizer_state_dict']['state'][0]['step']=torch.tensor(0.)
        with self.assertRaises(ValueError):validate_resume(bad,cfg,digest,'sample')

    def test_optimizer_restores_moments_then_overrides_old_lr(self):
        # Fabricated Adam state, no optimizer step or model training needed.
        parameter=torch.nn.Parameter(torch.ones(2))
        old=torch.optim.AdamW([parameter],lr=1.5e-4,betas=(.9,.99),weight_decay=.0001)
        old.state[parameter]={'step':torch.tensor(1000.),'exp_avg':torch.tensor([2.,3.]),'exp_avg_sq':torch.tensor([4.,5.])}
        state=deepcopy(old.state_dict())
        new_parameter=torch.nn.Parameter(torch.zeros(2))
        new=torch.optim.AdamW([new_parameter],lr=1e-5,betas=(.9,.99),weight_decay=.0001)
        restored=restore_optimizer(new,{'optimizer_state_dict':state},1e-5)
        self.assertEqual(new.param_groups[0]['lr'],1e-5)
        self.assertEqual(float(new.state[new_parameter]['step']),1000)
        torch.testing.assert_close(new.state[new_parameter]['exp_avg'],state['state'][0]['exp_avg'])
        torch.testing.assert_close(new.state[new_parameter]['exp_avg_sq'],state['state'][0]['exp_avg_sq'])
        self.assertEqual(restored['state_entries'],1)
        self.assertTrue(torch.equal(new_parameter,torch.zeros(2)))

    def test_report_rejects_missing_continuation_updates(self):
        from iam_tools.report_autopsy import report
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'result.json').write_text(json.dumps({'start_step':1000,'steps':2000,'additional_steps':1000}))
            (root/'metrics.jsonl').write_text('')
            (root/'fixed_metrics.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'incomplete'):report(root)

    def test_clipping_fraction(self):
        self.assertEqual(clipping_summary([{'gradient_norm':x} for x in [2,10,20,40]],10)['fraction'],.5)
        self.assertIsNone(clipping_summary([],10)['fraction'])

if __name__=='__main__':unittest.main()
