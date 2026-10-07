from pathlib import Path
import sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from iam_tools.ocr_recurrent import make_head
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.frozen_ocr_study import collate_latents

class RecurrentOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
    def fixture(self):
        c=frame_cache({'a':line(13),'b':line(25)},4);h=make_head(self.cfg,3,None);h.eval();return c,h
    def test_packed_batch_matches_unpadded_and_ignores_nan_padding(self):
        c,h=self.fixture();x,l,m=collate_latents(c,['a','b']);x=x.masked_fill(~m[:,None],float('nan'));out=h(x,padding_mask=~m)
        for j,sid in enumerate(['a','b']):
            a=c[sid];n=int(a['mask'].sum());one=h(a['mu'],padding_mask=~a['mask'])
            torch.testing.assert_close(out[:n,j],one[:,0],atol=2e-6,rtol=2e-6)
        self.assertTrue(torch.isfinite(out).all())
    def test_extra_padding_does_not_change_backward_recurrence(self):
        c,h=self.fixture();x=c['a']['mu'];m=c['a']['mask'];n=x.shape[-1];expected=h(x,padding_mask=~m)
        x=torch.cat((x,torch.full((1,48,9),float('nan'))),-1);m=torch.cat((m,torch.zeros(1,9,dtype=torch.bool)),1)
        torch.testing.assert_close(h(x,padding_mask=~m)[:n],expected,atol=1e-6,rtol=1e-6)
    def test_initialization_is_seeded_without_changing_global_rng(self):
        before=torch.get_rng_state().clone();a=make_head(self.cfg,3,None,seed=17);b=make_head(self.cfg,3,None,seed=17)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertEqual(float(a.output_fc.bias[0].detach()),0)
    def test_ctc_backward_finite_and_trainable(self):
        c,h=self.fixture();h.train();x,l,m=collate_latents(c,['a','b']);loss=h.get_ocr_loss(x,l,m);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(torch.isfinite(p.grad).all() for p in h.parameters()));self.assertGreater(float(h.rnn.weight_ih_l0.grad.abs().sum()),0)
    def test_unused_latent_noise_is_excluded(self):
        c,h=self.fixture();x=c['a']['mu'];expected=h(x);x=x.clone();x[:,20:]=float('nan');torch.testing.assert_close(h(x),expected,atol=0,rtol=0)
    def test_nonprefix_empty_and_attention_masks_rejected(self):
        c,h=self.fixture();x=c['a']['mu'];n=x.shape[-1];mask=torch.zeros(1,n,dtype=torch.bool);mask[0,1]=True
        with self.assertRaises(ValueError):h(x,padding_mask=mask)
        with self.assertRaises(ValueError):h(x,padding_mask=torch.ones_like(mask))
        with self.assertRaises(ValueError):h(x,attention_mask=torch.zeros(n,n,dtype=torch.bool))

class ReaderCacheGateTests(unittest.TestCase):
    def fixture(self):
        import tempfile,json
        from iam_tools.ocr_reader_cache import GATE_REL
        from iam_tools.ocr_pool_study import SOURCE,SHA
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name);d=root/GATE_REL;(d/'clean_control').mkdir(parents=True);pool=root/'pool';pool.mkdir();cfg={'same':'codec'};m={'records':{'a':{}}}
        (pool/'manifest.json').write_text('samepool');(d/'pool-manifest.json').write_text('samepool')
        (d/'clean_control/config.json').write_text(json.dumps(dict(cfg=cfg,source_rel=SOURCE,source_sha256=SHA)))
        (d/'codec-preflight.json').write_text(json.dumps(dict(passed=True,failed_checks=[],lines=[{'sample_id':'a'}])))
        return root,pool,m,cfg,d
    def test_valid_reused_gate_binds_source_config_pool_and_all_ids(self):
        from unittest.mock import patch
        from iam_tools.ocr_reader_cache import validate_gate
        from iam_tools.ocr_pool_study import SHA
        root,pool,m,cfg,d=self.fixture()
        with patch('iam_tools.ocr_reader_cache.file_sha',return_value=SHA):
            self.assertEqual(validate_gate(root,pool,m,cfg)[0],d)
            with self.assertRaises(ValueError):validate_gate(root,pool,m,{'different':'codec'})
            (pool/'manifest.json').write_text('changedpool')
            with self.assertRaises(ValueError):validate_gate(root,pool,m,cfg)
    def test_failed_or_incomplete_decoder_evidence_cannot_be_reused(self):
        import json
        from unittest.mock import patch
        from iam_tools.ocr_reader_cache import validate_gate
        from iam_tools.ocr_pool_study import SHA
        root,pool,m,cfg,d=self.fixture()
        for v in [dict(passed=False,failed_checks=[],lines=[{'sample_id':'a'}]),dict(passed=True,failed_checks=['failure'],lines=[{'sample_id':'a'}]),dict(passed=True,failed_checks=[],lines=[{'sample_id':'other'}])]:
            (d/'codec-preflight.json').write_text(json.dumps(v))
            with patch('iam_tools.ocr_reader_cache.file_sha',return_value=SHA):
                with self.assertRaises(ValueError):validate_gate(root,pool,m,cfg)
    def test_current_codec_sha_must_still_match(self):
        from unittest.mock import patch
        from iam_tools.ocr_reader_cache import validate_gate
        root,pool,m,cfg,d=self.fixture()
        with patch('iam_tools.ocr_reader_cache.file_sha',return_value='wrong'):
            with self.assertRaises(ValueError):validate_gate(root,pool,m,cfg)
    def test_recurrent_launcher_has_no_implicit_allocation(self):
        import ast
        s=(ROOT/'modal_ocr_recurrent.py').read_text();tree=ast.parse(s)
        self.assertIn('if not train:',s);self.assertIn('1000<=steps<=8000',s)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('modal_') for n in ast.walk(tree)))

class RecurrentReportTests(unittest.TestCase):
    def fixture(self):
        import copy
        from iam_tools.report_ocr_recurrent import ARMS,FIXED
        c={k:k for k in FIXED};c['max_updates']=2;c['architecture']='transformer';d=copy.deepcopy(c);d['architecture']='bigru'
        r=dict(sample_schedule_sha256='same',last_step=2,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True)
        log=[dict(step=i+1,sample_ids=[str(i)],lr=1e-4) for i in range(2)]
        return {'transformer':c,'bigru':d},{a:copy.deepcopy(r) for a in ARMS},{a:copy.deepcopy(log) for a in ARMS}
    def test_architecture_pairing_checks_exposure_and_common_contract(self):
        import copy
        from iam_tools.report_ocr_recurrent import pairing
        c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
        for k in ['feature_stats','seed','lr_drop_step','geometry_gate_mode']:
            bad=copy.deepcopy(c);bad['bigru'][k]='wrong'
            with self.assertRaises(ValueError):pairing(bad,r,l)
        bad=copy.deepcopy(l);bad['bigru'][0]['sample_ids']=['wrong']
        with self.assertRaises(ValueError):pairing(c,r,bad)
    def test_incomplete_or_unfrozen_runs_not_promoted(self):
        import copy
        from iam_tools.report_ocr_recurrent import pairing
        c,r,l=self.fixture()
        for k,v in [('last_step',1),('stop_reason','wall_limit'),('entire_codec_bitwise_unchanged',False)]:
            bad=copy.deepcopy(r);bad['bigru'][k]=v
            with self.assertRaises(ValueError):pairing(c,bad,l)
        bad=copy.deepcopy(c);bad['bigru']['architecture']='transformer'
        with self.assertRaises(ValueError):pairing(bad,r,l)

if __name__=='__main__':unittest.main()
