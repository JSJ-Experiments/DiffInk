from pathlib import Path
import sys,copy,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.ocr_context_features import make_head
from iam_tools.ocr_dropout import set_dropout
from iam_tools.ocr_pool_expansion import state_digest

class DropoutOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
    def test_eval_weights_rng_and_optimizer_unchanged(self):
        h=make_head(self.cfg,3,'relative_scaled',points_per_frame=4);c=frame_cache({'a':line(13)},4)['a'];h.eval();expected=h(c['mu']);before=state_digest(h.state_dict());rng=torch.get_rng_state().clone()
        o=torch.optim.AdamW(h.parameters());opt=state_digest(o.state_dict());proof=set_dropout(h,.3)
        self.assertEqual(len(proof['sites']),4);self.assertEqual(before,state_digest(h.state_dict()));self.assertEqual(opt,state_digest(o.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(h(c['mu']),expected,atol=0,rtol=0)
        for m in h.modules():
            if isinstance(m,torch.nn.Dropout):self.assertEqual(m.p,.3)
            if isinstance(m,torch.nn.MultiheadAttention):self.assertEqual(m.dropout,.3)
    def test_dropout_changes_training_not_random_exposure_count(self):
        h=make_head(self.cfg,3,'relative_scaled',points_per_frame=4);weights=copy.deepcopy(h.state_dict());c=frame_cache({'a':line(13)},4)['a'];rng=torch.get_rng_state().clone();ends=[];losses=[]
        for p in [.1,.3]:
            head=make_head(self.cfg,3,'relative_scaled',points_per_frame=4);head.load_state_dict(weights);set_dropout(head,p);torch.set_rng_state(rng)
            loss=.5*head.get_ocr_loss(c['mu'],c['labels'],c['mask'])+.5*head.get_ocr_loss(c['mu'],c['labels'],c['mask']);loss.backward();self.assertTrue(torch.isfinite(loss));ends.append(torch.get_rng_state().clone());losses.append(float(loss.detach()))
        self.assertTrue(torch.equal(*ends));self.assertNotEqual(*losses);torch.set_rng_state(rng)
    def test_invalid_rate_or_no_dropout_model_rejected(self):
        h=make_head(self.cfg,3,'relative_scaled',points_per_frame=4)
        for p in [True,-1,1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):set_dropout(h,p)
        with self.assertRaises(ValueError):set_dropout(torch.nn.Linear(2,3),.3)
    def test_guarded_launcher_and_pinned_selected_parent(self):
        import ast
        s=(ROOT/'modal_ocr_dropout.py').read_text();ast.parse(s);self.assertIn('if not train:',s);self.assertIn('1000<=steps<=6000',s)
        from iam_tools.ocr_dropout_study import PARENT,PARENT_SHA,PARENT_UPDATES
        self.assertEqual(PARENT_UPDATES,10000);self.assertIn('/clean_control/head-best.pt',PARENT);self.assertEqual(PARENT_SHA,'f64d79e1ccfbfedaaafdaa365ba0458d9b0b28d2f7b99bca43d56cd86fe14998')

class DropoutReportTests(unittest.TestCase):
    def fixture(self):
        from iam_tools.report_ocr_dropout import FIXED,ARMS
        c={k:k for k in FIXED};c['max_updates']=2;c['dropout']=.1;c['objective']='0.5clean+0.5clean;two identical forwards in BOTH arms';d=copy.deepcopy(c);d['dropout']=.3
        r=dict(initial_state={'same':'head,moments,rng'},sample_schedule_sha256='same',final_dropout_rng_sha256='same',last_step=2,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True)
        logs=[dict(step=i+1,sample_ids=[str(i)],lr=1e-4) for i in range(2)]
        return dict(zip(ARMS,[c,d])),{a:copy.deepcopy(r) for a in ARMS},{a:copy.deepcopy(logs) for a in ARMS}
    def test_dropout_pair_contract_and_rng_exposure_guards(self):
        from iam_tools.report_ocr_dropout import pairing
        c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
        for k in ['feature_stats','dropout_sites','schedule_skip','parent_sha256']:
            bad=copy.deepcopy(c);bad['dropout_03'][k]='wrong'
            with self.assertRaises(ValueError):pairing(bad,r,l)
        bad=copy.deepcopy(r);bad['dropout_03']['final_dropout_rng_sha256']='wrong'
        with self.assertRaises(ValueError):pairing(c,bad,l)
        bad=copy.deepcopy(l);bad['dropout_03'][0]['sample_ids']=['other']
        with self.assertRaises(ValueError):pairing(c,r,bad)
    def test_only_predeclared_dropout_rates_accepted(self):
        from iam_tools.report_ocr_dropout import pairing
        c,r,l=self.fixture();bad=copy.deepcopy(c);bad['dropout_03']['dropout']=.5
        with self.assertRaises(ValueError):pairing(bad,r,l)

if __name__=='__main__':unittest.main()
