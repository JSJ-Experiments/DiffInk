from pathlib import Path
import copy, sys, unittest
import torch
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from iam_tools.ocr_context_features import make_head as original_head,transform
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.frozen_ocr_study import collate_latents
from iam_tools.ocr_local_context import make_head,attach_context,LocalContext
from iam_tools.ocr_pool_expansion import state_digest

class OCRLocalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
    def fixture(self):
        h=original_head(self.cfg,3,'relative_scaled',points_per_frame=4)
        o=torch.optim.AdamW(h.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
        c=frame_cache({'a':line(13),'b':line(19)},4);mu,l,m=collate_latents(c,['a','b'])
        h.get_ocr_loss(mu,l,m).backward();o.step();h.eval()
        return h,o,mu,l,m
    def test_attach_preserves_parent_function_optimizer_and_rng(self):
        h,o,x,l,m=self.fixture();weights=copy.deepcopy(h.state_dict());state=copy.deepcopy(o.state_dict());expected=h(x,padding_mask=~m)
        ends=[]
        for gate in (False,True):
            head=make_head(self.cfg,3,None,context=gate,attach=False);head.load_state_dict(weights)
            opt=torch.optim.AdamW(head.parameters());opt.load_state_dict(state);rng=torch.get_rng_state().clone();proof=attach_context(head,opt)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(proof['extra_parameters'],7744)
            after=opt.state_dict();after['param_groups'].pop();self.assertEqual(state_digest(state),state_digest(after))
            head.eval();torch.testing.assert_close(head(x,padding_mask=~m),expected,atol=0,rtol=0);ends.append(proof['effective_head_sha256'])
        self.assertEqual(*ends)
    def test_new_branch_is_trainable_and_dropout_stream_matches(self):
        h,o,x,l,m=self.fixture();ends=[];rng=torch.get_rng_state().clone()
        for gate in (False,True):
            head=make_head(self.cfg,3,None,context=gate,attach=False);head.load_state_dict(h.state_dict());attach_context(head)
            torch.set_rng_state(rng);loss=.5*head.get_ocr_loss(x,l,m)+.5*head.get_ocr_loss(x,l,m);loss.backward();ends.append(torch.get_rng_state().clone())
            self.assertEqual(bool(torch.count_nonzero(head.local_context.out.weight.grad)),gate)
            self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(head.local_context.out.weight.grad).all())
        self.assertTrue(torch.equal(*ends));torch.set_rng_state(rng)
    def test_padding_and_nan_cannot_leak_through_trained_context(self):
        h,o,x,l,m=self.fixture();head=make_head(self.cfg,3,None,context=True,attach=False);head.load_state_dict(h.state_dict());attach_context(head);head.eval()
        with torch.no_grad():head.local_context.out.weight.normal_(std=.1)
        a=x[:1,:,:4];mask=m[:1,:4];base=head(a,padding_mask=~mask)
        padded=torch.cat((a,torch.full((1,48,5),float('nan'))),2);pm=torch.cat((mask,torch.zeros(1,5,dtype=torch.bool)),1)
        torch.testing.assert_close(head(padded,padding_mask=~pm)[:4],base,atol=1e-6,rtol=1e-6)
        self.assertTrue(torch.isfinite(head(padded,padding_mask=~pm)).all())
    def test_context_uses_neighbors_but_never_changes_input(self):
        h,o,x,l,m=self.fixture();head=make_head(self.cfg,3,None,context=True,attach=False);head.load_state_dict(h.state_dict());attach_context(head);before=x.clone()
        with torch.no_grad():head.local_context.out.weight.fill_(.1)
        f=transform(x,m,'relative_scaled',None,4);r=head.local_context(f,m);changed=f.clone();changed[0,0,1]+=1
        s=head.local_context(changed,m)
        self.assertGreater(float((r[0,:,2]-s[0,:,2]).detach().abs().sum()),0);self.assertTrue(torch.equal(x,before))
        self.assertEqual(float(r.detach().masked_select(~m[:,None].expand_as(r)).abs().sum()),0)
    def test_selected_head_strict_reload_retains_gate_contract(self):
        for gate in (False,True):
            a=make_head(self.cfg,3,None,context=gate);b=make_head(self.cfg,3,None,context=gate)
            with torch.no_grad():a.local_context.out.weight.fill_(.15)
            b.load_state_dict(a.state_dict(),strict=True)
            c=frame_cache({'a':line(13)},4)['a'];a.eval();b.eval()
            torch.testing.assert_close(a(c['mu']),b(c['mu']),atol=0,rtol=0)
    def test_duplicate_attachment_rejected(self):
        h=make_head(self.cfg,3,None)
        with self.assertRaises(ValueError):attach_context(h)
    def test_launcher_guarded_and_self_contained(self):
        import ast
        source=(ROOT/'modal_ocr_local.py').read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('modal_') for n in ast.walk(tree)))
        self.assertIn('if not train:',source);self.assertIn('1000<=steps<=6000',source)

class OCRLocalReportTests(unittest.TestCase):
    def fixture(self):
        from iam_tools.report_ocr_local import FIXED,ARMS
        c={k:k for k in FIXED};c['max_updates']=2;c['local_context']=False;c['objective']='0.5clean+0.5clean;two identical forwards in BOTH arms';d=copy.deepcopy(c);d['local_context']=True
        r=dict(initial_state={'same':'head,parentmoments,newbranch'},sample_schedule_sha256='same',final_dropout_rng_sha256='same',last_step=2,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True)
        logs=[dict(step=i+1,sample_ids=[str(i)],lr=1e-4) for i in range(2)]
        return dict(zip(ARMS,[c,d])),{a:copy.deepcopy(r) for a in ARMS},{a:copy.deepcopy(logs) for a in ARMS}
    def test_pairing_rejects_state_rng_budget_or_exposure_drift(self):
        from iam_tools.report_ocr_local import pairing
        c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
        for key,v in [('initial_state',{}),('last_step',1),('entire_codec_bitwise_unchanged',False),('final_dropout_rng_sha256','wrong')]:
            bad=copy.deepcopy(r);bad['local_context'][key]=v
            with self.assertRaises(ValueError):pairing(c,bad,l)
        bad=copy.deepcopy(l);bad['local_context'][0]['sample_ids']=['wrong']
        with self.assertRaises(ValueError):pairing(c,r,bad)
    def test_config_intervention_must_be_gate_only(self):
        from iam_tools.report_ocr_local import pairing
        c,r,l=self.fixture()
        for key in ['local_definition','feature_stats','parent_sha256','objective']:
            bad=copy.deepcopy(c);bad['local_context'][key]='changed'
            with self.assertRaises(ValueError):pairing(bad,r,l)
        bad=copy.deepcopy(c);bad['local_context']['local_context']=False
        with self.assertRaises(ValueError):pairing(bad,r,l)

if __name__=='__main__':unittest.main()
