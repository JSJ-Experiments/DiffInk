from pathlib import Path
import copy,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.ocr_context_features import make_head as original_head,transform,unpack
from iam_tools.ocr_spatial_features import normalized_x,features,make_head,zero_unused_columns
from iam_tools.frozen_ocr_study import collate_latents
from iam_tools.ocr_pool_expansion import state_digest

class OCRSpatialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_real_min_max_phase_order_padding_and_final_eoc(self):
        c=frame_cache({'a':line(13)},4)['a'];r=normalized_x(c['mu'],c['mask']);f,real=unpack(c['mu'],c['mask'],4)
        expected=f[:,:,0]/(.04*12);torch.testing.assert_close(r[real],expected[real],atol=1e-7,rtol=1e-7);self.assertEqual(float(r[~real].abs().sum()),0.)
        dirty=c['mu'].clone();dirty[:,20:]=float('nan');dirty=torch.cat((dirty,torch.full((1,48,2),float('nan'))),2);mask=torch.cat((c['mask'],torch.zeros(1,2,dtype=torch.bool)),1)
        self.assertTrue(torch.isfinite(normalized_x(dirty,mask)).all());self.assertEqual(float(normalized_x(dirty,mask)[:,:,-2:].abs().sum()),0.)
    def test_translation_positive_scale_invariance_and_backward_preserved(self):
        c=frame_cache({'a':line(13)},4)['a'];mu=c['mu'].clone();f=mu[:,:20].reshape(1,4,5,-1);f[0,2,0,1]=.01
        before=normalized_x(mu,c['mask']);shift=mu.clone();shift[:,:20].reshape(1,4,5,-1)[:,:,0]*=1.7;shift[:,:20].reshape(1,4,5,-1)[:,:,0]+=3.
        torch.testing.assert_close(before,normalized_x(shift,c['mask']),atol=7e-7,rtol=2e-6)
        # Acquisition indexing is not sorted: original sixth→seventh point goes backward.
        seq=before.transpose(1,2).reshape(-1);self.assertLess(float(seq[6]),float(seq[5]))
    def test_constant_x_floor_and_gradient_only_real_used_fields(self):
        c=frame_cache({'a':line(13)},4)['a'];mu=c['mu'].clone();mu[:,:20].reshape(1,4,5,-1)[:,:,0]=3.
        self.assertEqual(float(normalized_x(mu,c['mask']).abs().sum()),0.)
        mu=c['mu'].clone().requires_grad_(True);normalized_x(mu,c['mask']).sum().backward();self.assertTrue(torch.isfinite(mu.grad).all());self.assertEqual(float(mu.grad[:,20:].abs().sum()),0.)
    def test_spatial_fields_do_not_change_local_features_or_pen(self):
        c=frame_cache({'a':line(13)},4)['a'];base=transform(c['mu'],c['mask'],'relative_scaled',None,4);aug=features(c['mu'],c['mask'],None,True)
        self.assertTrue(torch.equal(base[:,:20],aug[:,:20]));self.assertTrue(torch.equal(aug[:,20:24],normalized_x(c['mu'],c['mask'])));self.assertEqual(float(aug[:,24:].abs().sum()),0.)
        self.assertTrue(torch.equal(base,features(c['mu'],c['mask'],None,False)))
    def trained_parent(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1);h=original_head(cfg,3,'relative_scaled',points_per_frame=4);opt=torch.optim.AdamW(h.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
        cache=frame_cache({'a':line(13),'b':line(19)},4);mu,l,m=collate_latents(cache,['a','b']);h.get_ocr_loss(mu,l,m).backward();opt.step();h.eval();return cfg,h,opt,mu,l,m
    def test_common_zeroing_preserves_function_moments_counters_and_initial_parity(self):
        cfg,h,opt,mu,l,m=self.trained_parent();old={k:v.clone() for k,v in h.state_dict().items()};state=copy.deepcopy(opt.state_dict());expected=h(mu,padding_mask=~m).detach();ends=[]
        for spatial in (False,True):
            head=make_head(cfg,3,None,spatial=spatial);head.load_state_dict(old);o=torch.optim.AdamW(head.parameters());o.load_state_dict(state);before=state_digest(o.state_dict());proof=zero_unused_columns(head,o);head.eval()
            self.assertTrue(proof['old_moments_exactly_zero']);self.assertEqual(before,state_digest(o.state_dict()));torch.testing.assert_close(head(mu,padding_mask=~m),expected,atol=0,rtol=0);ends.append(proof['effective_head_sha256'])
        self.assertEqual(*ends)
    def test_zeroing_refuses_previously_used_columns(self):
        cfg,h,opt,mu,l,m=self.trained_parent();opt.state[h.input_proj.weight]['exp_avg'][0,20]=.1
        with self.assertRaises(ValueError):zero_unused_columns(h,opt)
    def test_spatial_train_gradient_and_dropout_rng_shape_parity(self):
        cfg,h,opt,mu,l,m=self.trained_parent();state=torch.get_rng_state().clone();ends=[]
        for spatial in (False,True):
            reader=make_head(cfg,3,None,spatial=spatial);reader.load_state_dict(h.state_dict());o=torch.optim.AdamW(reader.parameters());o.load_state_dict(opt.state_dict());zero_unused_columns(reader,o);torch.set_rng_state(state)
            loss=.5*reader.get_ocr_loss(mu,l,m)+.5*reader.get_ocr_loss(mu,l,m);loss.backward();self.assertTrue(torch.isfinite(loss));ends.append(torch.get_rng_state().clone())
            self.assertEqual(bool(torch.count_nonzero(reader.input_proj.weight.grad[:,20:24])),spatial)
        self.assertTrue(torch.equal(*ends));torch.set_rng_state(state)
    def test_launcher_self_contained_and_guarded(self):
        import ast
        s=(ROOT/'modal_ocr_spatial.py').read_text();tree=ast.parse(s);imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module];self.assertFalse(any(m.startswith('modal_') for m in imports));self.assertIn('if not train:',s)


class OCRSpatialReportTests(unittest.TestCase):
    def fixture(self):
        from iam_tools.report_ocr_spatial import FIXED,ARMS
        c={k:k for k in FIXED};c['max_updates']=2;c['spatial_features']=False;c['objective']='0.5clean+0.5clean;two identical forwards in BOTH arms';d=copy.deepcopy(c);d['spatial_features']=True
        r=dict(initial_state={'same':'zeroed columns/head/moments'},sample_schedule_sha256='same',final_dropout_rng_sha256='same',last_step=2,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True)
        l=[dict(step=i+1,sample_ids=[str(i)],lr=1e-4) for i in range(2)]
        return dict(zip(ARMS,[c,d])),{a:copy.deepcopy(r) for a in ARMS},{a:copy.deepcopy(l) for a in ARMS}
    def test_pairing_rejects_common_state_rng_exposure_or_budget_changes(self):
        from iam_tools.report_ocr_spatial import pairing
        c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
        for field,v in [('initial_state',{}),('final_dropout_rng_sha256','other'),('last_step',1),('stop_reason','wall_limit'),('entire_codec_bitwise_unchanged',False)]:
            bad=copy.deepcopy(r);bad['spatial_x'][field]=v
            with self.assertRaises(ValueError):pairing(c,bad,l)
        for field,v in [('sample_ids',['other']),('lr',5e-4),('step',2)]:
            bad=copy.deepcopy(l);bad['spatial_x'][0][field]=v
            with self.assertRaises(ValueError):pairing(c,r,bad)
    def test_only_spatial_flag_can_change_without_failing_controlled_config(self):
        from iam_tools.report_ocr_spatial import pairing
        c,r,l=self.fixture()
        for field in ('feature_stats','spatial_definition','schedule_skip','parent_sha256','objective'):
            bad=copy.deepcopy(c);bad['spatial_x'][field]='changed'
            with self.assertRaises(ValueError):pairing(bad,r,l)
        bad=copy.deepcopy(c);bad['spatial_x']['spatial_features']=False
        with self.assertRaises(ValueError):pairing(bad,r,l)

if __name__=='__main__':unittest.main()
