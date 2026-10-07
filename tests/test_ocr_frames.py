from pathlib import Path
import sys,unittest,tempfile
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_frame_study import split_tensor,frame_cache,paired_posterior_sampler
from iam_tools.ocr_context_features import unpack,transform,fit_stats,make_head
from iam_tools.ocr_pool_study import evaluate
from iam_tools.frozen_ocr_study import collate_latents


def line(n=13):
    t=(n+7)//8;a=torch.zeros(t*8,5);a[:n,:2]=torch.arange(n)[:,None]*torch.tensor([[.04,.03]])
    a[:,2]=1.;a[n-1:,2]=0.;a[n-1:,4]=1.;a[1,2]=0.;a[1,3]=1.
    mu=torch.zeros(1,48,t);mu[:,:40]=a.reshape(t,40).T
    return dict(mu=mu,lv=torch.full_like(mu,-8),mask=torch.ones(1,t,dtype=torch.bool),labels=torch.tensor([[0,1]]))

class OCRFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_exact_chronology_all_remainders_and_final_frame_mask(self):
        for n in range(4,25):
            c=line(n);old={'a':c};r=frame_cache(old,4)['a'];f,real=unpack(r['mu'],r['mask'],4)
            self.assertEqual(int(real.sum()),n);self.assertEqual(r['mu'].shape[-1],(n+3)//4)
            target=c['mu'][:,:40].reshape(1,8,5,-1).permute(0,3,1,2).reshape(1,-1,5)[:,:n]
            self.assertTrue(torch.equal(target,f.permute(0,3,1,2).reshape(1,-1,5)[:,:n]))
            self.assertTrue(torch.equal(c['labels'],r['labels']));self.assertIs(frame_cache(old,8),old)
    def test_gradient_reindexing_and_unused_noise_exclusion(self):
        x=torch.randn(2,48,3,requires_grad=True);r=split_tensor(x,4)
        self.assertEqual(r.shape,(2,48,6));r.sum().backward()
        self.assertTrue(torch.equal(x.grad[:,:40],torch.ones_like(x[:,:40])))
        self.assertEqual(float(x.grad[:,40:].abs().sum()),0.)
        x=x.detach();x[:,40:]=float('nan');self.assertTrue(torch.isfinite(split_tensor(x,4)).all())
    def test_default_eight_feature_path_identical_and_four_padding_safe(self):
        c=line();x,m=c['mu'],c['mask'];self.assertTrue(torch.equal(transform(x,m,'relative_scaled'),transform(x,m,'relative_scaled',points_per_frame=8)))
        r=frame_cache({'a':c},4)['a'];out=transform(r['mu'],r['mask'],'relative_scaled',points_per_frame=4)
        self.assertEqual(float(out[:,20:].abs().sum()),0.)
        f,real=unpack(r['mu'],r['mask'],4);self.assertEqual(float(out[:,:20].reshape(1,4,5,-1).permute(0,1,3,2)[~real].abs().sum()),0.)
        padded=torch.cat((r['mu'],torch.full((1,48,2),float('nan'))),2);mask=torch.cat((r['mask'],torch.zeros(1,2,dtype=torch.bool)),1)
        self.assertTrue(torch.isfinite(transform(padded,mask,points_per_frame=4)).all())
    def test_relative_translation_invariance_and_calibration_real_count(self):
        cache=frame_cache({'a':line()},4);c=cache['a'];shift=c['mu'].clone();shift[:,:20].reshape(1,4,5,-1)[:,:,0]+=4.
        torch.testing.assert_close(transform(shift,c['mask'],'relative_scaled',points_per_frame=4),transform(c['mu'],c['mask'],'relative_scaled',points_per_frame=4),atol=4e-7,rtol=1e-6)
        stats=fit_stats(cache,['a'],'relative_scaled',4);self.assertEqual(stats['real_points'],13)
    def test_original_noise_paired_across_both_readout_frames(self):
        cache={'a':line()};four=frame_cache(cache,4)
        rng=torch.get_rng_state().clone();torch.manual_seed(17);a=paired_posterior_sampler(cache,cache,8)('a',3);end8=torch.get_rng_state().clone()
        torch.manual_seed(17);b=paired_posterior_sampler(cache,four,4)('a',3);end4=torch.get_rng_state().clone();torch.set_rng_state(rng)
        self.assertTrue(torch.equal(end8,end4));self.assertTrue(torch.equal(split_tensor(a,4)[:,:,:4],b))
    def test_same_parameter_weights_and_four_finite_ctc_backward(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1);h8=make_head(cfg,3);h4=make_head(cfg,3,points_per_frame=4)
        self.assertTrue(all(torch.equal(v,h4.state_dict()[k]) for k,v in h8.state_dict().items()))
        cache=frame_cache({'a':line(),'b':line(19)},4);z,l,m=collate_latents(cache,['a','b']);loss=h4.get_ocr_loss(z,l,m);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in h4.parameters()))
        h4.eval();torch.testing.assert_close(h4(z,padding_mask=~m)[:4,:1],h4(cache['a']['mu'],padding_mask=~cache['a']['mask']),atol=2e-6,rtol=2e-6)
    def test_evaluation_sampler_callback_restores_rng_and_checks_shape(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1);h=make_head(cfg,3,points_per_frame=4);c={'a':line()};f=frame_cache(c,4)
        groups=dict(train=['a'],dev=['a'],held_out=['a']);state=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            r=evaluate(h,f,{'a':'ab'},groups,['a','b'],tmp,0,['a'],draws=2,posterior_sampler=paired_posterior_sampler(c,f,4))
            self.assertEqual(len(r['lines'][0]['sampled']),2);self.assertTrue(torch.equal(state,torch.get_rng_state()));self.assertTrue(h.training)
            with self.assertRaises(ValueError):evaluate(h,f,{'a':'ab'},groups,['a','b'],tmp,1,['a'],draws=2,posterior_sampler=lambda sid,n:torch.zeros(1,48,1))
    def test_reject_invalid_granularity_and_launcher_no_sibling_import(self):
        import ast
        for n in (1,2,16):
            with self.assertRaises(ValueError):split_tensor(torch.zeros(1,48,2),n)
            with self.assertRaises(ValueError):unpack(torch.zeros(1,48,2),torch.ones(1,2,dtype=torch.bool),n)
        imports=[n.module for n in ast.walk(ast.parse((ROOT/'modal_ocr_frames.py').read_text())) if isinstance(n,ast.ImportFrom) and n.module]
        self.assertFalse(any(n.startswith('modal_') for n in imports))

class OCRFrameReportTests(unittest.TestCase):
    def test_same_reference_slack_bins_despite_reader_frame_counts(self):
        from iam_tools.report_ocr_frames import slack_comparison
        records={'tight':dict(points=200,text='a'*23),'loose':dict(points=400,text='abcd')}
        # Tight would be CTC-infeasible at8; use no repeats for a valid fixture.
        records['tight']['text']='abcdefghijklmnopqrstuvw'
        def rows(t,l):return dict(lines=[dict(sample_id='tight',mu=dict(errors=t,characters=23)),dict(sample_id='loose',mu=dict(errors=l,characters=4))])
        r=slack_comparison(records,dict(points8=rows(5,2),points4=rows(2,1)),['tight','loose'])
        self.assertEqual(r['tight_le0.25']['lines'],1);self.assertEqual(r['loose_gt0.5']['lines'],1)
        self.assertEqual(r['tight_le0.25']['groups']['points4']['cer'],2/23)
        self.assertIsNone(r['middle_0.25to0.5']['groups']['points8']['cer'])
    def test_error_changes_align_by_id_not_row_order(self):
        from iam_tools.report_ocr_frames import compare_rows
        a=dict(lines=[dict(sample_id='x',mu=dict(errors=2)),dict(sample_id='y',mu=dict(errors=1))])
        b=dict(lines=[dict(sample_id='y',mu=dict(errors=2)),dict(sample_id='x',mu=dict(errors=1))])
        r=compare_rows(a,b,['x','y']);self.assertEqual((r['improved'],r['tied'],r['worsened']),(1,0,1))

class OCRSeedReplicationTests(unittest.TestCase):
    def test_seed_range_validated_before_any_allocation(self):
        from iam_tools.ocr_frame_study import validate_seed
        for seed in (0,42,137,2**31-1):self.assertEqual(validate_seed(seed),seed)
        for seed in (-1,2**31,False,True,137.,'137',None):
            with self.assertRaises(ValueError):validate_seed(seed)
    def test_new_seed_changes_weights_but_keeps_within_seed_pair_and_rng(self):
        from iam_tools.ocr_context_study import tensor_digest
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        state=torch.get_rng_state().clone();pairs=[]
        for seed in (42,137):
            a=make_head(cfg,3,seed=seed,points_per_frame=8);b=make_head(cfg,3,seed=seed,points_per_frame=4)
            self.assertEqual(tensor_digest(a.state_dict()),tensor_digest(b.state_dict()));pairs.append(tensor_digest(a.state_dict()))
        self.assertNotEqual(*pairs);self.assertTrue(torch.equal(state,torch.get_rng_state()))
    def test_default_seed42_preserves_historical_factory_and_runner_signature(self):
        import inspect
        from iam_tools.ocr_frame_study import run
        self.assertEqual(inspect.signature(run).parameters['seed'].default,42)
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        a=make_head(cfg,3);b=make_head(cfg,3,seed=42)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

if __name__=='__main__':unittest.main()
