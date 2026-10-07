from pathlib import Path
import sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from iam_tools.ocr_frame_study import split_tensor,frame_cache,paired_posterior_sampler,validate_frame_pair
from iam_tools.ocr_context_features import unpack,transform,fit_stats,make_head
from iam_tools.frozen_ocr_study import collate_latents
from iam_tools.report_ocr_frames import reader_pair,compare_rows
from test_ocr_frames import line

class OCRTwoFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_exact_chronology_eoc_all_remainders_and_no_interpolation(self):
        # The shared synthetic line() fixture has an internal stroke boundary at
        # index1, so n=2 would collide with its final-EOC marker. Start at3 to
        # exercise both odd/even two-point remainders with a valid trajectory.
        for n in range(3,34):
            c=line(n);f=frame_cache({'a':c},2)['a'];v,real=unpack(f['mu'],f['mask'],2)
            self.assertEqual(f['mu'].shape,(1,48,(n+1)//2));self.assertEqual(int(real.sum()),n)
            original=c['mu'][:,:40].reshape(1,8,5,-1).permute(0,3,1,2).reshape(1,-1,5)[:,:n]
            actual=v.permute(0,3,1,2).reshape(1,-1,5)[:,:n]
            self.assertTrue(torch.equal(actual,original));self.assertEqual(int(actual[0,-1,2:].argmax()),2)
            self.assertTrue(torch.equal(f['labels'],c['labels']))
    def test_gradient_and_nan_unused_fields_exclusion(self):
        x=torch.randn(2,48,3,requires_grad=True);v=split_tensor(x,2)
        self.assertEqual(v.shape,(2,48,12));v.sum().backward()
        self.assertTrue(torch.equal(x.grad[:,:40],torch.ones_like(x[:,:40])))
        self.assertEqual(float(x.grad[:,40:].abs().sum()),0.)
        x=x.detach();x[:,40:]=float('nan');self.assertTrue(torch.isfinite(split_tensor(x,2)).all())
    def test_relative_features_calibration_and_padding_nan_safety(self):
        cache=frame_cache({'a':line()},2);c=cache['a'];stats=fit_stats(cache,['a'],'relative_scaled',2)
        self.assertEqual(stats['real_points'],13);shift=c['mu'].clone();shift[:,:10].reshape(1,2,5,-1)[:,:,0]+=4.
        a=transform(c['mu'],c['mask'],'relative_scaled',stats,2)
        torch.testing.assert_close(a,transform(shift,c['mask'],'relative_scaled',stats,2),atol=3e-5,rtol=3e-5)
        self.assertEqual(float(a[:,10:].abs().sum()),0.)
        fields,real=unpack(c['mu'],c['mask'],2)
        self.assertEqual(float(a[:,:10].reshape(1,2,5,-1).permute(0,1,3,2)[~real].abs().sum()),0.)
        padded=torch.cat((c['mu'],torch.full((1,48,3),float('nan'))),2);mask=torch.cat((c['mask'],torch.zeros(1,3,dtype=torch.bool)),1)
        self.assertTrue(torch.isfinite(transform(padded,mask,'relative_scaled',stats,2)).all())
    def test_original_noise_stream_paired_on_every_preserved_field(self):
        original={'a':line(13)};two=frame_cache(original,2);four=frame_cache(original,4)
        state=torch.get_rng_state().clone()
        torch.manual_seed(97);a=paired_posterior_sampler(original,four,4)('a',3);end4=torch.get_rng_state().clone()
        torch.manual_seed(97);b=paired_posterior_sampler(original,two,2)('a',3);end2=torch.get_rng_state().clone()
        torch.manual_seed(97);c=paired_posterior_sampler(original,original,8)('a',3);torch.set_rng_state(state)
        self.assertTrue(torch.equal(end4,end2));self.assertTrue(torch.equal(b,split_tensor(c,2)[:,:,:7]))
        expected=a[:,:20].reshape(3,4,5,-1).permute(0,3,1,2).reshape(3,-1,5)[:,:13]
        actual=b[:,:10].reshape(3,2,5,-1).permute(0,3,1,2).reshape(3,-1,5)[:,:13]
        self.assertTrue(torch.equal(actual,expected))
    def test_equal_weights_finite_ctc_backward_and_masked_batch_parity(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        h2=make_head(cfg,3,points_per_frame=2);h4=make_head(cfg,3,points_per_frame=4)
        self.assertTrue(all(torch.equal(v,h4.state_dict()[k]) for k,v in h2.state_dict().items()))
        cache=frame_cache({'a':line(13),'b':line(23)},2);z,l,m=collate_latents(cache,['a','b'])
        loss=h2.get_ocr_loss(z,l,m);loss.backward();self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in h2.parameters()))
        h2.eval();torch.testing.assert_close(h2(z,padding_mask=~m)[:7,:1],h2(cache['a']['mu'],padding_mask=~cache['a']['mask']),atol=3e-6,rtol=3e-6)
    def test_only_predeclared_pair_no_silent_default_change(self):
        import inspect
        from iam_tools.ocr_frame_study import run
        self.assertEqual(inspect.signature(run).parameters['frame_pair'].default,(8,4))
        for pair in ((8,4),(4,2)):self.assertEqual(validate_frame_pair(pair),pair)
        for pair in ((8,2),(2,4),(4,4),[4,2],(4,True),(4.,2),()):
            with self.assertRaises(ValueError):validate_frame_pair(pair)
    def test_generic_reports_reject_missing_or_mixed_arms_and_correct_count_labels(self):
        self.assertEqual(reader_pair({'points4':{},'points2':{}}),(4,2))
        self.assertEqual(reader_pair({'points8':{},'points4':{}}),(8,4))
        for r in ({'points2':{}},{'points8':{},'points2':{}},{'points8':{},'points4':{},'points2':{}}):
            with self.assertRaises(ValueError):reader_pair(r)
        a=dict(lines=[dict(sample_id='x',mu=dict(errors=3)),dict(sample_id='y',mu=dict(errors=1))])
        b=dict(lines=[dict(sample_id='y',mu=dict(errors=2)),dict(sample_id='x',mu=dict(errors=1))])
        r=compare_rows(a,b,['x','y'],(4,2))
        self.assertEqual((r['improved'],r['tied'],r['worsened']),(1,0,1))
        self.assertEqual(r['lines'][0],dict(sample_id='x',errors4=3,errors2=1))

if __name__=='__main__':unittest.main()
