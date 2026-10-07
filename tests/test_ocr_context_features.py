from pathlib import Path
import sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_context_features import unpack,relative_x,transform,fit_stats,local_attention_mask,make_head


def payload(n=13,t=3):
    a=torch.zeros(t*8,5);a[:n,:2]=torch.arange(n)[:,None]*torch.tensor([[.04,.03]])
    a[:,2]=1.;a[n-1:,2]=0.;a[n-1:,4]=1.
    x=torch.zeros(1,48,t);x[:,:40]=a.reshape(t,40).T
    return x,torch.arange(t)[None,:]<(n+7)//8


class OCRContextFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_polyphase_real_mask_includes_first_eoc_not_padding(self):
        x,m=payload();f,r=unpack(x,m);self.assertEqual(int(r.sum()),13)
        self.assertEqual(f.shape,(1,8,5,3))
    def test_relative_x_preserves_geometry_up_to_translation(self):
        x,m=payload(24,3);f,r=unpack(x,m);d=relative_x(f)
        recovered=d[:,0].cumsum(-1)[:,None]+f[:,0,0,:1,None]+d
        # Phase0 already holds cumulative anchor; do not add its delta twice.
        recovered[:,0]=d[:,0].cumsum(-1)+f[:,0,0,:1]
        torch.testing.assert_close(recovered,f[:,:,0],atol=1e-7,rtol=1e-6)
        shifted=x.clone();shifted[:,:40].reshape(1,8,5,3)[:,:,0]+=5.
        torch.testing.assert_close(transform(x,m,'relative_scaled'),transform(shifted,m,'relative_scaled'),atol=5e-7,rtol=1e-6)
    def test_synthetic_tail_does_not_encode_negative_line_width(self):
        x,m=payload();out=transform(x,m,'relative_scaled');f,r=unpack(x,m)
        p=out[:,:40].reshape(1,8,5,3)
        self.assertEqual(float(p.permute(0,1,3,2)[~r].abs().sum()),0.)
    def test_unused_and_padded_nan_features_excluded(self):
        x,m=payload();x[:,40:]=float('nan');x[:,:,2]=float('nan')
        out=transform(x,m);self.assertTrue(torch.isfinite(out).all());self.assertEqual(float(out[:,40:].abs().sum()),0.)
    def test_fit_stats_uses_only_supplied_train_ids_and_real_points(self):
        x,m=payload();cache={'train':dict(mu=x,mask=m),'held':dict(mu=x*float('nan'),mask=m)}
        stats=fit_stats(cache,['train'],'global_scaled');self.assertEqual(stats['real_points'],13)
        self.assertAlmostEqual(stats['mean'][0],.24,places=6)
    def test_local_mask_bounds_real_queries_but_keeps_dummy_rows_finite(self):
        valid=torch.tensor([[True,True,False,False,False],[True]*5]);a=local_attention_mask(valid,2,1).reshape(2,2,5,5)
        self.assertFalse(a[0,0,4,0]);self.assertTrue(a[1,0,4,0]);self.assertFalse(a[1,0,4,3])
        self.assertTrue((~a).any(-1).all());self.assertTrue(a[0,:,:2,2:].all())
    def test_local_head_padding_parity_finite_backward_and_outside_context_zero(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=2)
        h=make_head(cfg,4,radius=1).eval();a,ma=payload(8,4);b,mb=payload(32,4)
        x=torch.cat((a,b)).requires_grad_();m=torch.cat((ma,mb))
        out=h(x,padding_mask=~m);self.assertTrue(torch.isfinite(out).all());out[0].square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[1,:,3:].abs().sum()),0.)
        torch.testing.assert_close(out[:1,:1],h(a[:,:,:1],padding_mask=~ma[:,:1]),atol=2e-6,rtol=2e-6)
    def test_fresh_head_weights_identical_across_modes_rng_preserved(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        state=torch.get_rng_state();a=make_head(cfg,4);b=make_head(cfg,4,mode='relative_scaled',radius=2)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_default_attention_contract_preserved(self):
        from model.ocr import ChineseHandwritingOCR
        h=ChineseHandwritingOCR(16,16,2,1,4,dropout=0).eval()
        x=torch.randn(2,16,5);pad=torch.tensor([[False]*5,[False]*3+[True]*2])
        with torch.no_grad():
            a=h(x,padding_mask=pad);b=h(x,padding_mask=pad,attention_mask=None)
            c=h(x,padding_mask=pad,attention_mask=torch.zeros(5,5,dtype=torch.bool))
        self.assertTrue(torch.equal(a,b));torch.testing.assert_close(a[:3],c[:3],atol=2e-6,rtol=2e-6)

    def test_coverage_missing_labels_and_identical_transcripts_reported(self):
        from iam_tools.report_ocr_context import text_coverage
        c=text_coverage({'a':'aba','b':'ax','c':'aba'},{'train':['a'],'held_out':['b','c']})
        self.assertEqual(c['train_characters'],dict(a=2,b=1))
        self.assertEqual(c['held_out_unseen_characters'],dict(x=1))
        self.assertEqual(c['held_out_total_characters'],5)
        self.assertEqual(c['held_out_exact_transcripts_in_train'],1)

if __name__=='__main__':unittest.main()
