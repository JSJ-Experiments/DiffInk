import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
from iam_tools.fast_geometry import masked_geometry, masked_pen, fast_terms, GeometryGraphs
from model.losses import target_difference_loss
from iam_tools.latent_integration import DELTA_WEIGHT, terms
from iam_tools.pen_refit import refit_loss


class FastGeometryTests(unittest.TestCase):
    def test_geometry_matches_values_gradients_and_excludes_nan_padding(self):
        for seed in range(3):
            torch.manual_seed(seed)
            p=torch.randn(2,16,2,dtype=torch.float64,requires_grad=True)
            t=torch.randn_like(p);m=torch.arange(16)[None,:]<torch.tensor([13,7])[:,None]
            s=torch.randint(0,3,(2,16));p.data[~m]=float('nan');t[~m]=float('nan')
            slow=(p[m]-t[m]).square().mean()+DELTA_WEIGHT*target_difference_loss(p,t,s,m)
            fast=masked_geometry(p,t,s,m)
            torch.testing.assert_close(fast,slow,atol=1e-14,rtol=1e-14)
            a,=torch.autograd.grad(slow,p,retain_graph=True);b,=torch.autograd.grad(fast,p)
            torch.testing.assert_close(a,b,atol=1e-14,rtol=1e-14)
            self.assertTrue(torch.isfinite(b).all());self.assertTrue((b[~m]==0).all())

    def test_pen_matches_focal_value_and_gradients(self):
        for state in ([0,0,1,0,2],[0,0,0,0,2],[0,0,0,0,0]):
            x=torch.randn(1,3,8,dtype=torch.float64,requires_grad=True)
            s=torch.tensor([state+[2]*3]);m=torch.tensor([[True]*5+[False]*3])
            slow=refit_loss(x.transpose(1,2)[m],s[m],'bounded_three_state')
            fast=masked_pen(x,s,m)
            # Reference class counts intentionally FP32; fast uses logits dtype.
            torch.testing.assert_close(fast,slow,atol=1e-7,rtol=1e-7)
            a,=torch.autograd.grad(slow,x,retain_graph=True);b,=torch.autograd.grad(fast,x)
            torch.testing.assert_close(a,b,atol=1e-7,rtol=1e-7)
            self.assertTrue((b[:,:,-3:]==0).all())
            x.data[:,:,-3:]=float('nan')
            value=masked_pen(x,s,m)
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(torch.autograd.grad(value,x)[0]).all())

    def test_full_loss_with_identical_noise_matches_reference(self):
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Conv1d(5,4,8,stride=8)
                self.conv_mu=torch.nn.Conv1d(4,4,1);self.conv_logvar=torch.nn.Conv1d(4,4,1)
                self.fc=torch.nn.Linear(4,123)
            def to_model_space(self,x):return x
            def decode(self,z,padding_mask=None):return self.fc(z.repeat_interleave(8,dim=-1).transpose(1,2)).transpose(1,2)
            def kl_divergence_new(self,*args):return args[0].sum()*0
        model=Tiny();raw=torch.randn(1,5,16);raw[:,2:]=0;raw[:,2]=1;raw[:,2,-3:]=0;raw[:,4,-3:]=1
        mask=torch.arange(16)[None,:]<14;noise=torch.randn(1,4,2)
        slow=terms(model,(raw,mask,torch.zeros(1,1)),epsilon=noise,pen_on_mean=True)
        a=torch.stack([slow[k] for k in ('mean_geometry','sampled_geometry','pen')]);b=fast_terms(model,raw,mask,noise)
        torch.testing.assert_close(a,b)
        ga=torch.autograd.grad(a.sum(),tuple(model.parameters()),retain_graph=True)
        gb=torch.autograd.grad(b.sum(),tuple(model.parameters()))
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
        from iam_tools.fullset_joint import joint_loss
        reference=joint_loss(model,(raw,mask,torch.zeros(1,1)))
        mean=fast_terms(model,raw,mask)
        torch.testing.assert_close(mean[0],reference['point']+DELTA_WEIGHT*reference['delta'])
        torch.testing.assert_close(mean[2],reference['pen'])
        self.assertEqual(float(mean[1].detach()),0.)
        with self.assertRaises(ValueError):GeometryGraphs(model,.01)

    def test_parallel_metrics_equal_serial_including_real_corners(self):
        from iam_tools.metric_workers import metric_pool,geometry_job
        xy=np.array([[0.,0.],[1.,0.],[1.,1.],[2.,1.],[3.,0.]])
        target=xy.copy();xy[2]+=.02;states=np.array([0,0,1,0,2])
        expected=geometry_job(xy,target,states)
        with metric_pool(2) as pool:
            self.assertEqual(pool.submit(geometry_job,xy,target,states).result(),expected)
        with self.assertRaises(ValueError):metric_pool(0)

    def test_dense_capture_context_restores_nested_setting_even_on_error(self):
        from types import SimpleNamespace
        from iam_tools.fast_geometry import dense_training_decoder
        transformer=SimpleNamespace(use_nested_tensor=True)
        model=SimpleNamespace(transformer_decoder=SimpleNamespace(transformer=transformer))
        with self.assertRaises(RuntimeError):
            with dense_training_decoder(model):
                self.assertFalse(transformer.use_nested_tensor)
                raise RuntimeError('test')
        self.assertTrue(transformer.use_nested_tensor)

    def test_same_line_draw_batches_preserve_noise_order_padding_and_groupnorm(self):
        from iam_tools.writer_expansion import posterior_outputs
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.norm=torch.nn.GroupNorm(1,4);self.fc=torch.nn.Conv1d(4,123,1)
            def decode(self,z,padding_mask=None):
                self.asserted_masks.append(padding_mask.clone())
                return self.fc(self.norm(z.repeat_interleave(8,dim=-1)))
            def ocr_model(self,z):return z.transpose(1,2).transpose(0,1)
        m=Tiny().eval();mu=torch.randn(1,4,2);lv=mu*0-5;mask=torch.arange(16)[None,:]<14
        outputs=[];rng=[]
        for size in [1,4,21]:
            m.asserted_masks=[];torch.manual_seed(881)
            values=list(posterior_outputs(m,mu,lv,mask,14,2,20,size));outputs.append(values);rng.append(torch.get_rng_state())
            self.assertTrue(all(torch.equal(row,~mask[0]) for batch in m.asserted_masks for row in batch))
        for values in outputs[1:]:
            for a,b in zip(outputs[0],values):
                self.assertEqual(a[0],b[0]);np.testing.assert_allclose(a[1],b[1],atol=2e-6,rtol=2e-6)
                np.testing.assert_array_equal(a[2],b[2]);self.assertEqual(a[3],b[3])
        for state in rng[1:]:self.assertTrue(torch.equal(state,rng[0]))
        with self.assertRaises(ValueError):list(posterior_outputs(m.train(),mu,lv,mask,14,2))


if __name__=='__main__':unittest.main()
