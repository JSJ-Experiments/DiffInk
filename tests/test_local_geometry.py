import unittest
import torch
from iam_tools.local_geometry import target_relative_difference_loss as loss

class LocalGeometryTests(unittest.TestCase):
    def test_matches_authentic_corners_and_translation_not_smoothing(self):
        target=torch.tensor([[[0.,0.],[1,0],[1,1],[2,1]]]);states=torch.tensor([[0,0,0,2]]);mask=torch.ones(1,4,dtype=torch.bool)
        self.assertEqual(float(loss(target,target,states,mask)),0.)
        self.assertEqual(float(loss(target+3,target,states,mask)),0.)
        pred=target.clone();pred[0,1,1]=.2
        self.assertGreater(float(loss(pred,target,states,mask)),0.)
    def test_nonuniform_lengths_are_floored_and_scale_invariant(self):
        t=torch.tensor([[[0.,0.],[.01,0],[1,0],[2,0]]]);p=t.clone();p[0,1,1]=.01
        s=torch.tensor([[0,0,0,2]]);m=torch.ones(1,4,dtype=torch.bool)
        length=torch.tensor([.01,.99,1.]);floor=torch.quantile(length,.25)
        d=torch.diff(p-t,dim=1)[0]/length.clamp_min(floor)[:,None]
        torch.testing.assert_close(loss(p,t,s,m),d.square().mean())
        torch.testing.assert_close(loss(p*7,t*7,s,m),loss(p,t,s,m))
    def test_ignores_stroke_jumps_nan_padding_with_finite_zero_pad_gradient(self):
        t=torch.tensor([[[0.,0],[1,0],[5,5],[5,6],[float('nan'),float('nan')]]]);p=t.clone();p[:,:2]+=2;p[:,2:4]-=3;p.requires_grad_()
        s=torch.tensor([[0,1,0,2,2]]);m=torch.tensor([[True,True,True,True,False]])
        value=loss(p,t,s,m);self.assertEqual(float(value.detach()),0.);value.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertTrue((p.grad[:,4:]==0).all())
    def test_zero_target_edges_and_no_stroke_windows_are_safe(self):
        p=torch.zeros(1,3,2,requires_grad=True);t=torch.zeros_like(p);m=torch.ones(1,3,dtype=torch.bool)
        for s in [torch.tensor([[0,0,2]]),torch.tensor([[1,1,2]])]:
            value=loss(p,t,s,m);self.assertEqual(float(value.detach()),0.);value.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
    def test_invalid_shapes_and_bounds(self):
        p=torch.zeros(1,3,2);s=torch.zeros(1,3,dtype=torch.long);m=torch.ones(1,3,dtype=torch.bool)
        for quantile in [0,1]:
            with self.assertRaises(ValueError):loss(p,p,s,m,quantile)
        with self.assertRaises(ValueError):loss(p,p[:,:2],s,m)
        with self.assertRaises(ValueError):loss(p,p,s,m.float())

    def test_readout_stage_bounds_fail_before_data_loading(self):
        from iam_tools.xy_readout_probe import run
        with self.assertRaises(ValueError):run('missing','missing','missing','missing','missing','missing',feature_stage='unknown')

    def test_gradient_calibration_is_train_only_fixed_and_has_no_side_effects(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        from iam_tools.local_geometry import calibrate_relative
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Conv1d(5,4,8,stride=8)
                self.conv_mu=torch.nn.Conv1d(4,4,1);self.decoder=torch.nn.Conv1d(4,4,1)
                self.transformer_decoder=torch.nn.Module();self.transformer_decoder.fc=torch.nn.Linear(4,123)
            def to_model_space(self,x):return x
            def decode(self,z,padding_mask=None):
                x=self.decoder(z).repeat_interleave(8,dim=-1).transpose(1,2)
                return self.transformer_decoder.fc(x).transpose(1,2)
        torch.manual_seed(3);model=Tiny().eval();raw=torch.zeros(1,5,16)
        raw[:,0]=torch.linspace(0,1,16);raw[:,1]=torch.sin(torch.linspace(0,3,16));raw[:,2]=1;raw[:,2,-1]=0;raw[:,4,-1]=1
        batch=(raw,torch.ones(1,16,dtype=torch.bool),torch.zeros(1,1))
        before={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
        c=calibrate_relative(model,[batch],.2)
        self.assertAlmostEqual(c['relative_weight']*c['decoder_gradient_norms']['relative']/c['decoder_gradient_norms']['geometry'],.2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(p.grad is None for p in model.parameters()))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        with self.assertRaises(ValueError):calibrate_relative(model,[],.2)
