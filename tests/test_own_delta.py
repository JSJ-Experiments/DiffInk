import unittest
import torch
from iam_tools.own_delta import displacement_mask,displacement_loss,paired_delta_losses,delta_coefficients
from iam_tools.continuous_prefix import paired_feedback_losses
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter
class OwnDeltaTests(unittest.TestCase):
    def test_target_pen_mask_crosses_block_boundaries_and_ignores_origin_jumps_padding(self):
        p=torch.zeros(1,2,8,dtype=torch.long);p.flatten()[3]=1;p.flatten()[8]=1;p.flatten()[10]=2;m=torch.arange(16).reshape(1,2,8)<11
        selected=displacement_mask(p,m,'ink').flatten();self.assertEqual(selected.nonzero().flatten().tolist(),[1,2,3,5,6,7,8,10]);self.assertTrue(selected[8]);self.assertFalse(selected[9])
        self.assertTrue(torch.equal(displacement_mask(p,m,'all'),m))
        pred=torch.zeros(1,2,8,2,requires_grad=True);q=torch.ones_like(pred);loss=displacement_loss(pred,q,p,m,'ink');loss.backward();self.assertEqual(float(pred.grad.flatten(0,2)[~selected].abs().sum()),0.)
        changed=pred.detach().clone();changed.flatten(0,2)[~selected]=1000
        self.assertEqual(float(displacement_loss(changed,q,p,m,'ink')),float(loss.detach()))
    def test_target_matching_rewards_authentic_corners_not_smoothness(self):
        p=torch.zeros(1,1,8,dtype=torch.long);p.flatten()[-1]=2;m=torch.ones_like(p,dtype=torch.bool);corner=torch.tensor([[[[0.,0.],[1.,0.],[0.,1.],[-1.,0.],[0.,-1.],[1.,0.],[0.,1.],[-1.,0.]]]])
        self.assertEqual(float(displacement_loss(corner,corner,p,m,'ink')),0.)
        self.assertGreater(float(displacement_loss(torch.zeros_like(corner),corner,p,m,'ink')),0.)
    def test_all_matches_existing_real_offset_mse(self):
        torch.manual_seed(7);x=torch.randn(2,3,8,2);q=torch.randn_like(x);p=torch.zeros(2,3,8,dtype=torch.long);m=torch.arange(24)[None].repeat(2,1).reshape(2,3,8)<torch.tensor([12,20])[:,None,None]
        torch.testing.assert_close(displacement_loss(x,q,p,m,'all'),(x-q)[m].square().mean(),rtol=0,atol=0)
    def test_same_baseline_outputs_scalar_loss_state_rng_as_parent_helper(self):
        torch.manual_seed(11);model=PointFeedbackStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,point_feedback=False);f=torch.randn(2,3,40);t=torch.tensor([[0,1,2],[1,3,-1]]);w=torch.tensor([0,1]);q=torch.randn(2,3,8,2);p=torch.zeros(2,3,8,dtype=torch.long);m=torch.ones_like(p,dtype=torch.bool);cfg=dict(pen_weights=[1.,3.8,8.],offset_stats=dict(std=[.2,.07]),pen_weight=.02,teacher_anchor_weight=.01,own_xy_weight=.003,own_pen_weight=.002)
        state={k:v.clone() for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone();old=paired_feedback_losses(model,f,t,w,q,p,m,cfg,True);new=paired_delta_losses(model,f,t,w,q,p,m,cfg)
        for key in ['base','teacher_offset','teacher_pen','teacher_xy','own_xy','own_pen','own_offset_diagnostic']:torch.testing.assert_close(old[key],new[key],rtol=0,atol=0)
        torch.testing.assert_close(new['complete'],old['base']+.003*old['own_xy']+.002*old['own_pen'],rtol=0,atol=0)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertTrue(all(torch.equal(v,state[k]) for k,v in model.state_dict().items()))
        grads=torch.autograd.grad(new['own_ink_delta'],[model.point_readout.weight,model.clock.weight],retain_graph=True);self.assertGreater(float(grads[0][:2].abs().sum()),0.);self.assertEqual(float(grads[0][2:].abs().sum()),0.);self.assertGreater(float(grads[1].abs().sum()),0.)
    def test_bad_scope_masks_states_or_degenerate_calibration_rejected(self):
        p=torch.zeros(1,1,8,dtype=torch.long);m=torch.ones_like(p,dtype=torch.bool);x=torch.zeros(1,1,8,2)
        for mode in ['smooth',None]:
            with self.assertRaises(ValueError):displacement_loss(x,x,p,m,mode)
        bad=m.clone();bad[...,3]=False
        with self.assertRaises(ValueError):displacement_mask(p,bad,'ink')
        bad=p.clone();bad[...,3]=3
        with self.assertRaises(ValueError):displacement_mask(bad,m,'ink')
        p.fill_(1)
        with self.assertRaises(ValueError):displacement_loss(x,x,p,m,'ink')
        self.assertEqual(delta_coefficients(2,10,5),dict(ink=.05,all=.1))
        for a in [0,float('nan')]:
            with self.assertRaises(ValueError):delta_coefficients(a,10,5)
if __name__=='__main__':unittest.main()
