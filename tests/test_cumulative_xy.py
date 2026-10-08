import unittest
import torch
from iam_tools.cumulative_xy import cumulative_xy_loss, anchor_coefficient


class CumulativeXYTests(unittest.TestCase):
    def test_matches_rendered_absolute_error_with_jumps_and_origin(self):
        torch.manual_seed(1)
        pred=torch.randn(2,3,8,2,requires_grad=True);truth=torch.randn_like(pred)
        mask=torch.ones(2,3,8,dtype=torch.bool);mask[1,2,3:]=False
        stats=dict(std=[.2,.07],mean=[.05,-.01])
        absolute=lambda q:(q*torch.tensor(stats['std'])+torch.tensor(stats['mean'])).flatten(1,2).cumsum(1)
        expected=(absolute(pred)-absolute(truth))[mask.flatten(1)].square().mean()
        torch.testing.assert_close(cumulative_xy_loss(pred,truth,mask,stats),expected)
    def test_padding_has_no_gradient_or_effect_and_no_cross_line_accumulation(self):
        a=torch.ones(2,2,8,2,requires_grad=True);b=torch.zeros_like(a);mask=torch.ones(2,2,8,dtype=torch.bool);mask[0,1]=False
        loss=cumulative_xy_loss(a,b,mask,dict(std=[1.,1.]));loss.backward()
        self.assertTrue((a.grad[~mask]==0).all());self.assertGreater(float(a.grad[1,0].sum()),0.)
        changed=a.detach().clone();changed[~mask]=9999.
        torch.testing.assert_close(loss,cumulative_xy_loss(changed,b,mask,dict(std=[1.,1.])))
        expected=(sum(j*j for j in range(1,9))+sum(j*j for j in range(1,17)))/24
        self.assertAlmostEqual(float(loss.detach()),expected,places=5)
    def test_exact_target_zero_and_opposite_displacements_cancel_only_later(self):
        a=torch.zeros(1,1,8,2);mask=torch.ones(1,1,8,dtype=torch.bool);stats=dict(std=[1.,1.])
        self.assertEqual(float(cumulative_xy_loss(a,a,mask,stats)),0.)
        a[0,0,0,0]=1.;a[0,0,1,0]=-1.
        self.assertEqual(float(cumulative_xy_loss(a,torch.zeros_like(a),mask,stats)),1/16)
    def test_invalid_masks_scales_and_shapes_fail_closed(self):
        a=torch.zeros(1,1,8,2);mask=torch.ones(1,1,8,dtype=torch.bool)
        for std in ([0.,1.],[1.,float('nan')],[1.]):
            with self.assertRaises(ValueError):cumulative_xy_loss(a,a,mask,dict(std=std))
        mask[0,0,2]=False
        with self.assertRaises(ValueError):cumulative_xy_loss(a,a,mask,dict(std=[1.,1.]))
        with self.assertRaises(ValueError):cumulative_xy_loss(a,a,torch.zeros_like(mask),dict(std=[1.,1.]))
    def test_gradient_calibration_not_arbitrary_weight(self):
        coefficient=anchor_coefficient(2.,20.)
        self.assertAlmostEqual(coefficient*20./2.,.15)
        for args in [(0.,1.),(1.,float('nan')),(1.,1.,.8)]:
            with self.assertRaises(ValueError):anchor_coefficient(*args)


if __name__=='__main__':unittest.main()
