import unittest
import torch
from iam_tools.continuous_prefix import continuous_prefix_forward,paired_feedback_losses
from iam_tools.generated_prefix import own_prefix_forward,paired_losses
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter


class ContinuousPrefixTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(4);model=PointFeedbackStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,point_feedback=False)
        return model,torch.tensor([[0,1,2],[1,3,-1]]),torch.tensor([0,1])
    def test_forward_identical_only_gradient_contract_changes(self):
        m,t,w=self.fixture();a=own_prefix_forward(m,t,w,5);b=continuous_prefix_forward(m,t,w,5)
        for j in [0,1]:torch.testing.assert_close(a[j],b[j],atol=0,rtol=0)
        for key in a[2]:torch.testing.assert_close(a[2][key],b[2][key],atol=0,rtol=0)
        self.assertFalse(a[2]['used_history'].requires_grad);self.assertTrue(b[2]['used_history'].requires_grad)
    def test_future_offsets_differentiate_earlier_xy_but_not_hard_pen_logits(self):
        m,t,w=self.fixture()
        for forward in [own_prefix_forward,continuous_prefix_forward]:
            captures=[];hook=m.point_readout.register_forward_hook(lambda module,args,out:captures.append(out))
            try:o,p,tr=forward(m,t,w,4)
            finally:hook.remove()
            g=torch.autograd.grad(o[:,1:].square().mean(),captures[0],allow_unused=True)[0]
            if forward==own_prefix_forward:self.assertTrue(g is None or float(g.abs().sum())==0.)
            else:
                self.assertGreater(float(g[...,:2].abs().sum()),0.)
                self.assertEqual(float(g[...,2:].abs().sum()),0.)
    def test_reject_wrong_architecture_and_budget(self):
        m,t,w=self.fixture()
        for n in [0,257,1.2]:
            with self.assertRaises(ValueError):continuous_prefix_forward(m,t,w,n)
        m.config['point_feedback']=True
        with self.assertRaises(ValueError):continuous_prefix_forward(m,t,w,4)
    def test_paired_loss_values_identical_both_gradient_policies_and_original(self):
        m,t,w=self.fixture();f=torch.randn(2,4,40);target=torch.randn(2,4,8,2);states=torch.zeros(2,4,8,dtype=torch.long);mask=torch.ones(2,4,8,dtype=torch.bool)
        cfg=dict(pen_weights=[1.,3.8,8.],offset_stats=dict(std=[.2,.07]),pen_weight=.02,teacher_anchor_weight=.01)
        original=paired_losses(m,f,t,w,target,states,mask,cfg)
        for policy in [False,True]:
            terms=paired_feedback_losses(m,f,t,w,target,states,mask,cfg,policy)
            for key in ['base','teacher_offset','teacher_pen','teacher_xy','own_xy','own_pen','own_offset_diagnostic']:
                torch.testing.assert_close(original[key],terms[key],atol=0,rtol=0)
        with self.assertRaises(ValueError):paired_feedback_losses(m,f,t,w,target,states,mask,cfg,None)


if __name__=='__main__':unittest.main()
