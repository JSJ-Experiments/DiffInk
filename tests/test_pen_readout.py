import unittest
import torch
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter
from iam_tools.pen_readout import extract_pen_features,make_pen_readout,install_pen_readout,assert_only_pen_rows_changed,pen_focal

class PenReadoutTests(unittest.TestCase):
    def model(self):
        torch.manual_seed(10);return PointFeedbackStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,point_feedback=False).eval()
    def test_frozen_features_reproduce_pen_logits_and_real_mask(self):
        m=self.model();f=torch.randn(2,3,40);text=torch.tensor([[0,1],[2,3]]);wi=torch.tensor([0,1]);mask=torch.ones(2,3,8,dtype=torch.bool);mask[1,2,3:]=False
        features,outputs=extract_pen_features(m,f,text,wi,mask);head=make_pen_readout(m)
        torch.testing.assert_close(head(features),outputs[1][mask],atol=1e-7,rtol=1e-5);self.assertFalse(features.requires_grad);self.assertEqual(len(features),int(mask.sum()));self.assertEqual(len(m.point_readout._forward_pre_hooks),0)
    def test_install_changes_only_three_pen_rows_and_preserves_teacher_xy(self):
        m=self.model();before={k:v.clone() for k,v in m.state_dict().items()};f=torch.randn(2,3,40);text=torch.tensor([[0,1],[2,3]]);wi=torch.tensor([0,1]);xy=m.teacher(f,text,wi)[0].detach().clone();head=make_pen_readout(m)
        with torch.no_grad():head.weight.add_(.1);head.bias.add_(.2)
        install_pen_readout(m,head);self.assertTrue(assert_only_pen_rows_changed(before,m.state_dict()));torch.testing.assert_close(xy,m.teacher(f,text,wi)[0],atol=0,rtol=0)
        bad={k:v.clone() for k,v in m.state_dict().items()};bad['point_readout.weight'][0,0]+=.01
        with self.assertRaises(ValueError):assert_only_pen_rows_changed(before,bad)
    def test_same_bounded_focal_formula_and_gradients(self):
        head=torch.nn.Linear(16,3);features=torch.randn(20,16);states=torch.tensor([0]*15+[1]*4+[2]);weights=torch.tensor([1.,3.8,8.]);loss=pen_focal(head(features),states,weights);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(head.weight.grad.abs().sum()),0.)
    def test_bad_shapes_and_nonfinite_head_fail(self):
        m=self.model();head=make_pen_readout(m)
        with torch.no_grad():head.bias[0]=float('nan')
        with self.assertRaises(ValueError):install_pen_readout(m,head)
        with self.assertRaises(ValueError):pen_focal(torch.zeros(2,2),torch.zeros(2,dtype=torch.long),torch.ones(3))

if __name__=='__main__':unittest.main()
