import unittest
import torch
from iam_tools.generated_prefix import own_prefix_forward, paired_losses, auxiliary_coefficients, free_stop_rows
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter


class GeneratedPrefixTests(unittest.TestCase):
    def setup_inputs(self):
        torch.manual_seed(100)
        model=PointFeedbackStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,point_feedback=False)
        text=torch.tensor([[0,1,2],[1,3,-1]]);writers=torch.tensor([0,1])
        return model,text,writers
    def test_no_target_fields_and_actual_free_prefix_parity(self):
        m,t,w=self.setup_inputs();m.eval()
        with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([5.,0.,-30.])
        p,s,trace=own_prefix_forward(m,t,w,4);gen=m.generate(t,w,dict(std=[1.,1.],mean=[0.,0.]),4)
        delta=gen['points'][...,:2]-torch.cat((torch.zeros_like(gen['points'][:,:1,:2]),gen['points'][:,:-1,:2]),1)
        torch.testing.assert_close(p.flatten(1,2),delta,atol=2e-7,rtol=1e-4)
        torch.testing.assert_close(s.argmax(-1).flatten(1),gen['points'][...,2:].argmax(-1))
        self.assertFalse(trace['used_history'].requires_grad)
    def test_continues_past_early_eoc_only_for_training_not_free_eval(self):
        m,t,w=self.setup_inputs();m.eval()
        with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([-5.,-5.,5.])
        p,s,tr=own_prefix_forward(m,t,w,4);gen=m.generate(t,w,dict(std=[1.,1.],mean=[0.,0.]),4)
        self.assertEqual(p.shape,(2,4,8,2));self.assertEqual(gen['stops'].tolist(),[1,1])
        self.assertTrue((s.argmax(-1)==2).all())
        self.assertTrue((tr['used_history'][:,1,16:].reshape(2,8,3).argmax(-1)==2).all())
        (p.square().mean()+s.square().mean()).backward()
        self.assertGreater(float(m.point_readout.weight.grad.abs().sum()),0.)
    def test_targets_affect_loss_but_never_own_history_prediction(self):
        m,t,w=self.setup_inputs();f=torch.randn(2,4,40);mask=torch.ones(2,4,8,dtype=torch.bool);mask[1,3,2:]=False
        targets=torch.randn(2,4,8,2);states=torch.zeros(2,4,8,dtype=torch.long)
        cfg=dict(pen_weights=[1.,3.8,8.],offset_stats=dict(std=[.2,.07]),pen_weight=.02,teacher_anchor_weight=.01)
        a=paired_losses(m,f,t,w,targets,states,mask,cfg);b=paired_losses(m,f+10,t,w,targets+4,states,mask,cfg)
        torch.testing.assert_close(a['own_trace']['used_history'],b['own_trace']['used_history'],atol=0,rtol=0)
        self.assertNotEqual(float(a['own_xy'].detach()),float(b['own_xy'].detach()))
        self.assertFalse(a['own_trace']['used_history'].requires_grad)
        (a['base']+.01*a['own_xy']+.01*a['own_pen']).backward()
        self.assertTrue(torch.isfinite(m.point_readout.weight.grad).all())
    def test_fixed_modest_gradient_coefficients_and_invalid_budget(self):
        c=auxiliary_coefficients(2.,20.,5.);self.assertAlmostEqual(c['xy']*20/2,.25);self.assertAlmostEqual(c['pen']*5/2,.10)
        m,t,w=self.setup_inputs()
        for n in [0,257,1.5]:
            with self.assertRaises(ValueError):own_prefix_forward(m,t,w,n)
        with self.assertRaises(ValueError):auxiliary_coefficients(1.,0.,1.)
    def test_stop_audit_uses_actual_early_stop_and_eos_index_without_source(self):
        m,t,w=self.setup_inputs();m.eval()
        with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([-5.,-5.,5.])
        gen=m.generate(t,w,dict(std=[1.,1.],mean=[0.,0.]),4)
        rows=free_stop_rows(gen,['a','b'],['abc','de'])
        self.assertEqual([r['generated_points'] for r in rows],[1,1]);self.assertTrue(all(r['found_eoc'] for r in rows))
        self.assertAlmostEqual(rows[1]['eos_attention_at_stop_or_cap'],float(gen['traces']['attention'][1,0,2]))
        self.assertTrue(all(r['no_source_length'] and r['clock_is_not_verified_glyph_alignment'] for r in rows))
    def test_stop_audit_reports_cap_as_cap_not_learned_stop(self):
        m,t,w=self.setup_inputs();m.eval()
        with torch.no_grad():m.point_readout.bias[2:]=torch.tensor([5.,0.,-30.])
        gen=m.generate(t,w,dict(std=[1.,1.],mean=[0.,0.]),4);rows=free_stop_rows(gen,['a','b'],['abc','de'])
        self.assertEqual([r['generated_points'] for r in rows],[32,32]);self.assertFalse(any(r['found_eoc'] for r in rows))
        self.assertAlmostEqual(rows[0]['center_at_learned_stop_or_cap'],float(gen['traces']['center'][0,3]))


if __name__=='__main__':unittest.main()
