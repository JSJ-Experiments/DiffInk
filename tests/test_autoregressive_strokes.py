import unittest
import numpy as np
import torch
from iam_tools.autoregressive_strokes import MonotonicStrokeWriter,offset_fields,fit_offset_stats,encode_offsets,decode_offsets,StrokePool,reconstruction_terms

class AutoregressiveStrokeTests(unittest.TestCase):
    def model(self,adaptive=True):
        torch.manual_seed(5);return MonotonicStrokeWriter(4,2,.5,width=16,text_width=8,writer_width=4,adaptive=adaptive)
    def inputs(self):return torch.randn(2,5,40),torch.tensor([[0,1,2],[1,3,-1]]),torch.tensor([0,1])
    def test_teacher_does_not_see_current_or_future_target(self):
        m=self.model();f,t,w=self.inputs();a=m.teacher(f,t,w);f2=f.clone();f2[:,2:]+=13.;b=m.teacher(f2,t,w)
        torch.testing.assert_close(a[0][:,:3],b[0][:,:3],rtol=0,atol=0);self.assertGreater(float((a[0][:,3:]-b[0][:,3:]).detach().abs().max()),1e-5)
    def test_fixed_and_adaptive_identical_initial_predictions_and_positive_progress(self):
        a=self.model(True);b=self.model(False);b.load_state_dict(a.state_dict());f,t,w=self.inputs();oa,pa,ta=a.teacher(f,t,w);ob,pb,tb=b.teacher(f,t,w)
        torch.testing.assert_close(oa,ob,atol=1e-7,rtol=1e-6);torch.testing.assert_close(pa,pb,atol=1e-7,rtol=1e-6)
        self.assertTrue((ta['advance']>0).all());self.assertTrue((ta['center'][:,1:]>ta['center'][:,:-1]).all());self.assertTrue(torch.isfinite(ta['attention']).all());torch.testing.assert_close(ta['attention'].sum(-1),torch.ones(2,5))
    def test_batch_text_padding_invariance(self):
        m=self.model();f,t,w=self.inputs();o,p,tr=m.teacher(f,t,w);a,b,c=m.teacher(f[1:],t[1:,:2],w[1:]);torch.testing.assert_close(o[1:],a,atol=1e-6,rtol=1e-5);torch.testing.assert_close(p[1:],b,atol=1e-6,rtol=1e-5)
    def test_autoregressive_free_prefix_independent_of_budget_and_teacher(self):
        m=self.model().eval();_,t,w=self.inputs();stats=dict(mean=[0.,0.],std=[.1,.1])
        # Prevent learned stop for prefix test without changing inference code.
        with torch.no_grad():m.readout.bias[16:]=torch.tensor([3.,0.,-30.]*8)
        a=m.generate(t,w,stats,max_blocks=3);b=m.generate(t,w,stats,max_blocks=6)
        torch.testing.assert_close(a['points'],b['points'][:,:24],rtol=0,atol=0);self.assertFalse(a['found_eoc'].any());self.assertEqual(a['stops'].tolist(),[24,24])
    def test_learned_first_eoc_and_missing_not_forced(self):
        m=self.model().eval();_,t,w=self.inputs();stats=dict(mean=[0.,0.],std=[.1,.1])
        with torch.no_grad():m.readout.weight.zero_();m.readout.bias[16:]=torch.tensor([0.,0.,5.]*8)
        a=m.generate(t,w,stats,max_blocks=5);self.assertEqual(a['stops'].tolist(),[1,1]);self.assertTrue(a['found_eoc'].all());self.assertEqual(a['points'].shape[1],8)
    def targets(self):
        xy=torch.tensor([[0.,.2],[.2,.3],[.1,.5],[.2,.4],[.3,.4],[.5,.6],[.6,.3],[.7,.3],[.8,.4]])
        p=torch.nn.functional.one_hot(torch.tensor([0,0,1,0,0,1,0,0,2]),3).float();return torch.cat((xy,p),-1)
    def test_offset_roundtrip_including_penup_backtrack_partial_block(self):
        q=self.targets();d,p=offset_fields(q);stats=fit_offset_stats({'s':q},['s']);torch.testing.assert_close(decode_offsets(encode_offsets(d,stats),stats).cumsum(0),q[:,:2],atol=1e-7,rtol=1e-6)
        pool=StrokePool({'s':q},{'s':dict(text='ab')},['a','b'],['s'],stats);f,o,p,mask,t=pool.select(['s']);self.assertEqual(mask.sum().item(),9);self.assertEqual(mask.shape,(1,2,8));torch.testing.assert_close(decode_offsets(o.reshape(-1,2)[:9],stats).cumsum(0),q[:,:2],atol=1e-7,rtol=1e-6)
    def test_padding_excluded_even_nonfinite_predictions(self):
        q=self.targets();stats=fit_offset_stats({'s':q},['s']);pool=StrokePool({'s':q},{'s':dict(text='ab')},['a','b'],['s'],stats);f,o,p,mask,t=pool.select(['s']);pred=o.clone();pred[~mask]=float('nan');logits=torch.zeros(*mask.shape,3);logits[~mask]=float('nan');loss=reconstruction_terms(pred,logits,o,p,mask,torch.ones(3));self.assertEqual(float(loss['offset']),0.);self.assertTrue(torch.isfinite(loss['pen']))
    def test_train_only_stats_and_final_only_eoc_guard(self):
        q=self.targets()
        with self.assertRaises(ValueError):fit_offset_stats({'s':q,'held':q},['s'])
        bad=q.clone();bad[3,2:]=torch.tensor([0.,0.,1.])
        with self.assertRaises(ValueError):offset_fields(bad)
    def test_backward_clock_and_history_receive_gradients(self):
        m=self.model();f,t,w=self.inputs();o,p,tr=m.teacher(f,t,w);(o.square().mean()+p.square().mean()).backward();self.assertGreater(float(m.clock.weight.grad.abs().sum()),0.);self.assertGreater(float(m.history.weight_ih.grad.abs().sum()),0.)
    def test_generation_rejects_train_mode_or_invalid_budget_text(self):
        m=self.model();_,t,w=self.inputs();stats=dict(mean=[0.,0.],std=[.1,.1])
        with self.assertRaises(ValueError):m.generate(t,w,stats)
        m.eval()
        for budget in [0,257,1.2]:
            with self.assertRaises(ValueError):m.generate(t,w,stats,max_blocks=budget)
        for text in [torch.tensor([[0,-1,1]]),torch.tensor([[-1,-1]]),torch.tensor([[4]])]:
            with self.assertRaises(ValueError):m.memory(text,torch.tensor([0]))

if __name__=='__main__':unittest.main()
