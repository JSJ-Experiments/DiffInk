import unittest
import numpy as np
import torch
from iam_tools.weak_alignment import (packed_labels,PackedAlignmentPool,attention_log_probs,
    capture_last_alignment,alignment_loss,alignment_summary)
from iam_tools.generation_prefix_contract import PrefixContractWriter


class WeakAlignmentTests(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(214)
        m=PrefixContractWriter(causal_queries=True,alignment=True,writer_count=2,vocab_size=6,width=16,heads=4,depth=2).eval()
        torch.nn.init.normal_(m.final[-1].weight,std=.1)
        x=torch.zeros(2,4,384);mask=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool)
        text=torch.tensor([[0,1,1],[2,3,-1]])
        return m,(x,torch.ones(2),text,mask),torch.tensor([0,1])

    def test_frames_packed_no_blank_interpolation_or_trailing_labels(self):
        np.testing.assert_array_equal(packed_labels(np.array([0,-1,1,2,-1]),3,17),[[0,-1],[1,2],[-1,-1]])
        np.testing.assert_array_equal(packed_labels(np.array([0]),1,1),[[0,-1]])
        for a,n,c in [(np.array([0,1]),5,1),(np.array([0.]),1,1),(np.array([-2]),1,1),(np.array([1]),1,1)]:
            with self.assertRaises(ValueError):packed_labels(a,c,n)

    def test_pool_exact_train_only_and_minimal_padding(self):
        labels={'a':np.array([[0,-1]]),'b':np.array([[0,1],[1,-1]])}
        pool=PackedAlignmentPool(labels,['b','a'])
        self.assertEqual(pool.select(['a']).shape,(1,1,2))
        self.assertEqual(pool.select(['a','b'])[0,1].tolist(),[-1,-1])
        with self.assertRaises(ValueError):pool.select(['held'])
        with self.assertRaises(ValueError):PackedAlignmentPool(labels,['a'])

    def test_readout_matches_mha_weights_with_gaussian_and_padding(self):
        m,args,_=self.fixture();att=m.blocks[-1].cross_attention
        q=torch.randn(2,4,16);k=torch.randn(2,5,16)
        bias=torch.randn(8,4,5);bias[:,:,4]=float('-inf')
        _,weights=att(q,k,k,attn_mask=bias,need_weights=True,average_attn_weights=False)
        p=attention_log_probs(att,q,k,attn_mask=bias).exp()
        torch.testing.assert_close(p,weights,atol=1e-7,rtol=1e-6)
        pad=torch.tensor([[0,0,0,0,1],[0,0,0,1,1]],dtype=torch.bool)
        _,weights=att(q,k,k,key_padding_mask=pad,need_weights=True,average_attn_weights=False)
        torch.testing.assert_close(attention_log_probs(att,q,k,key_padding_mask=pad).exp(),weights,atol=1e-7,rtol=1e-6)
        with self.assertRaises(ValueError):attention_log_probs(att,q,k,attn_mask=torch.zeros(3,3))

    def test_readout_does_not_change_output_state_or_rng_and_removes_hook(self):
        m,args,wi=self.fixture();state={k:v.clone() for k,v in m.state_dict().items()};before=torch.get_rng_state().clone()
        a=m(*args,writer_ids=wi)
        with capture_last_alignment(m) as c:b=m(*args,writer_ids=wi)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in state.items()))
        self.assertEqual(c['log_probs'].shape,(2,4,4,4))
        self.assertFalse(m.blocks[-1].cross_attention._forward_pre_hooks)
        try:
            with capture_last_alignment(m):raise RuntimeError('test')
        except RuntimeError:pass
        self.assertFalse(m.blocks[-1].cross_attention._forward_pre_hooks)

    def test_fractional_labels_bos_and_global_head(self):
        p=torch.tensor([[[[.1,.2,.7],[.1,.3,.6]],[[.1,.2,.7],[.1,.3,.6]],[[.9,.05,.05],[.9,.05,.05]]]])
        labels=torch.tensor([[[0,1],[1,-1]]]);text=torch.tensor([[1,2]]);mask=torch.ones(1,2,dtype=torch.bool)
        loss=alignment_loss(p.log(),labels,mask,text)
        expect=-(.5*(np.log(.2)+np.log(.7))+np.log(.6))/2
        self.assertAlmostEqual(float(loss),expect,places=6)
        p=p.clone().requires_grad_();alignment_loss(p.log(),labels,mask,text).backward()
        self.assertTrue((p.grad[:,2]==0).all());self.assertGreater(float(p.grad[:,:2].abs().sum()),0)
        dup=torch.tensor([[[0,0],[1,1]]]);single=torch.tensor([[[0,-1],[1,-1]]])
        self.assertEqual(float(alignment_loss(p.log(),dup,mask,text)),float(alignment_loss(p.log(),single,mask,text)))

    def test_nonvacuous_gradients_only_routing_not_readout_or_v_rows(self):
        m,args,wi=self.fixture();labels=torch.tensor([[[0,-1],[1,2],[2,-1],[2,2]],[[0,-1],[1,-1],[1,1],[-1,-1]]])
        with capture_last_alignment(m) as c:m(*args,writer_ids=wi)
        loss=alignment_loss(c['log_probs'],labels,args[3],args[2]);loss.backward()
        grad=m.blocks[-1].cross_attention.in_proj_weight.grad
        self.assertGreater(float(grad[:32].abs().sum()),.01);self.assertTrue((grad[32:]==0).all())
        self.assertIsNone(m.final[-1].weight.grad);self.assertIsNone(m.blocks[-1].cross_attention.out_proj.weight.grad)
        self.assertGreater(float(m.text.weight.grad.abs().sum()),0)
        summary=alignment_summary(c['log_probs'].detach(),labels,args[3],args[2]);self.assertEqual(summary['supervised_queries'],7)
        self.assertGreater(summary['cross_entropy'],0)

    def test_padding_rejected_and_no_labels_zero_finite_gradients(self):
        logp=torch.tensor([[[[0.,float('-inf'),float('-inf')]]]*2],requires_grad=True)
        blank=torch.full((1,1,2),-1);mask=torch.ones(1,1,dtype=torch.bool);text=torch.tensor([[0,-1]])
        loss=alignment_loss(logp,blank,mask,text);self.assertEqual(float(loss),0);loss.backward();self.assertTrue(torch.isfinite(logp.grad).all())
        for lab,ma in [(torch.tensor([[[1,-1]]]),mask),(torch.tensor([[[0,-1]]]),~mask),(torch.tensor([[[-2,-1]]]),mask)]:
            with self.assertRaises(ValueError):alignment_loss(logp,lab,ma,text)

    def test_capture_preserves_prefix_and_checkpoint_contract(self):
        m,args,wi=self.fixture()
        with capture_last_alignment(m) as c:a=m(*args,writer_ids=wi)
        x,time,text,mask=args;long=torch.zeros(2,9,384)
        with capture_last_alignment(m) as c:b=m(long,time,text,torch.ones(2,9,dtype=torch.bool),writer_ids=wi)
        torch.testing.assert_close(a[0],b[0,:4],rtol=1e-5,atol=2e-6)
        n=PrefixContractWriter(**m.config).eval();n.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(a,n(*args,writer_ids=wi)))
