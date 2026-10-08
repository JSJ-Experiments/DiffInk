import unittest
import torch
from iam_tools.generation_prefix_contract import PrefixContractWriter
from iam_tools.generation_position_contract import PositionContractWriter


class PrefixContractTests(unittest.TestCase):
    def fixture(self, causal=True):
        torch.manual_seed(72)
        model=PrefixContractWriter(causal_queries=causal, alignment=True, writer_count=2,
            vocab_size=5, width=16, depth=3, heads=4, blocks_per_character=1.2).eval()
        # Zero-initialized readout would make every invariance test vacuous.
        torch.nn.init.normal_(model.final[-1].weight,std=.1)
        return model

    def call(self, m, n, *, valid=None, batch=1, suffix=None, null=False):
        x=torch.zeros(batch,n,384);mask=torch.arange(n)[None].expand(batch,-1)<(n if valid is None else valid)
        if suffix is not None:x[:,suffix:]=torch.randn_like(x[:,suffix:])*10
        text=torch.tensor([[0,1,2]]).expand(batch,-1)
        return m(x,torch.ones(batch),text,mask,writer_ids=torch.zeros(batch,dtype=torch.long),
                 drop_text=torch.full((batch,),null,dtype=torch.bool))

    def test_control_exact_old_absolute_and_identical_parameter_keys(self):
        a=self.fixture(False);cfg=dict(a.config);cfg.pop('causal_queries')
        old=PositionContractWriter(**cfg).eval();old.load_state_dict(a.state_dict())
        args=(torch.zeros(1,6,384),torch.ones(1),torch.tensor([[0,1,2]]),torch.ones(1,6,dtype=torch.bool))
        w=torch.zeros(1,dtype=torch.long)
        self.assertTrue(torch.equal(old(*args,writer_ids=w),a(*args,writer_ids=w)))
        b=self.fixture(True)
        self.assertEqual(set(a.state_dict()),set(b.state_dict()))
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))

    def test_extended_and_truncated_budget_prefix_invariant(self):
        m=self.fixture();a=self.call(m,6)
        for n in [7,12,20]:torch.testing.assert_close(a,self.call(m,n)[:,:6],rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(a[:,:3],self.call(m,3),rtol=1e-5,atol=2e-6)
        self.assertGreater(float(a.detach().std()),.01)
        m=self.fixture(False)
        self.assertGreater(float((self.call(m,6)-self.call(m,12)[:,:6]).detach().abs().max()),1e-4)

    def test_future_inputs_cannot_affect_prefix(self):
        m=self.fixture();a=self.call(m,12)
        b=self.call(m,12,suffix=6)
        torch.testing.assert_close(a[:,:6],b[:,:6],rtol=0,atol=0)
        self.assertGreater(float((a[:,6:]-b[:,6:]).detach().abs().max()),1e-4)

    def test_padding_batch_and_null_prefix(self):
        m=self.fixture()
        for null in [False,True]:
            a=self.call(m,6,null=null)
            b=self.call(m,12,valid=6,batch=2,null=null)
            torch.testing.assert_close(a,b[:1,:6],rtol=1e-5,atol=2e-6)
            self.assertTrue((b[:,6:]==0).all());self.assertTrue(torch.isfinite(b).all())

    def test_mask_orientation_and_gradients(self):
        m=self.fixture();x=torch.randn(1,8,384,requires_grad=True)
        y=m(x,torch.ones(1),torch.tensor([[0,1,2]]),torch.ones(1,8,dtype=torch.bool),writer_ids=torch.zeros(1,dtype=torch.long))
        y[:,:3].square().sum().backward()
        self.assertGreater(float(x.grad[:,:3].abs().sum()),0)
        self.assertTrue((x.grad[:,3:]==0).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_roundtrip_and_explicit_contract(self):
        m=self.fixture();n=PrefixContractWriter(**m.config).eval();n.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(self.call(m,6),self.call(n,6)))
        for kw in [dict(causal_queries=1),dict(position_policy='relative100'),dict(alignment=False)]:
            with self.assertRaises(ValueError):PrefixContractWriter(writer_count=1,vocab_size=2,**kw)

    def test_paths_are_separate_and_guarded(self):
        from iam_tools.generation_prefix_contract_study import checked_path
        self.assertEqual(str(checked_path('checkpoints/iam_generation_prefix_contract/run','data')),'data/checkpoints/iam_generation_prefix_contract/run')
        for p in ['other/run','checkpoints/iam_generation_position_contract/run','checkpoints/iam_generation_prefix_contract/../bad']:
            with self.assertRaises(ValueError):checked_path(p,'data')
