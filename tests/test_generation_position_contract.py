import unittest
import torch
from iam_tools.generation_alignment import AlignedWriterDenoiser
from iam_tools.generation_position_contract import PositionContractWriter, contract_lr


class PositionContractTests(unittest.TestCase):
    def fixture(self, policy):
        torch.manual_seed(72)
        model = PositionContractWriter(position_policy=policy, alignment=True, writer_count=2,
            vocab_size=5, width=16, depth=2, heads=4, blocks_per_character=1.2)
        torch.nn.init.normal_(model.final[-1].weight, std=.02)
        model.eval()
        args = (torch.zeros(2, 6, 384), torch.ones(2), torch.tensor([[0,1,2],[1,2,-1]]),
                torch.tensor([[1]*6,[1,1,1,1,0,0]], dtype=torch.bool))
        return model, args, torch.tensor([0,1])

    def test_control_exactly_matches_existing_soft_model(self):
        m, args, w = self.fixture('relative100')
        cfg = dict(m.config); cfg.pop('position_policy')
        old = AlignedWriterDenoiser(**cfg).eval(); old.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(old(*args, writer_ids=w), m(*args, writer_ids=w)))

    def test_identical_fresh_parameters_but_distinct_features(self):
        a, args, w = self.fixture('relative100'); b, _, _ = self.fixture('absolute')
        self.assertEqual(set(a.state_dict()), set(b.state_dict()))
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertGreater(float((a(*args, writer_ids=w)-b(*args, writer_ids=w)).detach().abs().max()), 1e-5)

    def test_absolute_query_features_do_not_depend_on_requested_length(self):
        m, args, _ = self.fixture('absolute')
        original = m.query_positions(6, args[-1], torch.float32)
        extended = m.query_positions(7, torch.ones(2,7,dtype=torch.bool), torch.float32)
        self.assertTrue(torch.equal(original, extended[:,:6]))
        control, _, _ = self.fixture('relative100')
        self.assertFalse(torch.equal(control.query_positions(6,args[-1],torch.float32),
                                    control.query_positions(7,torch.ones(2,7,dtype=torch.bool),torch.float32)[:,:6]))

    def test_absolute_mask_null_and_gradients(self):
        m,args,w = self.fixture('absolute')
        y=m(*args,drop_text=torch.ones(2,dtype=torch.bool),writer_ids=w)
        self.assertTrue(torch.isfinite(y).all()); self.assertTrue((y[1,4:]==0).all())
        y.square().sum().backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))

    def test_roundtrip_preserves_contract_and_rejects_invalid(self):
        m,args,w=self.fixture('absolute'); n=PositionContractWriter(**m.config).eval();n.load_state_dict(m.state_dict())
        self.assertTrue(torch.equal(m(*args,writer_ids=w),n(*args,writer_ids=w)))
        with self.assertRaises(ValueError):PositionContractWriter(position_policy='unknown')
        with self.assertRaises(ValueError):PositionContractWriter(alignment=False,writer_count=1,vocab_size=2)

    def test_exact_bounded_schedule(self):
        from iam_tools.generation_composition import learning_rate
        for step in [1,1000,8000,16000,24000]:self.assertEqual(contract_lr(step),learning_rate(step,24000))
        for step in [24001,36000,48000]:self.assertEqual(contract_lr(step),1e-5)
        for step in [0,48001,1.5,True]:
            with self.assertRaises(ValueError):contract_lr(step)

    def test_duration_refit_allows_only_tiny_blas_roundoff(self):
        import copy
        from iam_tools.generation_composition import fit_duration
        from iam_tools.generation_position_contract_study import verify_duration_refit
        records={'a':dict(text='hello',writer_id='w',points=200),'b':dict(text='world',writer_id='w',points=232)}
        d=fit_duration(records,['a','b'],['w']);b=copy.deepcopy(d);b['coefficients'][0]+=1e-14
        verify_duration_refit(b,d,records,['a','b'])
        b['coefficients'][0]+=.01
        with self.assertRaises(ValueError):verify_duration_refit(b,d,records,['a','b'])

    def test_duration_scope_and_rounded_predictions_remain_exact(self):
        import copy
        from iam_tools.generation_composition import fit_duration
        from iam_tools.generation_position_contract_study import verify_duration_refit, checked_path
        records={'a':dict(text='hello',writer_id='w',points=200),'b':dict(text='world',writer_id='w',points=232)}
        d=fit_duration(records,['a','b'],['w']);b=copy.deepcopy(d);b['train_ids'].reverse()
        with self.assertRaises(ValueError):verify_duration_refit(b,d,records,['a','b'])
        for p in ['other/x','checkpoints/iam_generation_position_contract/../x','/checkpoints/iam_generation_position_contract/x']:
            with self.assertRaises(ValueError):checked_path(p,'data')
