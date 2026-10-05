import importlib.util
import json
import tempfile
from pathlib import Path
import unittest
import torch
from iam_tools.inkvae import logspace_gmm_nll,train_opt_in,fixed_evaluation,greedy_ctc,edit_distance

ROOT=Path(__file__).resolve().parents[1]

class SmokeRunnerTests(unittest.TestCase):
    def test_logspace_density_matches_ordinary_gaussian(self):
        pi=torch.ones(1,1,2)
        zero=torch.zeros(1,1,2);one=torch.ones(1,1,2)
        result=logspace_gmm_nll(pi,zero,zero,one,one,zero,zero,zero)
        torch.testing.assert_close(result,torch.full((1,2),1.8378770664))

    def test_far_coordinates_keep_finite_nonzero_mean_gradient(self):
        # Static loss gradient test only; no optimizer or parameter update.
        mean=torch.zeros(1,1,2,requires_grad=True)
        one=torch.ones_like(mean);zero=torch.zeros_like(mean)
        loss=logspace_gmm_nll(one,mean,zero,one,one,zero,one*100,zero).mean()
        grad=torch.autograd.grad(loss,mean)[0]
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(grad).all())
        self.assertTrue((grad!=0).all())

    def test_fixed_evaluation_reuses_noise_and_preserves_training_rng(self):
        model=torch.nn.Linear(2,2).train()
        torch.manual_seed(123)
        state=torch.random.get_rng_state().clone()
        with fixed_evaluation(model,1042,'cpu'):
            first=torch.randn(10)
            self.assertFalse(model.training)
            self.assertFalse(torch.is_grad_enabled())
        self.assertTrue(model.training)
        torch.testing.assert_close(state,torch.random.get_rng_state())
        with fixed_evaluation(model,1042,'cpu'):second=torch.randn(10)
        torch.testing.assert_close(first,second)
        with self.assertRaises(RuntimeError):
            with fixed_evaluation(model,1042,'cpu'):raise RuntimeError('test cleanup')
        self.assertTrue(model.training)
        torch.testing.assert_close(state,torch.random.get_rng_state())

    def test_ctc_collapse_and_edit_distance(self):
        self.assertEqual(greedy_ctc([1,1,0,1,2,2,0],['a','b']),'aab')
        self.assertEqual(edit_distance('hello','helo'),1)
        self.assertEqual(edit_distance('','abc'),3)

    def test_report_rejects_incomplete_metrics(self):
        from iam_tools.report_inkvae import report
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'result.json').write_text(json.dumps({'steps':2}))
            (root/'metrics.jsonl').write_text(json.dumps({'step':1,'total':1})+'\n')
            (root/'fixed_metrics.jsonl').write_text('')
            with self.assertRaisesRegex(ValueError,'incomplete'):report(root)

    def test_explicit_training_ack_required(self):
        with self.assertRaises(ValueError):train_opt_in('missing','missing',allow_experimental=False)
        spec=importlib.util.spec_from_file_location('modal_smoke_guard',ROOT/'modal_inkvae.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertFalse(m.require_opt_in(False,False))
        with self.assertRaises(ValueError):m.require_opt_in(True,False)
        self.assertTrue(m.require_opt_in(True,True))

if __name__=='__main__':unittest.main()
