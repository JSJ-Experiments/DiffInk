import importlib.util
from pathlib import Path
import unittest
import torch
from iam_tools.inkvae import logspace_gmm_nll,train_opt_in

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

    def test_explicit_training_ack_required(self):
        with self.assertRaises(ValueError):train_opt_in('missing','missing',allow_experimental=False)
        spec=importlib.util.spec_from_file_location('modal_smoke_guard',ROOT/'modal_inkvae.py')
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertFalse(m.require_opt_in(False,False))
        with self.assertRaises(ValueError):m.require_opt_in(True,False)
        self.assertTrue(m.require_opt_in(True,True))

if __name__=='__main__':unittest.main()
