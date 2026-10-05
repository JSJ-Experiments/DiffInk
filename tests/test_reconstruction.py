from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from iam_tools.reconstruction import (check_config, expected_xy, masked_xy_mse, temporal_errors,
                                      geometry_gate, objectives, initialize, deterministic_forward, train_pair)
from iam_tools.pen_ab import WEIGHT_KEY, BIAS_KEY
from test_vae_ctc import VAE, small_config

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/DiffInk' if (ROOT / 'third_party/DiffInk').exists() else ROOT


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def cfg(self):
        return yaml.safe_load((REPO / 'configs/vae_iam_reconstruction.yaml').read_text())

    def batch(self):
        data = torch.randn(1, 24, 5)
        data[:, :, 2:] = torch.tensor([1., 0., 0.])
        data[:, 10, 2:] = torch.tensor([0., 1., 0.])
        data[:, 21:, 2:] = torch.tensor([0., 0., 1.])
        mask = torch.arange(24)[None] < 22
        return data, mask, torch.tensor([[0, 1, 2]]), [], torch.tensor([0])

    def test_mixture_expectation_and_padding_gradient(self):
        output = torch.zeros(1, 123, 4, requires_grad=True)
        with torch.no_grad():
            output[:, 23:43] = 2
            output[:, 43:63] = -3
        xy = expected_xy(output)
        torch.testing.assert_close(xy, torch.tensor([[[2., -3.]] * 4]))
        target = torch.ones_like(xy)
        mask = torch.tensor([[True, True, True, False]])
        target[~mask] = float('nan')
        loss = masked_xy_mse(xy, target, mask)
        self.assertAlmostEqual(float(loss.detach()), 8.5)
        loss.backward()
        self.assertEqual(output.grad[:, :3].count_nonzero(), 0)
        self.assertEqual(output.grad[:, 63:].count_nonzero(), 0)
        self.assertEqual(output.grad[:, :, 3].count_nonzero(), 0)
        self.assertGreater(output.grad[:, 23:63, :3].count_nonzero(), 0)
        # Unequal means ensure softmax mixture probabilities have real gradients too.
        second = torch.randn(1, 123, 4, requires_grad=True)
        masked_xy_mse(expected_xy(second), torch.zeros_like(xy), mask).backward()
        self.assertGreater(second.grad[:, 3:23].count_nonzero(), 0)

    def test_stroke_aware_differences_exclude_jumps_and_cross_stroke_windows(self):
        truth = np.array([[0, 0], [1, 0], [2, 0], [100, 0], [101, 0], [102, 0]], dtype=float)
        prediction = truth.copy(); prediction[3:] += [50, 100]
        states = np.array([0, 0, 1, 0, 0, 2])
        row = temporal_errors(prediction, truth, states)
        self.assertEqual(row['velocity']['within_true_strokes']['windows'], 4)
        self.assertEqual(row['second_difference']['within_true_strokes']['windows'], 2)
        self.assertEqual(row['velocity']['within_true_strokes']['vector_rmse'], 0)
        self.assertEqual(row['second_difference']['within_true_strokes']['vector_rmse'], 0)
        self.assertGreater(row['velocity']['all_points_including_penup_jumps']['vector_rmse'], 0)

    def test_small_point_jitter_is_visible_in_differences_and_singletons_safe(self):
        truth = np.column_stack([np.arange(9), np.zeros(9)])
        predicted = truth.copy(); predicted[:, 1] = .1 * (-1)**np.arange(9)
        row = temporal_errors(predicted, truth, np.array([0]*8 + [2]))
        self.assertAlmostEqual(row['velocity']['within_true_strokes']['vector_rmse'], .2)
        self.assertAlmostEqual(row['second_difference']['within_true_strokes']['vector_rmse'], .4)
        short = temporal_errors(np.zeros((1, 2)), np.zeros((1, 2)), np.array([2]))
        self.assertIsNone(short['second_difference']['within_true_strokes']['vector_rmse'])

    def test_strict_geometry_gate_not_absolute_rmse_or_nll_gate(self):
        baseline = {'axes': {'x': {'rmse_model_units': .059, 'correlation': .9999},
                             'y': {'rmse_model_units': .031, 'correlation': .976}}}
        self.assertTrue(geometry_gate(baseline, baseline)['passed'])
        for axis, key, value in [('x', 'rmse_model_units', .06), ('y', 'correlation', .975)]:
            bad = deepcopy(baseline); bad['axes'][axis][key] = value
            self.assertFalse(geometry_gate(baseline, bad)['passed'])

    def test_actual_deterministic_branch_has_gradients_without_aux_pen_or_variance(self):
        torch.manual_seed(42); model = VAE(small_config())
        parent = {'model_state_dict': deepcopy(model.state_dict())}
        cfg = self.cfg(); batch = self.batch()
        initialize(model, parent, 'deterministic_mse', cfg, None)
        self.assertFalse(model.training)
        with patch.object(model, 'get_ocr_loss', side_effect=AssertionError('CTC off')), \
             patch.object(model, 'get_style_loss', side_effect=AssertionError('style off')), \
             patch.object(model, 'reparameterize', side_effect=AssertionError('latent mean only')):
            loss, terms, output = objectives(model, batch, cfg, {'model_input_scale': .01}, 'deterministic_mse', 'cpu')
            loss.backward()
            after = deterministic_forward(model, batch, .01)
        self.assertTrue(torch.equal(output, after))
        self.assertEqual(set(terms), {'expected_xy_mse'})
        self.assertTrue(all(p.grad is None for module in (model.conv_logvar, model.ocr_model, model.style_classifier) for p in module.parameters()))
        self.assertGreater(model.transformer_decoder.fc.weight.grad[3:63].count_nonzero(), 0)
        self.assertEqual(model.transformer_decoder.fc.weight.grad[:3].count_nonzero(), 0)
        self.assertEqual(model.transformer_decoder.fc.weight.grad[63:].count_nonzero(), 0)
        self.assertGreater(model.conv_mu.weight.grad.count_nonzero(), 0)

    def test_joint_inserts_only_trained_b_rows_and_uses_bounded_focal(self):
        torch.manual_seed(42); model = VAE(small_config())
        parent = {'model_state_dict': deepcopy(model.state_dict())}
        b = {WEIGHT_KEY: torch.randn_like(model.transformer_decoder.fc.weight[:3]),
             BIAS_KEY: torch.randn_like(model.transformer_decoder.fc.bias[:3])}
        cfg = self.cfg(); initialize(model, parent, 'joint', cfg, b)
        self.assertTrue(torch.equal(model.transformer_decoder.fc.weight[:3], b[WEIGHT_KEY]))
        self.assertTrue(torch.equal(model.transformer_decoder.fc.weight[3:], parent['model_state_dict'][WEIGHT_KEY][3:]))
        with patch.object(model, 'get_ocr_loss', side_effect=AssertionError('CTC off')), \
             patch.object(model, 'get_style_loss', side_effect=AssertionError('style off')):
            loss, terms, _ = objectives(model, self.batch(), cfg, {'model_input_scale': .01}, 'joint', 'cpu')
            loss.backward()
        torch.testing.assert_close(loss, terms['gmm_nll'] + terms['pen_focal'])
        self.assertGreater(model.transformer_decoder.fc.weight.grad[:3].count_nonzero(), 0)
        self.assertTrue(all(p.grad is None for module in (model.ocr_model, model.style_classifier) for p in module.parameters()))
        # The next branch must reset to ORIGINAL step-2000 rows, not carry joint state.
        initialize(model, parent, 'deterministic_mse', cfg, None)
        self.assertTrue(torch.equal(model.transformer_decoder.fc.weight, parent['model_state_dict'][WEIGHT_KEY]))

    def test_branch_reset_clears_stale_gradients_in_newly_frozen_parameters(self):
        model = VAE(small_config()); parent = {'model_state_dict': deepcopy(model.state_dict())}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        cfg = self.cfg(); initialize(model, parent, 'deterministic_mse', cfg, None)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        loss, _, _ = objectives(model, self.batch(), cfg, {'model_input_scale': .01}, 'deterministic_mse', 'cpu')
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.conv_logvar.parameters()))

    def test_caps_auxiliary_and_modal_guards(self):
        cfg = self.cfg(); check_config(cfg)
        for key, value in [('joint_steps', 301), ('mse_steps', 1001), ('max_wall_seconds', 601),
                           ('ctc_weight', 1), ('derivative_loss_weight', .1), ('mse_latent', 'sampled'), ('source_sha256', 'other')]:
            bad = deepcopy(cfg); bad[key] = value
            with self.assertRaises(ValueError): check_config(bad)
        with self.assertRaises(ValueError): train_pair('missing', 'missing')
        spec = importlib.util.spec_from_file_location('modal_reconstruction_guard', ROOT / 'modal_reconstruction.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertFalse(module.require_opt_in(False, False))
        with self.assertRaises(ValueError): module.require_opt_in(True, False)
        self.assertTrue(module.require_opt_in(True, True))


if __name__ == '__main__':
    unittest.main()
