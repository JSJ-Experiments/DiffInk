from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import yaml

from iam_tools.pen_ab import (POLICIES, WEIGHT_KEY, BIAS_KEY, check_config, policy_weights,
                              masked_focal, boundary_metrics, clone_pen_head, merge_head,
                              assert_nonpen_unchanged, run, capture, output_xy)
from iam_tools.check_batch import load_module
from test_vae_ctc import VAE, small_config

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'third_party/DiffInk' if (ROOT / 'third_party/DiffInk').exists() else ROOT


class PenABTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_exact_real_weights_and_padding_ignored(self):
        labels = torch.tensor([[0] * 544 + [1] * 36 + [2] + [2] * 3])
        mask = torch.arange(584)[None] < 581
        a = policy_weights(labels, mask, POLICIES[0])
        b = policy_weights(labels, mask, POLICIES[1])
        torch.testing.assert_close(a, torch.tensor([581 / 544, 581 / 36, 581.]))
        torch.testing.assert_close(b, torch.tensor([1., (544 / 36)**.5, 8.]))
        labels[~mask] = -99
        torch.testing.assert_close(a, policy_weights(labels, mask, POLICIES[0]))
        torch.testing.assert_close(b, policy_weights(labels, mask, POLICIES[1]))

    def test_focal_matches_upstream_on_real_points_in_both_arms(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 8, 3, requires_grad=True)
        targets = torch.tensor([[0, 0, 0, 1, 0, 2, 2, 2]])
        mask = torch.tensor([[True] * 6 + [False] * 2])
        gmm = load_module(REPO / 'model/gmm.py', 'pen_ab_test_gmm')
        for policy in POLICIES:
            alpha = policy_weights(targets, mask, policy)
            actual = masked_focal(logits, targets, mask, alpha)
            reference = gmm.FocalLoss(gamma=2)(logits[:, :6], targets[:, :6], alpha[None])
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        changed = logits.detach().clone(); changed[:, 6:] = float('nan')
        torch.testing.assert_close(masked_focal(changed, targets, mask, alpha), actual)
        actual.backward()
        self.assertEqual(logits.grad[:, 6:].count_nonzero(), 0)
        self.assertGreater(logits.grad[:, :6].count_nonzero(), 0)

    def test_boundary_metrics_not_majority_accuracy(self):
        truth = np.array([0] * 544 + [1] * 36 + [2])
        row = boundary_metrics(np.zeros(581), truth)
        self.assertAlmostEqual(row['pen_accuracy_not_gate'], 544 / 581)
        self.assertEqual(row['pen_up_f1'], 0)
        self.assertFalse(row['final_eoc_correct'])
        correct = boundary_metrics(truth, truth)
        for key in ['pen_up_precision', 'pen_up_recall', 'pen_up_f1']:
            self.assertEqual(correct[key], 1.)
        self.assertEqual(correct['non_final_false_eoc_count'], 0)
        prediction = np.array([1, 0, 2, 1, 0])
        row = boundary_metrics(prediction, np.array([0, 1, 0, 1, 2]))
        self.assertEqual((row['pen_up_tp'], row['pen_up_fp'], row['pen_up_fn']), (1, 1, 1))
        self.assertEqual(row['pen_up_f1'], .5)
        self.assertEqual(row['non_final_false_eoc_count'], 1)
        self.assertFalse(row['final_eoc_correct'])

    def test_separate_head_prevents_adamw_decay_or_moments_mutating_gmm(self):
        torch.manual_seed(42)
        fc = nn.Linear(7, 123).requires_grad_(False)
        source = {WEIGHT_KEY: fc.weight.detach().clone(), BIAS_KEY: fc.bias.detach().clone(),
                  'other': torch.randn(4)}
        a, b = clone_pen_head(fc), clone_pen_head(fc)
        self.assertTrue(all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items()))
        # Even intentionally excessive weight decay must be confined to standalone head.
        optimizer = torch.optim.AdamW(a.parameters(), lr=.01, weight_decay=.5)
        features = torch.randn(1, 16, 7)
        logits = a(features)
        labels = torch.tensor([[0] * 12 + [1] * 3 + [2]])
        mask = torch.ones_like(labels, dtype=torch.bool)
        loss = masked_focal(logits, labels, mask, policy_weights(labels, mask, POLICIES[1]))
        loss.backward(); optimizer.step()
        self.assertTrue(all(p.grad is None for p in fc.parameters()))
        merged = merge_head(source, a)
        self.assertTrue(assert_nonpen_unchanged(source, merged))
        self.assertFalse(torch.equal(source[WEIGHT_KEY][:3], merged[WEIGHT_KEY][:3]))
        self.assertTrue(torch.equal(source[WEIGHT_KEY], fc.weight))
        bad = deepcopy(merged); bad[WEIGHT_KEY][3, 0] += 1
        with self.assertRaises(AssertionError): assert_nonpen_unchanged(source, bad)
        bad = deepcopy(merged); bad['other'][0] += 1
        with self.assertRaises(AssertionError): assert_nonpen_unchanged(source, bad)

    def test_full_vae_forward_invariance_after_head_merge(self):
        torch.manual_seed(42)
        model = VAE(small_config()).eval().requires_grad_(False)
        data = torch.randn(1, 24, 5); data[:, :, 2:] = torch.tensor([1., 0., 0.])
        data[:, 10, 2:] = torch.tensor([0., 1., 0.]); data[:, -1, 2:] = torch.tensor([0., 0., 1.])
        batch = (data, torch.ones(1, 24, dtype=torch.bool), torch.tensor([[0, 1, 2]]), [], torch.tensor([0]))
        cfg = {'model_input_scale': .01}
        source = {k: v.clone() for k, v in model.state_dict().items()}
        with patch.object(model, 'get_ocr_loss', side_effect=AssertionError('CTC must stay off')), \
             patch.object(model, 'get_style_loss', side_effect=AssertionError('style must stay off')):
            features, before, nll = capture(model, batch, cfg)
            head = clone_pen_head(model.transformer_decoder.fc)
            with torch.no_grad(): head.weight.add_(.1); head.bias.add_(torch.tensor([.2, -.1, .3]))
            model.load_state_dict(merge_head(source, head))
            features_after, after, nll_after = capture(model, batch, cfg)
        self.assertTrue(torch.equal(features, features_after))
        self.assertTrue(torch.equal(before[:, 3:], after[:, 3:]))
        self.assertTrue(torch.equal(output_xy(before), output_xy(after)))
        self.assertEqual(nll, nll_after)
        self.assertTrue(assert_nonpen_unchanged(source, model.state_dict()))
        self.assertFalse(torch.equal(before[:, :3], after[:, :3]))

    def test_guards_checkpoint_and_cpu_bounds(self):
        cfg = yaml.safe_load((REPO / 'configs/vae_iam_pen_ab.yaml').read_text())
        check_config(cfg)
        for key, value in [('source_step', 1000), ('source_sha256', 'other'), ('device', 'cuda'),
                           ('max_steps', 1001), ('max_wall_seconds', 181), ('base_lr', .01), ('weight_cap', 9)]:
            changed = deepcopy(cfg); changed[key] = value
            with self.assertRaises(ValueError): check_config(changed)
        with self.assertRaises(ValueError): run('missing', 'missing', train=True)


if __name__ == '__main__':
    unittest.main()
