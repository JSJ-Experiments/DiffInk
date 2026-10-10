import copy
import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from iam_tools.corpus_dit import learning_rate
from iam_tools.corpus_dit_study import schedule
from iam_tools.report_wsl_conditioning import (
    checked_points, controls, verify_denoising, verify_matched_draws,
    verify_terminal, verify_updates,
)


class NativeReportTests(unittest.TestCase):
    def fixture(self):
        cfg = dict(max_updates=3, batch=2, schedule_seed=5, clip=1.,
                   warmup_updates=1, lr=5e-5, min_lr=1e-6,
                   max_train_wall_seconds=1800, eval_steps=[0, 1, 3],
                   diffusion_steps=1000, backend='native')
        ids = ['a', 'b', 'c']
        data = dict(scope=dict(splits=dict(train=ids)),
                    records={i: dict(points=32) for i in ids})
        batches = list(schedule(ids, dict.fromkeys(ids, 4), 3, 2, 5))
        rows = [dict(step=j, sample_ids=b, objective=1., xy16_x0_mse=1.,
                     pen24_x0_mse=1., raw_grad_norm=.5, clipped=False,
                     lr=learning_rate(j, cfg), train_seconds=float(j),
                     text_dropped=False, prefix_retained=False,
                     active_tokens=4*len(b), timesteps=[j]*len(b),
                     gpu_peak_allocated_bytes=100, data_draw_digest='a'*64)
                for j, b in enumerate(batches, 1)]
        order = verify_updates(cfg, data, rows, 3)
        history = [dict(step=s, aggregate={'dev_unseen_text': {'1.0': {'cer':v}}})
                   for s, v in [(0, 1.), (1, .7), (3, .8)]]
        result = dict(arm='concat', last_step=3, training_order_sha256=order,
                      history=history, best_step=1, best_dev_cer=.7,
                      stop='budget_completed', train_seconds=3., clip_fraction=0.,
                      backend='native', no_confirmations_opened=True,
                      not_promoted=True, codec_reader_unchanged=True)
        return cfg, data, rows, history, result

    def test_completed_prefix_not_a_final_claim(self):
        cfg, data, rows, _, _ = self.fixture()
        verify_updates(cfg, data, rows[:1], 1)
        verify_updates(cfg, data, [], 0)
        with self.assertRaisesRegex(ValueError, 'sequential'):
            verify_updates(cfg, data, rows[:1], 3)

    def test_update_drift_fails_closed(self):
        for key, value in [('prefix_retained', True), ('objective', .1),
                           ('sample_ids', ['held']), ('lr', .1),
                           ('timesteps', [1000]), ('active_tokens', 1),
                           ('clipped', True), ('train_seconds', float('nan')),
                           ('data_draw_digest', 'missing')]:
            cfg, data, rows, _, _ = self.fixture()
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_updates(cfg, data, rows, 3)

    def test_time_cannot_decrease(self):
        cfg, data, rows, _, _ = self.fixture()
        rows[1]['train_seconds'] = .1
        with self.assertRaisesRegex(ValueError, 'monotonic'):
            verify_updates(cfg, data, rows, 3)

    def test_matched_actual_rng_not_just_schedule(self):
        _, _, rows, _, _ = self.fixture()
        self.assertTrue(verify_matched_draws({'concat': rows})['comparison_pending'])
        b = copy.deepcopy(rows)
        self.assertEqual(verify_matched_draws({'concat': rows, 'joint': b})['compared_updates'], 3)
        b[1]['data_draw_digest'] = 'b'*64
        with self.assertRaisesRegex(ValueError, 'DATA draws'):
            verify_matched_draws({'concat': rows, 'joint': b})

    def test_matched_prefix_allows_different_progress(self):
        _, _, rows, _, _ = self.fixture()
        self.assertEqual(verify_matched_draws({'concat': rows, 'joint': copy.deepcopy(rows[:1])})['compared_updates'], 1)

    def test_terminal_selection_and_frozen_contract(self):
        cfg, _, rows, hist, result = self.fixture()
        verify_terminal(cfg, result, rows, hist, result['training_order_sha256'])
        for key, value in [('best_step', 3), ('no_confirmations_opened', False),
                           ('codec_reader_unchanged', False), ('clip_fraction', .1),
                           ('stop', 'wall_limit'), ('backend', 'T4')]:
            r = copy.deepcopy(result)
            r[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify_terminal(cfg, r, rows, hist, result['training_order_sha256'])

    def test_terminal_cannot_skip_evaluations(self):
        cfg, _, rows, hist, result = self.fixture()
        result['history'] = hist[1:]
        with self.assertRaisesRegex(ValueError, 'timeline'):
            verify_terminal(cfg, result, rows, hist, result['training_order_sha256'])

    def test_terminal_wall_cap_can_have_different_actual_milestones(self):
        cfg, _, rows, hist, result = self.fixture()
        cfg['max_train_wall_seconds'] = 1.5
        hist[-1]['step'] = 2
        result.update(arm='joint', last_step=2, stop='wall_limit', train_seconds=2.)
        verify_terminal(cfg, result, rows[:2], hist, result['training_order_sha256'])

    def diagnostic(self):
        cfg = dict(eval_train_ids=['a'], eval_dev_ids=['b'])
        rows = [dict(split=s, sample_id=sid, timestep=t, policy=p,
                     xy16_mse=1., pen24_mse=2., active40_mse=1.6)
                for s, sid in [('train', 'a'), ('dev_unseen_text', 'b')]
                for t in [0, 10, 100, 500, 900, 999]
                for p in ['correct', 'rotated_character_order', 'null']]
        summary = {s:{str(t):{p:dict(xy16_mse=1., pen24_mse=2., active40_mse=1.6)
                              for p in ['correct', 'rotated_character_order', 'null']}
                      for t in [0, 10, 100, 500, 900, 999]}
                   for s in ['train', 'dev_unseen_text']}
        return cfg, dict(step=3, rows=rows, summary=summary,
                         shared_device_noise_across_arms=True,
                         scope='target-informed; NOT free generation')

    def test_diagnostic_scope_decomposition_and_complete_rows(self):
        cfg, diag = self.diagnostic()
        verify_denoising(diag, cfg, 3)
        for mode in ['drop', 'duplicate', 'scope', 'decomposition', 'aggregate']:
            cfg, diag = self.diagnostic()
            if mode == 'drop': diag['rows'].pop()
            if mode == 'duplicate': diag['rows'].append(copy.deepcopy(diag['rows'][0]))
            if mode == 'scope': diag['scope'] = 'free generation'
            if mode == 'decomposition': diag['rows'][0]['active40_mse'] = 0.
            if mode == 'aggregate': diag['summary']['train']['0']['correct']['xy16_mse'] = 0.
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                verify_denoising(diag, cfg, 3)

    def test_supplied_and_requested_control_cer_are_distinct(self):
        rows = [dict(split=s, policy=p, seed=73142, guidance=1., characters=4,
                     errors=4, conditioning_text='' if p == 'null' else 'abc',
                     errors_against_supplied_text=None if p == 'null' else 1)
                for s in ['train', 'dev_unseen_text'] for p in ['correct', 'swapped', 'null']]
        out = controls(rows)
        self.assertEqual(out[1]['requested_text_cer'], 1.)
        self.assertEqual(out[1]['supplied_text_cer'], 1/3)
        self.assertIsNone(out[2]['supplied_text_cer'])
        with self.assertRaisesRegex(ValueError, 'controls'):
            controls(rows[:-1])

    def test_packed_row_identity_fails_closed(self):
        row = dict(split='train', seed=1, guidance=1., policy='correct',
                   sample_id='a', generated_points=2)
        with tempfile.TemporaryDirectory() as tmp:
            with h5py.File(Path(tmp)/'eval.h5', 'w') as hf:
                g = hf.create_group('train/1/1.0/correct/a')
                g.create_dataset('points', data=np.zeros((2, 5)))
                g.attrs['row'] = json.dumps(row)
                self.assertEqual(checked_points(hf, row).shape, (2, 5))
                g.attrs['row'] = '{}'
                with self.assertRaisesRegex(ValueError, 'identity'):
                    checked_points(hf, row)


if __name__ == '__main__':
    unittest.main()
