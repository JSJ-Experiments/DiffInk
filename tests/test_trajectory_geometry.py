import unittest
import numpy as np
import torch
from test_vae_ctc import REPO
from model.losses import target_difference_loss, target_tangent_loss
from iam_tools.trajectory_geometry import geometry_metrics, wrap_angle, point_to_polyline
from iam_tools.readout_diagnostic import linear_readout


class TargetGeometryTests(unittest.TestCase):
    def test_differences_match_targets_not_generic_smoothing(self):
        # A real target corner has zero loss when exactly reconstructed.
        target = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [2., 1.]]])
        states = torch.tensor([[0, 0, 0, 2]])
        mask = torch.ones(1, 4, dtype=torch.bool)
        for order in (1, 2):
            self.assertEqual(float(target_difference_loss(target, target, states, mask, order)), 0)
            # Translation should not alter segment geometry.
            self.assertEqual(float(target_difference_loss(target+3, target, states, mask, order)), 0)
        rounded = target.clone(); rounded[0, 1] = torch.tensor([.5, .5])
        self.assertGreater(float(target_difference_loss(rounded, target, states, mask)), 0)

    def test_padding_nan_and_penup_jumps_excluded_from_backward(self):
        target = torch.tensor([[[0., 0.], [1., 0.], [100., 0.], [101., 0.], [float('nan'), float('nan')]]])
        prediction = target.clone(); prediction[0, 2:4, 0] += 9; prediction.requires_grad_(True)
        mask = torch.tensor([[True, True, True, True, False]])
        states = torch.tensor([[0, 1, 0, 2, 2]])
        for order in (1, 2):
            prediction.grad = None
            loss = target_difference_loss(prediction, target, states, mask, order)
            self.assertEqual(float(loss.detach()), 0); loss.backward()
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertTrue(torch.equal(prediction.grad[:, 4], torch.zeros_like(prediction.grad[:, 4])))

    def test_exact_angles_and_rotation(self):
        target = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 1.]])
        states = np.array([0, 0, 0, 2])
        m = geometry_metrics(target, target, states)
        self.assertEqual(m['turn_angle_error_degrees']['p99'], 0)
        rotated = np.column_stack([-target[:, 1], target[:, 0]])
        m = geometry_metrics(rotated, target, states)
        self.assertAlmostEqual(m['tangent_angle_error_degrees']['median'], 90)
        self.assertAlmostEqual(m['turn_angle_error_degrees']['median'], 0)
        self.assertAlmostEqual(float(wrap_angle(2*np.pi-.1)), -.1)

    def test_zero_length_edges_and_stroke_breaks_do_not_define_angles(self):
        points = np.array([[0., 0.], [0., 0.], [1., 0.], [100., 0.], [101., 0.]])
        m = geometry_metrics(points, points, [0, 0, 1, 0, 2])
        self.assertEqual(m['tangent_angle_error_degrees']['count'], 2)
        self.assertEqual(m['turn_angle_error_degrees']['count'], 0)

    def test_geometric_distance_to_segments_not_just_vertices(self):
        points = np.array([[.5, .2], [2., 0.]])
        line = np.array([[0., 0.], [1., 0.]])
        np.testing.assert_allclose(point_to_polyline(points, line), [.2, 1.])
        duplicate = np.array([[0., 0.], [0., 0.], [1., 0.]])
        np.testing.assert_allclose(point_to_polyline(points, duplicate), [.2, 1.])

    def test_linear_readout_diagnostic_recovers_affine_mapping(self):
        torch.manual_seed(42)
        features = torch.randn(30, 4, dtype=torch.float64)
        target = features@torch.randn(4, 2, dtype=torch.float64)+torch.tensor([2., -1.])
        prediction, rank = linear_readout(features, target)
        self.assertEqual(rank, 5); torch.testing.assert_close(prediction, target)

    def test_gradient_and_short_sequence(self):
        p = torch.randn(2, 5, 2, requires_grad=True)
        t = torch.randn_like(p); mask = torch.ones(2, 5, dtype=torch.bool)
        s = torch.tensor([[0, 1, 0, 0, 2], [0, 0, 0, 0, 2]])
        loss = target_difference_loss(p, t, s, mask, 2); loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()), 0)
        single = torch.zeros(1, 1, 2, requires_grad=True)
        target_difference_loss(single, single, torch.tensor([[2]]), torch.ones(1, 1, dtype=torch.bool), 2).backward()
        self.assertTrue(torch.equal(single.grad, torch.zeros_like(single)))

    def test_tangent_targets_preserve_corners_ignore_scale_and_handle_collapse(self):
        target = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [1., 1.]]])
        s = torch.tensor([[0, 0, 0, 2]]); mask = torch.ones(1, 4, dtype=torch.bool)
        self.assertEqual(float(target_tangent_loss(target, target, s, mask)), 0)
        self.assertEqual(float(target_tangent_loss(target*2, target, s, mask)), 0)
        p = torch.zeros_like(target, requires_grad=True)
        loss = target_tangent_loss(p, target, s, mask);loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all()); self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_tangent_padding_nan_is_excluded_before_normalization(self):
        t = torch.tensor([[[0., 0.], [1., 0.], [float('nan'), float('nan')]]])
        p = t.clone().requires_grad_(True)
        loss = target_tangent_loss(p, t, torch.tensor([[0, 2, 2]]), torch.tensor([[True, True, False]]))
        self.assertEqual(float(loss.detach()), 0); loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
