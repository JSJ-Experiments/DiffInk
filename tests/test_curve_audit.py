import unittest
import numpy as np
from iam_tools.curve_audit import curve_metrics, split_xy


class CurveAuditTests(unittest.TestCase):
    def test_exact_and_boundary_exclusion(self):
        target = np.array([[0., 0.], [1., 0.], [100., 0.], [101., 0.]])
        states = np.array([0, 1, 0, 2])
        exact = curve_metrics(target, target, states)
        self.assertEqual(exact['point_distance_max'], 0)
        prediction = target + np.array([[0., 0.], [0., 0.], [50., 0.], [50., 0.]])
        m = curve_metrics(prediction, target, states)
        self.assertEqual(m['first_difference']['windows'], 2)
        self.assertEqual(m['first_difference']['vector_rmse'], 0)
        self.assertEqual(m['second_difference']['windows'], 0)
        self.assertIsNone(m['second_difference']['relative_rms_error'])
        strokes = list(split_xy(target, states))
        self.assertEqual([len(s) for s in strokes], [2, 2])

    def test_small_alternating_position_error_exposes_direction_error(self):
        target = np.column_stack([np.arange(9), np.zeros(9)])
        prediction = target.copy(); prediction[:, 1] = (-1.)**np.arange(9)*.1
        m = curve_metrics(prediction, target, np.array([0]*8+[2]))
        self.assertAlmostEqual(m['y_rmse'], .1)
        self.assertAlmostEqual(m['first_difference']['vector_rmse'], .2)
        self.assertAlmostEqual(m['second_difference']['vector_rmse'], .4)
        self.assertIsNone(m['second_difference']['relative_rms_error'])

    def test_single_point_and_invalid_input(self):
        m = curve_metrics([[0, 0]], [[0, 0]], [2])
        self.assertEqual(m['first_difference']['windows'], 0)
        with self.assertRaises(ValueError): curve_metrics([[float('nan'), 0]], [[0, 0]], [2])
        with self.assertRaises(ValueError): curve_metrics([[0, 0]], [[0, 0]], [3])
