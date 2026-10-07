import unittest
import numpy as np
from iam_tools.ocr_acquisition_audit import backward_jumps

class AcquisitionTests(unittest.TestCase):
    def test_negative_local_curve_is_not_a_pen_jump(self):
        a=np.array([[100,0,1,0,0],[0,0,1,0,0],[100,0,0,0,1]],dtype=float)
        self.assertEqual(backward_jumps(a),0);a[0,2:]=[0,1,0];self.assertEqual(backward_jumps(a),1)
    def test_threshold_is_in_model_units_and_final_eoc_has_no_next_point(self):
        a=np.array([[100,0,0,1,0],[70,0,0,0,1]],dtype=float)
        self.assertEqual(backward_jumps(a),0);self.assertEqual(backward_jumps(a,threshold=.2),1)
        self.assertEqual(backward_jumps(np.array([[0,0,0,0,1]],dtype=float)),0)
    def test_invalid_shape_and_values(self):
        for a in (np.zeros((2,4)),np.array([[float('nan'),0,0,0,1]])):
            with self.assertRaises(ValueError):backward_jumps(a)
        with self.assertRaises(ValueError):backward_jumps(np.zeros((2,5)),threshold=0)

if __name__=='__main__':unittest.main()
