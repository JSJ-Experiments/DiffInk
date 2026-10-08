import unittest,numpy as np
from iam_tools.stroke_diagnostics import statistics,compare,aggregate

def points(states):
 q=np.zeros((len(states),5));q[:,:2]=np.arange(len(states))[:,None];q[np.arange(len(states)),np.array(states)+2]=1;return q
class StrokeDiagnosticsTests(unittest.TestCase):
    def test_final_eoc_not_extra_stroke_and_singletons_count(self):
        r=statistics(points([0,1,1,0,2]),4)
        self.assertEqual(r['pen_lifts'],2);self.assertEqual(r['rendered_strokes'],3);self.assertEqual(r['singleton_strokes'],1);self.assertEqual(r['connected_segments'],2)
    def test_final_pen_up_no_empty_extra_stroke(self):
        r=statistics(points([0,1]),2);self.assertEqual(r['pen_lifts'],1);self.assertEqual(r['internal_pen_lifts'],0);self.assertEqual(r['rendered_strokes'],1)
        self.assertEqual(statistics(points([0,0]),2)['rendered_strokes'],1)
        self.assertEqual(statistics(points([2]),1)['singleton_strokes'],1)
    def test_does_not_silently_argmax_invalid_soft_pen(self):
        q=points([0,2]);q[0,2:]=[.5,.5,0]
        with self.assertRaises(ValueError):statistics(q,2)
        with self.assertRaises(ValueError):statistics(points([2]),0)
    def test_raw_count_shortfall_without_density_shortfall_is_not_severe(self):
        src=points([0,1]*10+[2]);gen=points([0,1]*3+[2]);r=compare(gen,src,12,32,True)
        self.assertLess(r['pen_lift_count_ratio'],.5);self.assertGreater(r['point_normalized_lift_ratio'],.5);self.assertFalse(r['severe_underlifting']);self.assertEqual(r['stop_stratum'],'EOC_before_80pct_estimated_cap')
    def test_equal_length_major_lift_deficit_flags_failure(self):
        src=points([0,1]*10+[2]);gen=points([0]*20+[2]);r=compare(gen,src,12,24,True)
        self.assertTrue(r['severe_underlifting']);s=aggregate([r]);self.assertTrue(s['corpus_severe_underlifting']);self.assertEqual(s['median_generated_strokes'],1)
    def test_missing_end_and_internal_eoc_disagree_with_saved_stop(self):
        src=points([0,1,0,2]);gen=points([0,0,0,0]);r=compare(gen,src,3,4,False);self.assertEqual(r['stop_stratum'],'estimated_cap_no_EOC')
        with self.assertRaises(ValueError):compare(gen,src,3,8,False)
        with self.assertRaises(ValueError):compare(points([0,2,0,2]),src,3,4,True)
        with self.assertRaises(ValueError):compare(src,src,3,4,False)
    def test_zero_lift_source_does_not_divide_by_zero(self):
        r=compare(points([0,2]),points([0,2]),1,2,True);self.assertIsNone(r['pen_lift_count_ratio']);self.assertFalse(aggregate([r])['corpus_severe_underlifting'])
if __name__=='__main__':unittest.main()
