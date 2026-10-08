import itertools
import unittest
import numpy as np
from iam_tools.ctc_alignment import forced_ctc


def collapse(a):
    return [c for j,c in enumerate(a) if c!=0 and (j==0 or c!=a[j-1])]


class ForcedCTCTests(unittest.TestCase):
    def test_matches_exhaustive_best_path(self):
        rng=np.random.default_rng(37)
        for labels in [[1],[1,2],[1,1],[1,2,1]]:
            p=rng.normal(size=(5,3));p-=np.log(np.exp(p).sum(1))[:,None]
            possible=[a for a in itertools.product(range(3),repeat=5) if collapse(a)==labels]
            best=max(sum(p[t,c] for t,c in enumerate(a)) for a in possible)
            q=forced_ctc(p,labels)
            self.assertAlmostEqual(q['log_probability'],best)
            self.assertEqual(collapse(q['emissions']),labels)
            self.assertTrue(all(q['token_frames']))

    def test_repeated_labels_require_blank(self):
        p=np.array([[-10.,0.],[0.,-10.],[-10.,0.]])
        q=forced_ctc(p,[1,1]);self.assertEqual(q['emissions'].tolist(),[1,0,1])
        self.assertEqual(q['token_indices'].tolist(),[0,-1,1])
        with self.assertRaises(ValueError):forced_ctc(p[:2],[1,1])

    def test_cannot_skip_or_substitute_labels(self):
        p=np.full((4,4),-10.);p[:,3]=0
        q=forced_ctc(p,[1,2]);self.assertEqual(collapse(q['emissions']),[1,2])
        self.assertNotIn(3,q['emissions'])

    def test_bad_inputs_rejected(self):
        for p,labels in [(np.ones((2,3)),[]),(np.ones((2,3)),[0]),(np.ones((2,3)),[3]),
                        (np.ones((2,3)),[1.5]),(np.full((2,3),np.nan),[1])]:
            with self.assertRaises(ValueError):forced_ctc(p,labels)
