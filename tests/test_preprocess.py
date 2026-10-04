import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from iam_tools.preprocess import convert, rdp_indices, sequence_strokes
from iam_tools.check_batch import load_module

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
mask=load_module(REPO/'utils/mask.py','mask_test')

class PreprocessTests(unittest.TestCase):
    def sample(self):
        return {'strokes':[[[0,0,0],[1,1,1],[2,2,2]],[[10,1,3]],[[20,2,4],[21,3,5]]]}

    def test_rdp_endpoints_order_and_closed_stroke(self):
        np.testing.assert_array_equal(rdp_indices([[0,0],[1,1],[2,2]],.5),[0,2])
        np.testing.assert_array_equal(rdp_indices([[0,0],[1,1],[0,0]],.5),[0,1,2])
        np.testing.assert_array_equal(rdp_indices([[0,0]],.5),[0])
        with self.assertRaises(ValueError):rdp_indices([[0,0]],-1)

    def test_geometry_orientation_states_times(self):
        q,ends,norm,simp,meta=convert(self.sample())
        self.assertEqual(q.shape[1],5)
        self.assertFalse(meta['paper_equivalent'])
        self.assertGreater(q[0,1],q[1,1])
        self.assertTrue((q[:,0]>=1).all())
        self.assertEqual(len(ends),3)
        self.assertEqual(q[-1,2:].tolist(),[0,0,1])
        for end in ends[:-1]:self.assertEqual(q[end-1,2:].tolist(),[0,1,0])
        decoded=sequence_strokes(q)
        for a,b in zip(decoded,simp):np.testing.assert_allclose(a[:,:2],b[:,:2],rtol=1e-6)
        self.assertEqual(simp[0][:,2].tolist(),[0,2])

    def test_degenerate_fails(self):
        with self.assertRaises(ValueError):convert({'strokes':[[[0,0,0]]]})
        with self.assertRaises(ValueError):convert(self.sample(),height=0)

    def test_prefix_stroke_boundary_both_layouts(self):
        q,*_=convert(self.sample())
        n=len(q);T=8
        seq=torch.tensor(np.vstack([q,np.tile([0,0,0,0,1],(T-n,1))]),dtype=torch.float32)[None]
        valid=torch.arange(T)[None]<n
        a=mask.build_prefix_mask_from_char_points([[]],valid,8,point_seq=seq)
        b=mask.build_prefix_mask_from_char_points([[]],valid,8,point_seq=seq.transpose(1,2))
        for x,y in zip(a,b):torch.testing.assert_close(x,y)
        cutoff=int((a[2][0]==0).sum());self.assertIn(cutoff,[2,3])
        with self.assertRaises(ValueError):mask.build_prefix_mask_from_char_points([[]],valid,8)

    def test_single_stroke_no_full_line_reference(self):
        q,*_=convert({'strokes':[[[0,0,0],[0,10,1]]]})
        seq=torch.tensor(np.vstack([q,np.tile([0,0,0,0,1],(6,1))]),dtype=torch.float32)[None]
        valid=torch.arange(8)[None]<2
        _,_,suffix=mask.build_prefix_mask_from_strokes(seq,valid)
        self.assertTrue(suffix.bool().all())

    def test_legacy_char_mask_unchanged(self):
        valid=torch.ones(1,16,dtype=torch.bool)
        latent,pad,suffix=mask.build_prefix_mask_from_char_points([[4,8,12,16]],valid,8,.3)
        self.assertEqual(suffix[0].tolist(),[0]*4+[1]*12)
        self.assertEqual(latent[0].tolist(),[0,1])
        self.assertEqual(pad[0].tolist(),[1,1])

    def test_original_train_loader_is_unchanged(self):
        import subprocess
        # Available in the pinned checkout; fork CI can skip this provenance check.
        old=subprocess.run(['git','show','97bc6a3c39a5bdaa9728daaab6d3707480006343:dataset/vae_dataset.py'],cwd=REPO,text=True,capture_output=True)
        if old.returncode:self.skipTest('upstream commit not locally available')
        self.assertEqual(old.stdout.split('class ValDataset')[0],(REPO/'dataset/vae_dataset.py').read_text().split('class ValDataset')[0])

class OCRCompatibilityTests(unittest.TestCase):
    def test_english_ctc_uses_exact_not_conservative_length(self):
        ocr=load_module(REPO/'model/ocr.py','ocr_check')
        # Evaluation-only loss check; no optimizer, backward pass, or training.
        model=ocr.ChineseHandwritingOCR(4,8,2,1,4,dropout=0).eval()
        with torch.no_grad():
            loss=model.get_ocr_loss(torch.randn(1,4,3),torch.tensor([[0,1,2]]),torch.ones(1,3,dtype=torch.bool))
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss),0)

if __name__=='__main__':unittest.main()
