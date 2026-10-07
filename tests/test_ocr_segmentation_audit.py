from pathlib import Path
import sys,unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_segmentation_audit import map_spikes_to_retained,frame_collisions,mixed_frame_fraction,ink_range_labels


class OCRSegmentationAuditTests(unittest.TestCase):
    def test_nearest_retained_mapping_and_collision_counts(self):
        mapped,delta=map_spikes_to_retained([0,3,9,20],[0,2,4,8,10,20])
        self.assertEqual(mapped.tolist(),[0,1,3,5]);self.assertEqual(delta.tolist(),[0,1,1,0])
        c=frame_collisions(mapped,2);self.assertTrue(c['has_collision']);self.assertEqual(c['excess'],1);self.assertEqual(c['max_chars_per_frame'],2)
        self.assertFalse(frame_collisions(mapped,1)['has_collision'])
    def test_mixed_frames_ignore_unlabelled_space_points(self):
        labels=np.array([0,0,-1,1,1,1,2,-1])
        a=mixed_frame_fraction(labels,4);b=mixed_frame_fraction(labels,2)
        self.assertEqual((a['mixed_frames'],a['frames'],a['max_chars_per_frame']),(2,2,2))
        self.assertEqual((b['mixed_frames'],b['frames'],b['max_chars_per_frame']),(0,4,1))
    def test_ink_ranges_map_across_strokes_and_keep_spaces_unlabelled(self):
        strokes=[np.zeros((4,3)),np.zeros((5,3))]
        annotation={'gt_segmentation':{'segments':[
            {'substring':'a','inkRanges':[{'startStroke':0,'startPoint':1,'endStroke':1,'endPoint':1}]},
            {'substring':' '},
            {'substring':'b','inkRanges':[{'startStroke':1,'startPoint':2,'endStroke':1,'endPoint':4}]}]}}
        text,labels=ink_range_labels(annotation,strokes)
        self.assertEqual(text,'a b');self.assertEqual(labels.tolist(),[-1,0,0,0,0,0,2,2,2])


if __name__=='__main__':unittest.main()
