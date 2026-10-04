from pathlib import Path
import tempfile
import unittest
import json
import h5py
from iam_tools.build import build
from iam_tools.check_batch import check

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT

class DatasetBuildTests(unittest.TestCase):
    def test_export_and_real_loader_writer_disjoint(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw=root/'raw';raw.mkdir()
            (raw/'ascii').mkdir();(raw/'lineStrokes').mkdir()
            (raw/'writers.xml').write_text('<opt>'+''.join(f'<Writer name="{i:05}"/>' for i in range(4))+'</opt>')
            (raw/'forms.txt').write_text('\n'.join(f'a01-00{i}u {i:05} 0' for i in range(4)))
            for i in range(4):
                form=f'a01-00{i}u'
                (raw/'ascii'/f'{form}.txt').write_text('OCR:\nnot the target\nCSR:\na test .\na test .\n')
                points=''.join(f'<Point x="{j}" y="{j%2*10}" time="{j}"/>' for j in range(240))
                for n in (1,2):
                    (raw/'lineStrokes'/f'{form}-{n:02}.xml').write_text('<Session><Stroke>'+points+'</Stroke></Session>')
            result=build(raw,root/'canonical',root/'out',train_size=2,val_size=1,holdout_writers=1,val_writers=1)
            self.assertEqual(len(result['sample_ids']['train']),2)
            self.assertFalse(result['paper_equivalent'])
            self.assertFalse(set(result['sample_writers']['train'])&set(result['sample_writers']['val']))
            with h5py.File(root/'out/tiny_train.h5') as hf:
                g=hf[next(iter(hf))]
                self.assertEqual(g['char_points_idx'].shape,(0,))
                self.assertEqual(g['line_text'][()].decode(),'a test .')
            report=check(root/'out',REPO,batch_size=2)
            self.assertFalse(report['training']);self.assertTrue(report['writer_disjoint'])

if __name__=='__main__':unittest.main()
