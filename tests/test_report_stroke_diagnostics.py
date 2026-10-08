import hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h5py,numpy as np
from iam_tools.report_stroke_diagnostics import generate
from iam_tools.report_corpus_dit import aggregate
from iam_tools.pen_ab import file_sha
class StrokeReportTests(unittest.TestCase):
    def fixture(self,root):
        p=root/'study';p.mkdir();(p/'baseline').mkdir();(root/'pool').mkdir();q=np.zeros((8,5));q[:,:2]=np.arange(8)[:,None]*.1;q[:-1,2]=1;q[-1,4]=1;src=q.copy();src[[1,3,5],2:]=[0,1,0]
        data={'records':{sid:{'text':'ab','points_sha256':hashlib.sha256(src.tobytes()).hexdigest()} for sid in ['a','b']},'duration':{}}
        (p/'dataset.json').write_text(json.dumps(data));cfg=dict(eval_noise_seeds=[11],eval_guidance=[1.],eval_train_ids=['a'],eval_dev_ids=['b'],max_updates=12000,pool_relative='pool',dataset_sha256=file_sha(p/'dataset.json'));(p/'config.json').write_text(json.dumps(cfg));splits=dict(train=['a'],dev_unseen_text=['b']);rows=[]
        with h5py.File(root/'pool/lines.h5','w') as h:
            for sid in ['a','b']:h.create_group(sid).create_dataset('point_seq',data=src)
        with h5py.File(p/'baseline/evaluation-1000.h5','w') as h:
            for split,ids in splits.items():
                sid=ids[0];r=dict(split=split,sample_id=sid,seed=11,guidance=1.,policy='correct',text='ab',decoded='ab',characters=2,errors=0,conditioning_text='ab',errors_against_supplied_text=0,no_source_trajectory=True,no_source_length=True,no_writer_id=True,estimated_blocks=3,generated_points=8,found_eoc=True);rows.append(r);g=h.create_group(f'{split}/11/1.0/correct/{sid}');g.create_dataset('points',data=q);g.attrs['row']=json.dumps(r)
        ev=dict(rows=rows,aggregate=aggregate(rows,splits,[1.]),packed_h5_sha256=file_sha(p/'baseline/evaluation-1000.h5'));(p/'baseline/eval-1000.json').write_text(json.dumps(ev));return p
    def test_all_rows_audited_and_fixed_scale_gallery(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.fixture(root)
            with patch('iam_tools.report_corpus_dit.duration',return_value=3):out=Path(generate('study',1000,['baseline'],root))
            r=json.loads((out/'summary.json').read_text());self.assertEqual(len(r['rows']),2);self.assertTrue(all(v['corpus_severe_underlifting'] for v in r['groups'].values()));self.assertIn('<svg ',(out/'index.html').read_text());self.assertNotIn('<circle',(out/'index.html').read_text())
    def test_source_and_generated_drift_fail_closed(self):
        for mode in ['source','generated']:
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);p=self.fixture(root);path=root/'pool/lines.h5' if mode=='source' else p/'baseline/evaluation-1000.h5'
                with h5py.File(path,'r+') as h:
                    key='a/point_seq' if mode=='source' else 'train/11/1.0/correct/a/points';h[key][0,0]=10
                with patch('iam_tools.report_corpus_dit.duration',return_value=3),self.assertRaises(ValueError):generate('study',1000,['baseline'],root)
    def test_unknown_arm_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root)
            with self.assertRaises(ValueError):generate('study',1000,['wrong'],root)
if __name__=='__main__':unittest.main()
