from pathlib import Path
import sys,unittest,tempfile
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_pool import prompt_family,normalized_text,blocked,round_robin,assert_splits,build
from iam_tools.ocr_pool_study import single_batch,ctc_per_line,local_metrics,evaluate
from iam_tools.ocr_context_features import make_head


class OCRPoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_prompt_family_removes_writer_version_not_document_identity(self):
        for sid in ['a01-000u-01','a01-000w-02','a01-000z-04','a01-000-03']:
            self.assertEqual(prompt_family(sid),'a01-000')
        with self.assertRaises(ValueError):prompt_family('unknown')
    def test_blocked_excludes_reserved_writer_prompt_and_normalized_text(self):
        entries=[dict(id='a01-000z-01',writer_id='1',text='clear'),dict(id='b01-001-01',writer_id='2',text='clear'),
                 dict(id='c01-002-01',writer_id='1',text=' Hello   WORLD '),dict(id='d01-003-01',writer_id='1',text='safe')]
        self.assertEqual([e['id'] for e in blocked(entries,{'2'},{'a01-000'},{normalized_text('hello world')})],['d01-003-01'])
    def test_round_robin_nested_small_balances_writer_population(self):
        items=[dict(id=f'{w}-{i}',writer_id=w) for w in ['1','2','3'] for i in range(4)]
        a=list(round_robin(items));self.assertEqual(a,list(round_robin(items[::-1])))
        self.assertEqual({e['writer_id'] for e in a[:3]},{'1','2','3'});self.assertEqual(len({e['id'] for e in a}),12)
    def test_split_guards_nesting_prompt_alias_and_population(self):
        records={k:dict(writer_id=w,prompt_family=p,text=t) for k,w,p,t in [('t','1','a','train'),('u','1','d','more'),('d','2','b','dev'),('h','1','c','held')]}
        splits=dict(small_train=['t'],large_train=['t','u'],dev=['d'],held_out=['h'])
        self.assertTrue(assert_splits(records,splits,['3'],['2'])['train_dev_writers_disjoint'])
        records['u']['prompt_family']='c'
        with self.assertRaisesRegex(ValueError,'prompt'):assert_splits(records,splits,['3'],['2'])
        records['u'].update(prompt_family='d',writer_id='4')
        with self.assertRaisesRegex(ValueError,'population'):assert_splits(records,splits,['3'],['2'])
    def test_dev_report_text_and_reserved_writer_guards(self):
        records={k:dict(writer_id=w,prompt_family=p,text=t) for k,w,p,t in [('t','1','a','train'),('d','2','b','shared'),('h','1','c',' SHARED ')]}
        splits=dict(small_train=['t'],large_train=['t'],dev=['d'],held_out=['h'])
        with self.assertRaisesRegex(ValueError,'transcript'):assert_splits(records,splits,['3'],['2'])
        records['h'].update(text='held',writer_id='3')
        with self.assertRaisesRegex(ValueError,'reservations'):assert_splits(records,splits,['3'],['2'])

    def test_single_batch_matches_released_collation(self):
        from dataset.vae_dataset import TrainDataset
        previous=(TrainDataset.text_cache,TrainDataset.writer_cache)
        try:
            TrainDataset.text_cache={'a':0,'b':1};TrainDataset.writer_cache=None
            points=np.array([[.1,.3,1,0,0]]*12+[[.2,.4,0,0,1]],dtype=np.float32)
            reference=TrainDataset.collate_fn([('1',torch.from_numpy(points),'aba',np.array([]))]);raw,mask,labels=single_batch(points,'aba',['a','b'])
            self.assertTrue(torch.equal(raw,reference[0].transpose(1,2)));self.assertTrue(torch.equal(mask,reference[1]));self.assertTrue(torch.equal(labels,reference[2]))
        finally:TrainDataset.text_cache,TrainDataset.writer_cache=previous
    def test_ctc_per_line_matches_actual_head_loss_and_restores_reduction(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        head=make_head(cfg,3).eval();x=torch.zeros(2,48,4);f=x[:,:40].reshape(2,8,5,4);f[:,:,2]=1;f[0,:,2,3]=0;f[0,:,4,3]=1;f[1,:,2,2:]=0;f[1,:,4,2:]=1
        mask=torch.tensor([[True]*4,[True]*3+[False]]);labels=torch.tensor([[0,0,1],[1,0,-1]])
        with torch.no_grad():
            logits=head(x,padding_mask=~mask);loss=ctc_per_line(head,logits,labels,mask).mean();reference=head.get_ocr_loss(x,labels,mask)
        torch.testing.assert_close(loss,reference);self.assertEqual(head.ctc.reduction,'mean')
        with self.assertRaises(ValueError):ctc_per_line(head,logits,torch.zeros(2,5,dtype=torch.long),mask)
    def test_light_geometry_matches_shared_geometric_angle_definition(self):
        from iam_tools.trajectory_geometry import geometry_metrics
        target=np.array([[0,0],[.1,.03],[.12,.11],[.2,.05]],dtype=float);pred=target+np.array([[0,0],[.001,.002],[-.001,.003],[.002,0]])
        states=np.array([0,0,0,2]);a=local_metrics(pred,target,states);b=geometry_metrics(pred,target,states)
        for key in ('tangent_angle_error_degrees','turn_angle_error_degrees'):self.assertEqual(a[key],b[key])
    def test_versioned_pool_reload_survives_current_alias_changes(self):
        import json,hashlib,shutil
        from iam_tools.ocr_pool_study import load_pool
        from iam_tools.pen_ab import file_sha
        records={k:dict(writer_id=w,prompt_family=p,text=t) for k,w,p,t in [('t','1','a','train'),('d','2','b','dev'),('h','1','c','held')]}
        splits=dict(small_train=['t'],large_train=['t'],dev=['d'],held_out=['h'])
        with tempfile.TemporaryDirectory() as d:
            current=Path(d)/'diffink/iam_ocr_pool';current.mkdir(parents=True)
            (current/'lines.h5').write_bytes(b'hashed payload');(current/'chars.json').write_text('{"a":0}')
            m=dict(records=records,splits=splits,test_writers=['3'],dev_writers=['2'],checks=assert_splits(records,splits,['3'],['2']),
                lines_h5_sha256=file_sha(current/'lines.h5'),vocab_sha256=file_sha(current/'chars.json'))
            (current/'manifest.json').write_text(json.dumps(m));sha=file_sha(current/'manifest.json')
            archived=Path(d)/'diffink/iam_ocr_pool_versions'/sha;shutil.copytree(current,archived)
            (current/'manifest.json').write_text('changed alias')
            pool,_,vocab=load_pool(d,sha);self.assertEqual(pool,archived);self.assertEqual(vocab,['a'])
            with self.assertRaises(ValueError):load_pool(d,'../bad')

    def test_partial_posterior_metrics_not_fabricated_as_full_train_draws(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1)
        head=make_head(cfg,3);a=np.array([[float(i+1),float(i%4+1),1,0,0] for i in range(25)],dtype=np.float32);a[-1,2:]=[0,0,1]
        raw,mask,labels=single_batch(a,'ab',['a','b']);raw[:,:2]*=.01
        mu=torch.zeros(1,48,4);mu[:,:40]=raw.transpose(1,2).reshape(1,4,40).transpose(1,2)
        cache={i:dict(mu=mu,lv=torch.full_like(mu,-12),mask=torch.ones(1,4,dtype=torch.bool),labels=labels) for i in 'abcd'}
        splits=dict(train=['a','b'],dev=['c'],held_out=['d'],common_train_probe=['a'])
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as d:
            row=evaluate(head,cache,{i:'ab' for i in cache},splits,['a','b'],d,0,['a','c','d'],draws=2)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(head.training)
        self.assertNotIn('sampled',row['groups']['train']);self.assertEqual(row['groups']['train']['posterior_lines'],1)
        self.assertEqual(row['groups']['common_train_probe']['sampled']['evaluations'],2)
        self.assertEqual([len(r['sampled']) for r in row['lines']],[2,0,2,2])

    def test_cache_decoder_uses_point_resolution_mask_not_latent_mask(self):
        import hashlib,h5py
        from unittest.mock import patch
        from iam_tools.ocr_pool_study import cache_corpus
        class IdentityCodec:
            conv_mu=torch.nn.Identity()
            conv_logvar=staticmethod(lambda x:torch.zeros_like(x))
            @staticmethod
            def to_model_space(raw):
                x=raw.clone();x[:,:2]*=.01;return x
            @staticmethod
            def encoder(x):return x.transpose(1,2).reshape(1,-1,40).transpose(1,2)
            @staticmethod
            def decode(z,padding_mask):
                a=z.transpose(1,2).reshape(1,-1,5).transpose(1,2)
                if padding_mask.shape!=(1,a.shape[-1]):raise AssertionError('decoder requires RAW point mask')
                return torch.cat((a[:,2:],a[:,:2]),dim=1)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);a=np.array([[float(i+1),float(i%4+1),1,0,0] for i in range(13)],dtype=np.float32)
            a[5,2:]=[0,1,0];a[-1,2:]=[0,0,1]
            with h5py.File(p/'lines.h5','w') as hf:hf.create_group('sample').create_dataset('point_seq',data=a)
            m=dict(records={'sample':dict(text='ab',points_sha256=hashlib.sha256(a.tobytes()).hexdigest())})
            with patch('model.losses.mixture_expectation',side_effect=lambda o:o[:,3:].transpose(1,2)):
                c,_,audit=cache_corpus(IdentityCodec(),p,m,['a','b'],p,device='cpu')
            self.assertTrue(audit['passed']);self.assertEqual(int(c['sample']['mask'].sum()),2)

    def test_unsafe_builder_output_never_touches_protected_input(self):
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'raw';raw.mkdir();original=Path(d)/'original';original.mkdir()
            with self.assertRaises(ValueError):build(raw,original,raw/'output')
            with self.assertRaises(ValueError):build(raw,original,original)
            with self.assertRaisesRegex(ValueError,'quotas'):build(raw,original,Path(d)/'out',small_size=10,train_size=2)

if __name__=='__main__':unittest.main()
