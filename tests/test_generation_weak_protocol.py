import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from iam_tools.generation_weak_alignment_study import validate_protocol,calibrate
from iam_tools.generation_weak_confirmation import fit_gate,reserve
from iam_tools.generation_prefix_contract import PrefixContractWriter
from iam_tools.weak_alignment import PackedAlignmentPool
from iam_tools.generation_cache import CachedLatentPool
from iam_tools.ocr_context_study import tensor_digest


class WeakProtocolTests(unittest.TestCase):
    def protocol(self):
        ids=[f'train-{i}' for i in range(256)]
        spec=dict(causal_queries=True,position_policy='absolute')
        cfg=dict(max_updates=48000,fresh=True,parent_step=0,neural_checkpoint_initialization=False,
            models={a:dict(spec) for a in ['control','weak_alignment']},alignment_fraction=.1,alignment_start_step=1001,calibration_step=1000,calibration_batches=8,
            model_seed=28142,schedule_seeds=[29142,39142],batch=8)
        return cfg,dict(training_ids={a:list(ids) for a in cfg['models']})

    def test_auxiliary_only_bounded_contract(self):
        cfg,data=self.protocol();validate_protocol(cfg,data)
        for field,value in [('alignment_fraction',10),('alignment_start_step',1),('calibration_step',0),('max_updates',50000),('batch',16)]:
            bad=dict(cfg,**{field:value})
            with self.assertRaises(ValueError):validate_protocol(bad,data)
        bad=copy.deepcopy(cfg);bad['models']['weak_alignment']['context_encoder']=True
        with self.assertRaises(ValueError):validate_protocol(bad,data)
        bad=copy.deepcopy(data);bad['training_ids']['control'][0]='held'
        with self.assertRaises(ValueError):validate_protocol(cfg,bad)

    def test_fresh_fit_gate_not_exposed_dev_selection(self):
        cfg,_=self.protocol()
        def result(cer=.04):return dict(stop='budget_completed',last_step=48000,best_step=48000,history=[dict(step=48000,aggregate=dict(all_train256=dict(correct=dict(free_cer=cer))))])
        good={a:result() for a in cfg['models']};fit_gate(cfg,good)
        for bad in [dict(good,weak_alignment=result(.051)),dict(good,weak_alignment=dict(result(),last_step=24000)),dict(good,weak_alignment=result(float('nan')))]:
            with self.assertRaises(ValueError):fit_gate(cfg,bad)

    def fixture(self,zero=False):
        torch.manual_seed(771);m=PrefixContractWriter(causal_queries=True,alignment=True,writer_count=1,vocab_size=2,width=16,depth=2,heads=4)
        if not zero:torch.nn.init.normal_(m.final[-1].weight,std=.1)
        ids=['a','b'];r={s:dict(text='ab',points=24,writer_id='w') for s in ids};latents={s:torch.randn(3,384) for s in ids}
        stats=dict(mean=torch.zeros(384),std=torch.ones(384));pool=CachedLatentPool(latents,r,'ab',ids,stats,device='cpu')
        aligned=PackedAlignmentPool({s:np.array([[0,-1],[0,1],[1,-1]]) for s in ids},ids)
        cfg=dict(writers=['w'],alignment_fraction=.1,calibration_step=1000,auxiliary_weights=dict(xy=.01,first_difference=.01))
        return m,pool,aligned,r,cfg,stats

    def test_calibration_nonvacuous_fixed_fraction_no_state_rng_gradient_mutation(self):
        m,p,a,r,cfg,stats=self.fixture();digest=tensor_digest(m.state_dict());rng=torch.get_rng_state().clone()
        # autograd.grad must not overwrite pre-existing optimizer gradients.
        for v in m.parameters():v.grad=torch.ones_like(v)
        grads=[v.grad.clone() for v in m.parameters()]
        cal=calibrate(m,p,a,r,cfg,[['a','b'],['b','a']],stats)
        self.assertGreater(cal['coefficient'],0);self.assertEqual(digest,tensor_digest(m.state_dict()));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v.grad,g) for v,g in zip(m.parameters(),grads)))
        bn=np.median([x['base_norm'] for x in cal['batches']]);an=np.median([x['auxiliary_norm'] for x in cal['batches']]);self.assertAlmostEqual(cal['coefficient']*an/bn,.1,places=10)

    def test_initial_zero_readout_calibration_is_rejected(self):
        m,p,a,r,cfg,stats=self.fixture(zero=True)
        with self.assertRaises(ValueError):calibrate(m,p,a,r,cfg,[['a','b']],stats)

    def test_reservation_excludes_all_three_sets_without_source_trajectory_io(self):
        from iam_tools.generation_capacity import DATA
        from iam_tools.generation_timing_study import PARENT
        writers=[f'w{i}' for i in range(8)];chars='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .'
        train=[f't{i}' for i in range(256)];history={s:dict(text=chars+str(i),writer_id=writers[i%8],prompt_family='train-form',points=32) for i,s in enumerate(train)}
        records=dict(history);records['dev']=dict(text='old dev',writer_id='w0',prompt_family='dev-form')
        old=[]
        for k in range(3):
            rows={f'old{k}-{j}':dict(text=f'Opened {k} {j}',writer_id=writers[j%8],prompt_family=f'opened-form{k}') for j in range(16)}
            records.update(rows);old.append(rows)
        for i,w in enumerate(writers):
            for j in range(2):records[f'new-{i}-{j}']=dict(text=f'Novel line {i} number {j}',writer_id=w,prompt_family=f'new-form{i}')
        manifest=dict(records=records,splits=dict(large_train=list(records)),dev_writers=[],test_writers=[])
        cfg=dict(vocab=list(chars),writers=writers);data=dict(training_ids={a:train for a in ['control','weak_alignment']},splits=dict(unseen_prompt=['dev']))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def write(rel,value):p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
            write(DATA+'/dataset.json',dict(records=history))
            synthetic_sets=[{f'reused-{j}':dict(text=f'Exposed synthetic set {k} prompt {j}',writer_id='w0') for j in range(16)} for k in range(3)]
            write(PARENT+'/dataset.json',dict(confirmation_seal=dict(ids=list(old[0]),records=old[0]),synthetic_seal=dict(records=synthetic_sets[0])))
            for j,(rel,rows) in enumerate(zip(['checkpoints/iam_generation_position_contract/20261008-101747/reserved-confirmation/seal.json','checkpoints/iam_generation_prefix_contract/20261008-113817/reserved-confirmation/seal.json'],old[1:]),1):
                write(rel,dict(paired=dict(ids=list(rows),records=rows),synthetic=dict(records=synthetic_sets[j])))
            with patch('iam_tools.ocr_pool_study.load_pool',return_value=(root/'DOES-NOT-EXIST',manifest,list(chars))):
                seal=reserve(root,cfg,data,reserved_utc='2026-10-08T00:00:00+00:00');self.assertEqual(seal,reserve(root,cfg,data,reserved_utc=seal['reserved_utc']))
            self.assertEqual(len(seal['paired']['ids']),16);self.assertEqual(set(seal['paired']['ids']),{f'new-{i}-{j}' for i in range(8) for j in range(2)})
            self.assertEqual(seal['history_record_count'],256+48+48);self.assertEqual(len(seal['source_history_sha256']),3);self.assertEqual(len(seal['exposed_paired_ids']),49)
