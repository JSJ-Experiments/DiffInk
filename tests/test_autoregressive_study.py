import ast
import copy
from pathlib import Path
import unittest
import torch
from iam_tools.autoregressive_study import checked_path,validate,learning_rate,inputs,calibrate
from iam_tools.autoregressive_strokes import MonotonicStrokeWriter,StrokePool,fit_offset_stats

class AutoregressiveStudyTests(unittest.TestCase):
    def test_path_guard(self):
        self.assertEqual(str(checked_path('/data','checkpoints/iam_autoregressive_study/run')),'/data/checkpoints/iam_autoregressive_study/run')
        for p in ['/tmp/x','checkpoints/iam_autoregressive_study/../bad','checkpoints/other/run']:
            with self.assertRaises(ValueError):checked_path('/data',p)
    def test_learning_rate_bounded_warmup_and_drop(self):
        cfg=dict(max_updates=8000,lr=2e-4,lr_final=5e-5,warmup_steps=200,lr_drop_step=6000)
        self.assertLess(learning_rate(1,cfg),2e-4);self.assertEqual(learning_rate(200,cfg),2e-4);self.assertEqual(learning_rate(5999,cfg),2e-4);self.assertEqual(learning_rate(6000,cfg),5e-5)
        with self.assertRaises(ValueError):learning_rate(8001,cfg)
    def test_only_fixed_vs_adaptive_and_established_scope(self):
        train=[str(j) for j in range(256)];dev=[f'd{j}' for j in range(8)];data=dict(training_ids=train,exposed_dev_ids=dev,fixed_train_ids=train[:8],records={s:{} for s in train+dev})
        cfg=dict(max_updates=8000,batch=16,model_seed=58142,schedule_seed=58143,max_train_wall_seconds=3600,models={'fixed':{'adaptive':False,'width':192},'adaptive':{'adaptive':True,'width':192}},offset_stats={'train_ids':train})
        validate(cfg,data);bad=copy.deepcopy(cfg);bad['models']['adaptive']['width']=384
        with self.assertRaises(ValueError):validate(bad,data)
        bad=copy.deepcopy(data);bad['records']['new-blind']={}
        with self.assertRaises(ValueError):validate(cfg,bad)
    def test_generation_inputs_use_text_not_source_length(self):
        self.assertEqual(inputs(['ab','a'],['a','b'],'cpu').tolist(),[[0,1],[0,-1]])
    def test_calibration_train_only_preserves_state_rng(self):
        torch.manual_seed(4);p=torch.cat((torch.randn(16,2),torch.nn.functional.one_hot(torch.tensor([0]*14+[1,2]),3).float()),-1);records={'s':dict(text='ab',writer_id='w')};stats=fit_offset_stats({'s':p},['s']);pool=StrokePool({'s':p},records,['a','b'],['s'],stats)
        model=MonotonicStrokeWriter(2,1,.5,width=16,text_width=8,writer_width=4);cfg=dict(writers=['w'],calibration_batches=1,pen_gradient_fraction=.25,pen_weights=[1.,3.,8.]);rng=torch.get_rng_state().clone()
        c=calibrate(model,pool,dict(records=records),cfg,[['s']]);self.assertTrue(c['state_rng_unchanged']);self.assertGreater(c['weight'],0);self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_modal_image_contains_eager_import_dependency_and_cpu_import_probe(self):
        path=Path(__file__).resolve().parent.parent/'modal_autoregressive_study.py';s=path.read_text()
        self.assertIn("add_local_dir(str(repo/'utils'),'/app/utils')",s)
        functions={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
        self.assertIn('from model.ocr import ChineseHandwritingOCR',ast.unparse(functions['prepare']))
    def test_durable_parent_explicit_guard_and_t4_limit(self):
        path=Path(__file__).resolve().parent.parent/'modal_autoregressive_study.py';func={n.name:n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)}
        main=func['main'];self.assertEqual(ast.unparse(main.body[0].test),'not train');self.assertTrue(any(isinstance(n,ast.Return) for n in main.body[0].body));self.assertIn('coordinate.spawn(relative)',ast.unparse(main));self.assertNotIn('prepare.remote()',ast.unparse(func['coordinate']));self.assertIn('calls_once(',ast.unparse(func['coordinate']));self.assertIn("['fixed', 'adaptive']",ast.unparse(func['coordinate']));self.assertIn('call.get()',ast.unparse(func['coordinate']))
        options={k.arg:ast.literal_eval(k.value) for k in func['research'].decorator_list[0].keywords if k.arg in ['gpu','cpu','memory','timeout','retries','max_containers']}
        self.assertEqual(options,dict(gpu='T4',cpu=2,memory=8192,timeout=5400,retries=0,max_containers=2))

if __name__=='__main__':unittest.main()
