import copy
import unittest
from pathlib import Path
import ast
from iam_tools.cumulative_xy_study import validate,ARMS,PARENT_SHA
from iam_tools.cumulative_xy import anchor_coefficient
from iam_tools.generation_capacity import SOURCE_H5_SHA
from iam_tools.ocr_joint_adapter import READER_SHA

class CumulativeXYStudyTests(unittest.TestCase):
    def fixture(self):
        ids=list('abcdefgh');data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:{} for s in ids})
        cfg=dict(parent_checkpoint_sha256=PARENT_SHA,source_h5_sha256=SOURCE_H5_SHA,reader_sha256=READER_SHA,max_updates=1000,batch=8,max_train_wall_seconds=1800,lr=1e-5,schedule_seed=62142,rollin_seed=62143,rollin_max_probability=.2,rollin_ramp_steps=500,models={a:dict(point_feedback=False,width=192) for a in ARMS},optimizer_restored=False,pen_weight=.024860149190817294,no_ocr_training_loss=True,not_promoted=True,anchor_calibration=dict(state_rng_unchanged=True,gradient_fraction=.15,weight=anchor_coefficient(2.,20.),offset_norm=2.,anchor_norm=20.))
        return cfg,data
    def test_exact_matched_protocol_and_gradient_fraction(self):
        cfg,data=self.fixture();validate(cfg,data)
        for key,value in [('optimizer_restored',True),('max_updates',8000),('lr',.001),('parent_checkpoint_sha256','bad')]:
            bad=copy.deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):validate(bad,data)
        cfg['anchor_calibration']['weight']*=10
        with self.assertRaises(ValueError):validate(cfg,data)
    def test_no_new_data_or_architecture_confound(self):
        cfg,data=self.fixture();cfg['models']['teacher_anchor']['width']=384
        with self.assertRaises(ValueError):validate(cfg,data)
        cfg,data=self.fixture();data['exposed_dev_ids']=['new']
        with self.assertRaises(ValueError):validate(cfg,data)
    def test_launcher_has_stable_study_argument_ledger_and_lightweight_import(self):
        tree=ast.parse(Path('modal_cumulative_xy.py').read_text());co=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate')
        self.assertEqual([a.arg for a in co.args.args],['relative'])
        imports=[n.module for n in ast.walk(co) if isinstance(n,ast.ImportFrom)]
        self.assertEqual(imports,['iam_tools.launch_ledger'])
        calls=[n for n in ast.walk(co) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calls_once']
        self.assertEqual(len(calls),1);self.assertEqual(ast.literal_eval(calls[0].args[1]),list(ARMS))
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        self.assertIsInstance(main.body[0],ast.If);self.assertTrue(any(isinstance(n,ast.Return) for n in ast.walk(main.body[0])))

if __name__=='__main__':unittest.main()
