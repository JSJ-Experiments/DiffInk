import copy,unittest,ast
from pathlib import Path
from iam_tools.continuous_prefix_study import validate,ARMS,PARENT_SHA,arm_weights
from iam_tools.generated_prefix import auxiliary_coefficients
from iam_tools.generation_capacity import SOURCE_H5_SHA
from iam_tools.ocr_joint_adapter import READER_SHA
class ContinuousPrefixStudyTests(unittest.TestCase):
    def fixture(self):
        ids=list('abcdefgh');data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:{} for s in ids})
        cfg=dict(parent_checkpoint_sha256=PARENT_SHA,source_h5_sha256=SOURCE_H5_SHA,reader_sha256=READER_SHA,max_updates=1000,batch=8,max_train_wall_seconds=1800,lr=1e-5,schedule_seed=64142,training_seed=64143,models={a:dict(point_feedback=False,width=192) for a in ARMS},optimizer_restored=True,pen_weight=.024860149190817294,teacher_anchor_weight=.009549097811244269,no_ocr_training_loss=True,not_promoted=True,auxiliary_calibration=dict(state_rng_unchanged=True,reference_continuous_feedback=True,fractions=dict(xy=.25,pen=.10),weights=auxiliary_coefficients(2.,200.,50.),base_norm=2.,own_xy_norm=200.,own_pen_norm=50.));return cfg,data
    def test_same_scalar_objective_models_optimizer_calibration_reference(self):
        cfg,data=self.fixture();validate(cfg,data);self.assertEqual(arm_weights(ARMS[0],cfg),arm_weights(ARMS[1],cfg))
        for k,v in [('optimizer_restored',False),('max_updates',10000),('teacher_anchor_weight',.1)]:
            bad=copy.deepcopy(cfg);bad[k]=v
            with self.assertRaises(ValueError):validate(bad,data)
        cfg['auxiliary_calibration']['reference_continuous_feedback']=False
        with self.assertRaises(ValueError):validate(cfg,data)
    def test_two_child_lightweight_ledger_coordinator(self):
        tree=ast.parse(Path('modal_continuous_prefix.py').read_text());co=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate');self.assertEqual([a.arg for a in co.args.args],['relative']);self.assertEqual([n.module for n in ast.walk(co) if isinstance(n,ast.ImportFrom)],['iam_tools.launch_ledger'])
        calls=[n for n in ast.walk(co) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calls_once'];self.assertEqual(ast.literal_eval(calls[0].args[1]),list(ARMS))
if __name__=='__main__':unittest.main()
