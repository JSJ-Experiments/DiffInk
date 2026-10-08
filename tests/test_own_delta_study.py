import ast,copy,unittest
from pathlib import Path
from iam_tools.own_delta_study import validate,ARMS,PARENT,PARENT_SHA,arm_weight
from iam_tools.own_delta import delta_coefficients
from iam_tools.generation_capacity import SOURCE_H5_SHA
from iam_tools.ocr_joint_adapter import READER_SHA
class OwnDeltaStudyTests(unittest.TestCase):
    def fixture(self):
        ids=list('abcdefgh');data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:{} for s in ids})
        cfg=dict(parent_checkpoint_sha256=PARENT_SHA,source_h5_sha256=SOURCE_H5_SHA,reader_sha256=READER_SHA,max_updates=2000,batch=8,max_train_wall_seconds=1800,lr=1e-5,schedule_seed=65142,training_seed=65143,models={a:dict(point_feedback=False,width=192) for a in ARMS},optimizer_restored=True,pen_weight=.024860149190817294,teacher_anchor_weight=.009549097811244269,own_xy_weight=5.006895593114276e-5,own_pen_weight=3.159880562284457e-6,weight_decay=0.,clip=5,no_ocr_training_loss=True,not_promoted=True,delta_calibration=dict(state_rng_unchanged=True,fraction=.25,weights=delta_coefficients(2,10,5),complete_norm=2,ink_norm=10,all_norm=5),eval_steps=[0,250,500,1000,2000],control_steps=[0,2000],free_max_blocks=256,parent_joint_step=1000,parent_checkpoint_relative=PARENT+'/continuous_xy/checkpoint-best.pt',output_volume='diffink-experiments-v2',source_volume_input_only=True,no_kl=True,no_style=True,no_new_confirmation=True)
        return cfg,data
    def test_matched_small_auxiliary_preserves_complete_parent_objective(self):
        cfg,data=self.fixture();validate(cfg,data)
        self.assertEqual([arm_weight(a,cfg) for a in ARMS],[0.,.05,.1])
        for k,v in [('own_xy_weight',0.),('optimizer_restored',False),('max_updates',10000),('pen_weight',1.),('parent_joint_step',2000),('no_kl',False),('control_steps',[0]),('output_volume','diffink-data')]:
            bad=copy.deepcopy(cfg);bad[k]=v
            with self.assertRaises(ValueError):validate(bad,data)
        cfg['delta_calibration']['weights']['ink']*=2
        with self.assertRaises(ValueError):validate(cfg,data)
    def test_no_new_confirmation_or_model_change(self):
        cfg,data=self.fixture();data['exposed_dev_ids']=['new']
        with self.assertRaises(ValueError):validate(cfg,data)
        cfg,data=self.fixture();cfg['models']['all_delta']['width']=384
        with self.assertRaises(ValueError):validate(cfg,data)
        with self.assertRaises(ValueError):arm_weight('smooth',cfg)
    def test_three_child_durable_coordinator_and_no_resume_template(self):
        source=Path('modal_own_delta.py').read_text();tree=ast.parse(source);co=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate')
        self.assertEqual([a.arg for a in co.args.args],['relative'])
        self.assertEqual([n.module for n in ast.walk(co) if isinstance(n,ast.ImportFrom)],['iam_tools.launch_ledger'])
        calls=[n for n in ast.walk(co) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calls_once'];self.assertEqual(ast.literal_eval(calls[0].args[1]),list(ARMS))
        self.assertNotIn('resume_source',source);self.assertIn('.with_mount_options(read_only=True)',source);self.assertIn("timeout=120,check=True",source)
        pre=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight');self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('spawn','remote') for n in ast.walk(pre)))
if __name__=='__main__':unittest.main()
