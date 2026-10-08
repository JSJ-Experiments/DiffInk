import ast
import copy
import json
from pathlib import Path
import unittest
from iam_tools.history_rollin_study import validate,teacher_gate,PARENT_SHA
from iam_tools.generation_capacity import SOURCE_H5_SHA
from iam_tools.ocr_joint_adapter import READER_SHA

class HistoryRollinStudyTests(unittest.TestCase):
    def fixture(self):
        ids=[str(j) for j in range(8)];data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:{} for s in ids})
        cfg=dict(parent_checkpoint_sha256=PARENT_SHA,source_h5_sha256=SOURCE_H5_SHA,reader_sha256=READER_SHA,max_updates=1000,max_train_wall_seconds=1800,batch=8,lr=1e-5,schedule_seed=60142,rollin_seed=60143,rollin_max_probability=.2,rollin_ramp_steps=500,models={a:dict(point_feedback=False,width=192) for a in ['teacher','rollin']},pen_weight=.024860149190817294,optimizer_restored=True,no_ocr_training_loss=True,not_promoted=True)
        return cfg,data
    def test_matched_parent_objective_scope_and_bounded_schedule(self):
        cfg,d=self.fixture();validate(cfg,d)
        for tag in ['parent','model','writer','aux','optimizer','LR','updates']:
            c=copy.deepcopy(cfg);data=copy.deepcopy(d)
            if tag=='parent':c['parent_checkpoint_sha256']='wrong'
            if tag=='model':c['models']['rollin']['width']=384
            if tag=='writer':data['records']['blind']={}
            if tag=='aux':c['no_ocr_training_loss']=False
            if tag=='optimizer':c['optimizer_restored']=False
            if tag=='LR':c['lr']=3e-4
            if tag=='updates':c['max_updates']=10000
            with self.assertRaises(ValueError):validate(c,data)
    def test_perline_gate_no_aggregate_hiding_and_oracle_reader_distinction(self):
        rows=[dict(sample_id=str(j),geometry=dict(x_rmse=.01,y_rmse=.01,first_difference=dict(vector_rmse=.01),turn_angle_error_degrees=dict(p90=20.)),pen=dict(pen_up_f1=.9,non_final_false_eoc_count=0,final_eoc_correct=True)) for j in range(8)]
        result=dict(teacher_lines=rows);parent=copy.deepcopy(result);reading=dict(rows=[dict(sample_id=str(j)) for j in range(8)],true_pen_cer=0.)
        self.assertTrue(teacher_gate(result,reading,parent)['passed'])
        result['teacher_lines'][7]['geometry']['x_rmse']=.3;self.assertFalse(teacher_gate(result,reading,parent)['passed'])
        result=copy.deepcopy(parent);reading['true_pen_cer']=.01;self.assertFalse(teacher_gate(result,reading,parent)['passed'])
        reading['rows']=reading['rows'][:-1]
        with self.assertRaises(ValueError):teacher_gate(result,reading,parent)
    def test_coordinator_is_lightweight_stable_idempotent_and_nonpreemptible(self):
        path=Path(__file__).resolve().parent.parent/'modal_history_rollin.py';text=path.read_text();functions={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        main=ast.unparse(functions['main']);coord=ast.unparse(functions['coordinate'])
        self.assertEqual(ast.unparse(functions['main'].body[0].test),'not train');self.assertIn('coordinate.spawn(relative)',main);self.assertIn('prepare.remote()',main);self.assertNotIn('prepare.remote()',coord);self.assertIn('calls_once(',coord);self.assertIn('modal.FunctionCall.from_id',coord)
        self.assertIn("['teacher', 'rollin']",coord)
        imports=[n.module for n in ast.walk(functions['coordinate']) if isinstance(n,ast.ImportFrom)]
        self.assertEqual(imports,['iam_tools.launch_ledger'])
        opts={k.arg:ast.literal_eval(k.value) for k in functions['coordinate'].decorator_list[0].keywords if k.arg in ['nonpreemptible','max_containers','memory']}
        self.assertEqual(opts,dict(memory=512,max_containers=1,nonpreemptible=True))
    def test_draw_consumption_and_objective_selection_logged(self):
        import inspect
        from iam_tools import history_rollin_study
        text=inspect.getsource(history_rollin_study.run)
        self.assertIn('draws=torch.rand(',text);self.assertIn("draws_sha256=",text);self.assertIn('opt.load_state_dict(',text)
        self.assertIn("if r['capacity_gate']['passed'] else None",text)

if __name__=='__main__':unittest.main()
