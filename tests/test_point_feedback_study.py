import ast
import copy
from pathlib import Path
import unittest
import torch
from iam_tools.point_feedback_study import validate,checked_path,learning_rate
from iam_tools.point_feedback_strokes import PointFeedbackStrokeWriter
from iam_tools.autoregressive_strokes import StrokePool,fit_offset_stats
from iam_tools.autoregressive_study import calibrate
from iam_tools.generation_capacity import SOURCE_H5_SHA
from iam_tools.ocr_joint_adapter import READER_SHA

class PointFeedbackStudyTests(unittest.TestCase):
    def test_scope_and_only_feedback_changed(self):
        ids=[str(j) for j in range(8)];data=dict(training_ids=ids,fixed_train_ids=ids,exposed_dev_ids=[],records={s:{} for s in ids})
        cfg=dict(max_updates=3000,max_train_wall_seconds=1800,batch=8,model_seed=59142,schedule_seed=59143,models={'no_feedback':dict(point_feedback=False,width=192),'point_feedback':dict(point_feedback=True,width=192)},offset_stats=dict(train_ids=ids),source_h5_sha256=SOURCE_H5_SHA,reader_sha256=READER_SHA)
        validate(cfg,data)
        for mode in ['width','dev','record','updates','stats']:
            c=copy.deepcopy(cfg);d=copy.deepcopy(data)
            if mode=='width':c['models']['point_feedback']['width']=384
            if mode=='dev':d['exposed_dev_ids']=['blind']
            if mode=='record':d['records']['extra']={}
            if mode=='updates':c['max_updates']=30000
            if mode=='stats':c['offset_stats']['train_ids']=ids[1:]
            with self.assertRaises(ValueError):validate(c,d)
    def test_path_and_lr(self):
        self.assertEqual(str(checked_path('/data','checkpoints/iam_point_feedback/run')),'/data/checkpoints/iam_point_feedback/run')
        for s in ['/tmp/x','checkpoints/iam_point_feedback/../bad','checkpoints/other/run']:
            with self.assertRaises(ValueError):checked_path('/data',s)
        cfg=dict(max_updates=3000,warmup_steps=100,lr=3e-4,lr_final=5e-5,lr_drop_step=2400)
        self.assertEqual(learning_rate(100,cfg),3e-4);self.assertEqual(learning_rate(2400,cfg),5e-5)
        with self.assertRaises(ValueError):learning_rate(3001,cfg)
    def test_shared_surrogate_calibration_is_rng_preserving(self):
        torch.manual_seed(4);p=torch.cat((torch.randn(16,2),torch.nn.functional.one_hot(torch.tensor([0]*14+[1,2]),3).float()),-1);records={'s':dict(text='ab',writer_id='w')};stats=fit_offset_stats({'s':p},['s']);pool=StrokePool({'s':p},records,['a','b'],['s'],stats)
        cfg=dict(writers=['w'],calibration_batches=1,pen_gradient_fraction=.25,pen_weights=[1.,3.,8.]);results=[]
        for policy in [False,True]:
            torch.manual_seed(7);model=PointFeedbackStrokeWriter(2,1,.5,width=16,text_width=8,writer_width=4,point_feedback=policy)
            surrogate=PointFeedbackStrokeWriter(2,1,.5,width=16,text_width=8,writer_width=4);surrogate.load_state_dict(model.state_dict());rng=torch.get_rng_state().clone()
            c=calibrate(surrogate,pool,dict(records=records),cfg,[['s']]);self.assertTrue(torch.equal(rng,torch.get_rng_state()));results.append(c)
        self.assertEqual(results[0],results[1])
    def test_guarded_durable_launcher(self):
        path=Path(__file__).resolve().parent.parent/'modal_point_feedback.py';text=path.read_text();func={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        self.assertIn("add_local_dir(str(repo/'utils'),'/app/utils')",text)
        self.assertEqual(ast.unparse(func['main'].body[0].test),'not train')
        self.assertTrue(any(isinstance(n,ast.Return) for n in func['main'].body[0].body))
        self.assertIn("['no_feedback', 'point_feedback']",ast.unparse(func['coordinate']))
        self.assertIn('call.get()',ast.unparse(func['coordinate']));self.assertIn('coordinate.spawn(relative)',ast.unparse(func['main']))
        self.assertNotIn('prepare.remote()',ast.unparse(func['coordinate']))
        self.assertIn('calls_once(',ast.unparse(func['coordinate']))
        options={k.arg:ast.literal_eval(k.value) for k in func['research'].decorator_list[0].keywords if k.arg in ['gpu','retries','max_containers']}
        self.assertEqual(options,dict(gpu='T4',retries=0,max_containers=2))

if __name__=='__main__':unittest.main()
