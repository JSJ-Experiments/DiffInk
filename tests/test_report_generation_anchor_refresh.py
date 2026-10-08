import json,tempfile,unittest
from pathlib import Path
import torch
from iam_tools.generation_anchor_refresh_study import ARMS
from iam_tools.report_generation_anchor_refresh import verify
from iam_tools.pen_ab import file_sha
class RefreshReportTests(unittest.TestCase):
 def fixture(self,p):
  trains={a:['a','b'] for a in ARMS};records={i:dict(text=i,prompt_family=i) for i in ['a','b','h']};results={}
  for arm,ids in trains.items():
   q=p/arm;q.mkdir();(q/'metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=ids))+'\n')
   for n in ['checkpoint-last.pt','checkpoint-best.pt']:(q/n).write_bytes(b'fixed')
   torch.save(dict(step=16000,model_state_dict=dict(w=torch.zeros(2)),optimizer_state_dict=dict(state={0:dict(step=torch.tensor(16000.),exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2))},param_groups=[dict(lr=1e-4,betas=(.9,.99),weight_decay=.01,eps=1e-8)])),q/'checkpoint-initial.pt')
   row=lambda i:dict(sample_id=i,policy='correct',geometry=dict(x_rmse=1.,y_rmse=1.,first_difference=dict(vector_rmse=1.)),pen_aligned_reference=dict(pen_up_f1=1.))
   for step in [0,1]:(q/f'eval-{step}.json').write_text(json.dumps(dict(lines=[row(i) for i in ids]+[row('h')])))
   results[arm]=dict(last_step=1,stop='budget_completed',codec_reader_unchanged=True,last_sha256=file_sha(q/'checkpoint-last.pt'),selected_sha256=file_sha(q/'checkpoint-best.pt'),best_step=0,history=[dict(step=s,train_score=2.25) for s in [0,1]])
  return dict(max_updates=1),results,dict(records=records,training_ids=trains,splits=dict(unseen_prompt=['h']))
 def test_restored_model_and_adam_and_exposure(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);v=verify(p,c,r,d);self.assertTrue(v['identical_model_and_adam']);self.assertEqual(v['exposure']['control']['total'],2)
 def test_adam_mutation_is_caught(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);f=p/'refresh_xy/checkpoint-initial.pt';s=torch.load(f,weights_only=False);s['optimizer_state_dict']['state'][0]['exp_avg']+=1;torch.save(s,f)
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_reset_optimizer_or_wrong_step_caught(self):
  for change in [lambda s:s.__setitem__('step',0),lambda s:s['optimizer_state_dict'].__setitem__('state',{})]:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t);c,r,d=self.fixture(p);f=p/'refresh_xy/checkpoint-initial.pt';s=torch.load(f,weights_only=False);change(s);torch.save(s,f)
    with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_identical_schedule_and_actual_train_selection(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'refresh_xy_segment/metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=['b','a']))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);r['refresh_xy']['history'][1]['train_score']=0.
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_no_held_or_checkpoint_drift(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);d['records']['a']['prompt_family']='h'
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'control/checkpoint-last.pt').write_bytes(b'drift')
   with self.assertRaises(ValueError):verify(p,c,r,d)
if __name__=='__main__':unittest.main()
