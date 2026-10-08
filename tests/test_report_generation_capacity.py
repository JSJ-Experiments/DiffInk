import json,tempfile,unittest
from pathlib import Path
import torch
from iam_tools.generation_capacity import ARMS
from iam_tools.report_generation_capacity import verify
from iam_tools.pen_ab import file_sha
class CapacityReportTests(unittest.TestCase):
 def fixture(self,p):
  trains=dict(small128=['a'],small256=['a','b'],larger256=['a','b'])
  records={i:dict(text=i,prompt_family=i) for i in ['a','b','h']};results={}
  groups=[dict(lr=1e-4,betas=(.9,.99),weight_decay=.01,eps=1e-8)]
  for arm,ids in trains.items():
   q=p/arm;q.mkdir();(q/'metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=ids))+'\n')
   for n in ['checkpoint-last.pt','checkpoint-best.pt']:(q/n).write_bytes(b'fixed')
   torch.save(dict(step=0,model_state_dict=dict(w=torch.zeros(2)),optimizer_state_dict=dict(state={},param_groups=groups)),q/'checkpoint-initial.pt')
   row=lambda i:dict(sample_id=i,policy='correct',geometry=dict(x_rmse=1.,y_rmse=1.,first_difference=dict(vector_rmse=1.)),pen_aligned_reference=dict(pen_up_f1=1.))
   for step in [0,1]:(q/f'eval-{step}.json').write_text(json.dumps(dict(lines=[row(i) for i in ids]+[row('h')])))
   results[arm]=dict(last_step=1,stop='budget_completed',codec_reader_unchanged=True,last_sha256=file_sha(q/'checkpoint-last.pt'),selected_sha256=file_sha(q/'checkpoint-best.pt'),best_step=0,history=[dict(step=s,train_score=2.25) for s in [0,1]])
  return dict(max_updates=1),results,dict(records=records,training_ids=trains,splits=dict(unseen_prompt=['h']))
 def test_all_guards_and_exposure(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);v=verify(p,c,r,d);self.assertTrue(v['full_actual_train_selection']);self.assertEqual(v['exposure']['small256']['total'],2)
 def test_matched256_batches(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'larger256/metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=['b','a']))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_fresh_optimizer(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);f=p/'small128/checkpoint-initial.pt';s=torch.load(f,weights_only=False);s['optimizer_state_dict']['state']={0:{'step':1}};torch.save(s,f)
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_all_actual_train_selection_not_held(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);r['larger256']['history'][1]['train_score']=0.
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_scope_sha_and_leakage(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);d['records']['b']['text']=' H '
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'small256/checkpoint-last.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_no_incomplete_arm_or_skipped_updates(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);r['larger256']['stop']='wall_limit'
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'small128/metrics.jsonl').write_text(json.dumps(dict(step=2,sample_ids=['a']))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
if __name__=='__main__':unittest.main()
