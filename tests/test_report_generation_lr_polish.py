import json,tempfile,unittest
from pathlib import Path
import torch
from iam_tools.generation_lr_polish_study import ARMS,LEARNING_RATES,PARENT_STEP
from iam_tools.report_generation_lr_polish import verify
from iam_tools.pen_ab import file_sha
class RefreshReportTests(unittest.TestCase):
 def fixture(self,p):
  trains={a:['a','b'] for a in ARMS};records={i:dict(text=i,prompt_family=i) for i in ['a','b','h']};results={}
  for arm,ids in trains.items():
   q=p/arm;q.mkdir();(q/'metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=ids,learning_rates=[LEARNING_RATES[arm]]))+'\n')
   for n in ['checkpoint-last.pt','checkpoint-best.pt']:(q/n).write_bytes(b'fixed')
   torch.save(dict(step=PARENT_STEP,model_state_dict=dict(w=torch.zeros(2)),optimizer_state_dict=dict(state={0:dict(step=torch.tensor(16000.),exp_avg=torch.ones(2),exp_avg_sq=torch.ones(2))},param_groups=[dict(lr=LEARNING_RATES[arm],betas=(.9,.99),weight_decay=.01,eps=1e-8)])),q/'checkpoint-initial.pt')
   row=lambda i:dict(sample_id=i,policy='correct',geometry=dict(x_rmse=1.,y_rmse=1.,first_difference=dict(vector_rmse=1.)),pen_aligned_reference=dict(pen_up_f1=1.))
   for step in [0,1]:(q/f'eval-{step}.json').write_text(json.dumps(dict(lines=[row(i) for i in ids]+[row('h')])))
   results[arm]=dict(last_step=1,stop='budget_completed',codec_reader_unchanged=True,last_sha256=file_sha(q/'checkpoint-last.pt'),selected_sha256=file_sha(q/'checkpoint-best.pt'),best_step=0,history=[dict(step=s,train_score=2.25) for s in [0,1]])
  return dict(max_updates=1,parent_step=PARENT_STEP,learning_rates=LEARNING_RATES,arm_weights={a:dict(xy=.1,first_difference=.1) for a in ARMS}),results,dict(records=records,training_ids=trains,splits=dict(unseen_prompt=['h']))
 def test_restored_model_and_adam_and_exposure(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);v=verify(p,c,r,d);self.assertTrue(v['identical_model_and_adam_moments']);self.assertEqual(v['exposure']['control']['total'],2)
 def test_adam_mutation_is_caught(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);f=p/'midpoint/checkpoint-initial.pt';s=torch.load(f,weights_only=False);s['optimizer_state_dict']['state'][0]['exp_avg']+=1;torch.save(s,f)
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_reset_optimizer_or_wrong_step_caught(self):
  for change in [lambda s:s.__setitem__('step',0),lambda s:s['optimizer_state_dict'].__setitem__('state',{})]:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t);c,r,d=self.fixture(p);f=p/'midpoint/checkpoint-initial.pt';s=torch.load(f,weights_only=False);change(s);torch.save(s,f)
    with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_identical_schedule_and_actual_train_selection(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'polish/metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=['b','a'],learning_rates=[LEARNING_RATES['polish']]))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);r['midpoint']['history'][1]['train_score']=0.
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_no_held_or_checkpoint_drift(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);d['records']['a']['prompt_family']='h'
   with self.assertRaises(ValueError):verify(p,c,r,d)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'control/checkpoint-last.pt').write_bytes(b'drift')
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_wrong_actual_or_logged_lr_caught(self):
  for mutate in ['actual','logged','config','loss']:
   with tempfile.TemporaryDirectory() as t:
    p=Path(t);c,r,d=self.fixture(p)
    if mutate=='actual':
     f=p/'polish/checkpoint-initial.pt';s=torch.load(f,weights_only=False);s['optimizer_state_dict']['param_groups'][0]['lr']=1e-4;torch.save(s,f)
    if mutate=='logged':(p/'polish/metrics.jsonl').write_text(json.dumps(dict(step=1,sample_ids=['a','b'],learning_rates=[1e-4]))+'\n')
    if mutate=='config':c['learning_rates']=dict(LEARNING_RATES,polish=1e-4)
    if mutate=='loss':c['arm_weights']['polish']['first_difference']=999.
    with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_incomplete_train_evaluation_caught(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);f=p/'polish/eval-0.json';e=json.loads(f.read_text());e['lines']=e['lines'][1:];f.write_text(json.dumps(e))
   with self.assertRaises(ValueError):verify(p,c,r,d)
if __name__=='__main__':unittest.main()
