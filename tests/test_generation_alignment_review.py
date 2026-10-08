import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from iam_tools.generation_alignment_review import verify
from iam_tools.generation_confirmation import confirm
from iam_tools.generation_duration_eval import evaluate_duration
from iam_tools.generation_composition import fit_duration,seal_synthetic
from iam_tools.ocr_context_study import tensor_digest
from iam_tools.pen_ab import file_sha


class ReviewTests(unittest.TestCase):
 def fixture(self,p):
  arms=['global','soft_gaussian'];cfg=dict(fresh=True,parent_step=0,max_updates=4,batch=2,eval_steps=[0,4],betas=[.9,.99],weight_decay=.01,models={a:dict(alignment=a!='global',width=16) for a in arms})
  data=dict(training_ids={a:['a','b'] for a in arms},splits=dict(unseen_prompt=['h']));results={}
  initial=dict(model_state_dict={'w':torch.zeros(2)},step=0,optimizer_state_dict=dict(state={},param_groups=[dict(lr=1e-5,betas=(.9,.99),weight_decay=.01)]),torch_rng_state=torch.tensor([1]),cuda_rng_state=torch.tensor([2]))
  for a in arms:
   folder=p/a;folder.mkdir();torch.save(initial,folder/'checkpoint-initial.pt');torch.save({'step':4},folder/'checkpoint-last.pt');torch.save({'step':4},folder/'checkpoint-best.pt')
   rows=[dict(step=s,sample_ids=['a','b'],learning_rates=[1e-5],loss=.2,gradient_norm=.1) for s in range(1,5)]
   (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));history=[]
   for s in [0,4]:
    (folder/f'evaluation-{s}.h5').write_bytes(b'packed');(folder/f'duration-evaluation-{s}.h5').write_bytes(b'packed_duration')
    ev=dict(lines=[dict(sample_id=i,policy='correct') for i in ['a','b','h']],score=2 if s==0 else 1,packed_h5_sha256=file_sha(folder/f'evaluation-{s}.h5'))
    (folder/f'eval-{s}.json').write_text(json.dumps(ev));(folder/f'duration-eval-{s}.json').write_text(json.dumps(dict(ids=['h'],packed_h5_sha256=file_sha(folder/f'duration-evaluation-{s}.h5'))));history.append(dict(step=s,train_score=ev['score']))
   results[a]=dict(last_step=4,best_step=4,codec_reader_unchanged=True,stop='budget_completed',initial_state_sha256=tensor_digest(initial['model_state_dict']),last_sha256=file_sha(folder/'checkpoint-last.pt'),selected_sha256=file_sha(folder/'checkpoint-best.pt'),history=history,schedule_sha256=hashlib.sha256(json.dumps([r['sample_ids'] for r in rows]).encode()).hexdigest())
  return cfg,data,results
 def check(self,p,c,d,r):
  with patch('iam_tools.generation_alignment_review.learning_rate',return_value=1e-5),patch('iam_tools.generation_alignment_review.score',side_effect=lambda ev,train:ev['score']):return verify(p,c,r,d)
 def test_valid_fresh_matched_protocol(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);c,d,r=self.fixture(p);g=self.check(p,c,d,r);self.assertTrue(g['identical_fresh_weights_empty_adam_rng']);self.assertEqual(g['exposure']['global']['total'],8)
 def test_rejects_scope_schedule_lr_empty_adam_and_selection_drift(self):
  for mutation in ['train','lr','adam','rng','selection','missing_train','sha','budget','bundle']:
   with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);c,d,r=self.fixture(p);a='soft_gaussian';folder=p/a
    if mutation=='train':d['training_ids'][a]=['a','h']
    elif mutation=='lr':rows=[json.loads(s) for s in (folder/'metrics.jsonl').read_text().splitlines()];rows[0]['learning_rates']=[5e-5];(folder/'metrics.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    elif mutation in ['adam','rng']:
     saved=torch.load(folder/'checkpoint-initial.pt',weights_only=False)
     if mutation=='adam':saved['optimizer_state_dict']['state']={0:{'step':torch.tensor(1)}}
     else:saved['cuda_rng_state']=torch.tensor([9])
     torch.save(saved,folder/'checkpoint-initial.pt')
    elif mutation=='selection':r[a]['best_step']=0
    elif mutation=='missing_train':e=json.loads((folder/'eval-4.json').read_text());e['lines']=e['lines'][1:];(folder/'eval-4.json').write_text(json.dumps(e))
    elif mutation=='sha':(folder/'evaluation-4.h5').write_bytes(b'changed')
    elif mutation=='budget':r[a]['stop']='wall_limit'
    else:c['models'][a]['width']=32
    with self.assertRaises(ValueError):self.check(p,c,d,r)
 def test_continuation_distinct_weights_restored_adam_not_fresh(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);c,d,r=self.fixture(p);c.update(fresh=False,parent_step=24000,lr=1e-5)
   for j,a in enumerate(['global','soft_gaussian']):
    q=p/a/'checkpoint-initial.pt';saved=torch.load(q,weights_only=False);saved['model_state_dict']['w']+=j;saved['optimizer_state_dict']['state']={0:{'step':torch.tensor(24000), 'exp_avg':torch.ones(2)}};torch.save(saved,q);r[a]['initial_state_sha256']=tensor_digest(saved['model_state_dict'])
   with patch('iam_tools.generation_alignment_review.score',side_effect=lambda ev,train:ev['score']):g=verify(p,c,r,d,continuation=True)
   self.assertFalse(g['identical_fresh_weights_empty_adam_rng']);self.assertTrue(g['identical_actual_minibatches_lrs'])
   with self.assertRaises(ValueError):self.check(p,c,d,r)
 def test_confirmation_refuses_early_unseal_and_overwrite(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)
   with self.assertRaises(ValueError):confirm(p,dict(max_updates=4),{},{'global':dict(stop='wall_limit',last_step=3)},None,None,None,None)
   (p/'confirmation').mkdir()
   with self.assertRaises(ValueError):confirm(p,{}, {}, {},None,None,None,None)


class DurationEvaluationTests(unittest.TestCase):
 def test_unpaired_synthetic_has_no_fabricated_oracle_metrics(self):
  model=torch.nn.Linear(1,1);model.eval();calls=[]
  def forward(x,time,text,mask,**kw):calls.append((x.clone(),mask.clone(),text.clone(),kw['drop_text'].clone()));return x
  model.forward=forward
  records={'s1':dict(text='ab',writer_id='w'),'s2':dict(text='ba',writer_id='w')}
  duration=fit_duration({'train':dict(text='ab',writer_id='w',points=16)},['train'],['w'])
  def decode(codec,reader,z,record,vocab):
   n=len(z)*8;return torch.zeros(n,5).numpy(),dict(free_decoded='ab',free_errors=int(record['text']!='ab'),characters=2,generated_points_at_stop=n,first_eoc_point=None,state_counts=[n,0,0],oracle_points=record['points'],oracle_window_decoded='bad',window_errors=9,internal_eoc_count=3)
  with tempfile.TemporaryDirectory() as tmp,patch('iam_tools.generation_duration_eval.transform',side_effect=lambda p,s,inverse:p),patch('iam_tools.generation_duration_eval.decode_sample',side_effect=decode):
   e=evaluate_duration(model,None,None,records,['a','b'],{},list(records),tmp,4,['w'],duration,controls=True)
  self.assertEqual(len(calls),3);self.assertTrue(all((x==0).all() for x,*_ in calls));self.assertTrue(torch.equal(calls[0][1],calls[1][1]));self.assertTrue(calls[2][3].all())
  for row in e['lines']:
   self.assertTrue(row['no_paired_target']);self.assertIsNone(row['actual_blocks_for_diagnostic_only']);self.assertIsNone(row['length_relative_error'])
   for key in ['oracle_points','oracle_window_decoded','window_errors','geometry']:self.assertNotIn(key,row)
  self.assertIsNone(e['aggregate']['estimated_correct']['mean_absolute_relative_length_error'])
 def test_synthetic_seal_deterministic_and_unpaired(self):
  chars=''.join(['IWeSheThey leftfoundkeptmoved thebluenotebookasmallletteroldbookacupoftea onthedeskbywindown eardo rin theoffice.'])
  writers=[str(j) for j in range(8)];r={w:dict(text=chars,writer_id=w,points=200) for w in writers};a=seal_synthetic(r,list(r),writers);self.assertEqual(a,seal_synthetic(r,list(r),writers));self.assertEqual(len(a['records']),16);self.assertTrue(all('points' not in v for v in a['records'].values()))
if __name__=='__main__':unittest.main()
