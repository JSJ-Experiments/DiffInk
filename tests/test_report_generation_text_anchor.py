import unittest,tempfile,json
from pathlib import Path
from iam_tools.report_generation_text_anchor import verify
from iam_tools.generation_text_anchor_study import PARENT_SHA,validate_budget
from iam_tools.pen_ab import file_sha
class AnchorReportTests(unittest.TestCase):
 def fixture(self,p):
  results={}
  for a in ['control','local_hint']:
   q=p/a;q.mkdir();(q/'metrics.jsonl').write_text(json.dumps(dict(sample_ids=['a','b']))+'\n')
   for n in ['checkpoint-last.pt','checkpoint-best.pt']:(q/n).write_bytes(b'fixed')
   results[a]=dict(last_step=1,codec_reader_unchanged=True,last_sha256=file_sha(q/'checkpoint-last.pt'),selected_sha256=file_sha(q/'checkpoint-best.pt'))
  data=dict(records={'a':dict(prompt_family='train',text='a'),'b':dict(prompt_family='train',text='b'),'h':dict(prompt_family='held',text='held')},training_ids={'control':['a','b'],'local_hint':['a','b']},splits={'unseen_prompt':['h']})
  return dict(parent_sha256=PARENT_SHA,max_updates=1),results,data
 def test_equal_scope_and_minibatches_pass(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);self.assertTrue(all(verify(p,c,r,d).values()))
 def test_same_scope_different_batch_guard(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'local_hint/metrics.jsonl').write_text(json.dumps(dict(sample_ids=['b','a']))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_bounded_updates(self):
  validate_budget(4000)
  for n in [True,0,999,6001,1000.]:
   with self.assertRaises(ValueError):validate_budget(n)
if __name__=='__main__':unittest.main()
