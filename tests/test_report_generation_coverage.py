import unittest,tempfile,json
from pathlib import Path
from iam_tools.report_generation_coverage import verify
from iam_tools.generation_coverage_study import PARENT_SHA
from iam_tools.pen_ab import file_sha
class ReportCoverageTests(unittest.TestCase):
 def fixture(self,p):
  records={'a':dict(prompt_family='a',text='a'),'b':dict(prompt_family='b',text='b'),'c':dict(prompt_family='c',text='c'),'h':dict(prompt_family='held',text='held')};trains={'small32':['a'],'writer_all':['a','b'],'broad1024':['a','b','c']};r={}
  for arm,ids in trains.items():
   q=p/arm;q.mkdir();(q/'metrics.jsonl').write_text(json.dumps(dict(sample_ids=ids))+'\n')
   for n in ['checkpoint-last.pt','checkpoint-best.pt']:(q/n).write_bytes(b'fixed')
   r[arm]=dict(last_step=1,codec_reader_unchanged=True,last_sha256=file_sha(q/'checkpoint-last.pt'),selected_sha256=file_sha(q/'checkpoint-best.pt'))
  return dict(parent_sha256=PARENT_SHA,max_updates=1),r,dict(records=records,training_ids=trains,splits={'unseen_prompt':['h']})
 def test_complete_guards(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);self.assertTrue(all(verify(p,c,r,d).values()))
 def test_held_text_case_whitespace_leakage(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);d['records']['c']['text']=' HELD '
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_out_of_scope_batch(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);(p/'small32/metrics.jsonl').write_text(json.dumps(dict(sample_ids=['h']))+'\n')
   with self.assertRaises(ValueError):verify(p,c,r,d)
 def test_incomplete_budget_or_checkpoint_drift(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);c,r,d=self.fixture(p);r['small32']['last_step']=0
   with self.assertRaises(ValueError):verify(p,c,r,d)
   r['small32']['last_step']=1;(p/'writer_all/checkpoint-best.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):verify(p,c,r,d)
if __name__=='__main__':unittest.main()
