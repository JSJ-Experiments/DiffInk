import json,tempfile,unittest
from pathlib import Path
from iam_tools.generation_path_study import validate_budget,aggregate
from iam_tools.report_generation_path import verify_run
from iam_tools.pen_ab import file_sha

class GenerationStudyTests(unittest.TestCase):
 def test_bounded_updates(self):
  for n in [1000,4000,6000]:validate_budget(n)
  for n in [0,999,6001,True,1000.0]:
   with self.assertRaises(ValueError):validate_budget(n)
 def fixture(self,p):
  result={}
  for arm in ['uniform','terminal','direct']:
   d=p/arm;d.mkdir();(d/'metrics.jsonl').write_text(json.dumps({'sample_ids':['a'],'noise_rng_sha256':'rng'})+'\n')
   (d/'checkpoint-last.pt').write_bytes(b'last');(d/'checkpoint-best.pt').write_bytes(b'best')
   result[arm]={'last_step':1,'codec_reader_unchanged':True,'final_noise_rng_sha256':'final','last_sha256':file_sha(d/'checkpoint-last.pt'),'selected_sha256':file_sha(d/'checkpoint-best.pt')}
  return result
 def test_guard_complete(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);self.assertTrue(all(verify_run(p,{'max_updates':1},r).values()))
 def test_two_arm_report_configuration(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);r={a:r[a] for a in ['uniform','direct']}
   self.assertTrue(all(verify_run(p,{'max_updates':1,'arms':['uniform','direct']},r).values()))
 def test_guard_batch_mismatch(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);(p/'direct/metrics.jsonl').write_text(json.dumps({'sample_ids':['held'],'noise_rng_sha256':'rng'})+'\n')
   with self.assertRaises(ValueError):verify_run(p,{'max_updates':1},r)
 def test_guard_noise_mismatch(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);r['terminal']['final_noise_rng_sha256']='other'
   with self.assertRaises(ValueError):verify_run(p,{'max_updates':1},r)
 def test_guard_checkpoint_drift(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);(p/'direct/checkpoint-best.pt').write_bytes(b'drift')
   with self.assertRaises(ValueError):verify_run(p,{'max_updates':1},r)
 def test_guard_frozen_state(self):
  with tempfile.TemporaryDirectory() as t:
   p=Path(t);r=self.fixture(p);r['uniform']['codec_reader_unchanged']=False
   with self.assertRaises(ValueError):verify_run(p,{'max_updates':1},r)
if __name__=='__main__':unittest.main()
