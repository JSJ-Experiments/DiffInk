import copy,unittest,torch
from iam_tools.generation_capacity import select_capacity,validate_budget,model_config
from iam_tools.generation_coverage import WriterTextDenoiser
class CapacityTests(unittest.TestCase):
 def fixture(self):
  ids=[f't{i}' for i in range(300)];held=['h0','h1'];records={i:dict(text=i,prompt_family=i,writer_id='1') for i in ids+held}
  return dict(records=records,training_ids=dict(broad1024=ids,small32=ids[:32],writer_all=ids[:41]),splits=dict(unseen_prompt=held))
 def test_nested_shared256_and_complete_eval(self):
  d=self.fixture();s=select_capacity(d)
  self.assertEqual(s['training_ids']['small128'],d['training_ids']['broad1024'][:128]);self.assertEqual(s['training_ids']['small256'],s['training_ids']['larger256'])
  self.assertEqual(set(s['splits']['all_train256']),set(s['splits']['all_train128'])|set(s['splits']['expansion256']));self.assertFalse(set(s['splits']['unseen_prompt'])&set(s['splits']['all_train256']))
 def test_global_form_or_text_leakage(self):
  for key,value in [('text',' H0 '),('prompt_family','h1')]:
   d=self.fixture();d['records']['t150'][key]=value
   with self.assertRaises(ValueError):select_capacity(d)
 def test_missing_retained_and_duplicates_rejected(self):
  for mutation in [lambda d:d['training_ids']['broad1024'].reverse(),lambda d:d['training_ids']['broad1024'].__setitem__(150,'t0')]:
   d=self.fixture();mutation(d)
   with self.assertRaises(ValueError):select_capacity(d)
 def test_budget_guard(self):
  for v in [True,1999,24001,2000.]:
   with self.assertRaises(ValueError):validate_budget(v)
  validate_budget(16000)
 def test_sizes_different_model_same_vocabulary(self):
  a=model_config(81,32);b=model_config(81,32,True)
  self.assertEqual(a['vocab_size'],b['vocab_size']);self.assertEqual(a['writer_count'],b['writer_count']);self.assertGreater(b['width'],a['width']);self.assertGreater(b['depth'],a['depth'])
 def test_fresh_small_initialization_policy(self):
  torch.manual_seed(18142);a=WriterTextDenoiser(2,channels=8,vocab_size=4,width=16,depth=1,heads=2)
  torch.manual_seed(18142);b=WriterTextDenoiser(**a.config)
  self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()));self.assertEqual(len(torch.optim.AdamW(a.parameters()).state),0)
if __name__=='__main__':unittest.main()
