import unittest,torch
from iam_tools.generation_coverage import WriterTextDenoiser,select_coverage
from iam_tools.latent_diffusion import TextLatentDenoiser
class CoverageTests(unittest.TestCase):
 def fixture(self):
  records={};small=[];held=[];large=[]
  for j in range(32):
   i=f's{j}';small.append(i);large.append(i);records[i]=dict(writer_id='1',prompt_family=f'f{j}',text=f'train{j}')
  for j in range(8):
   i=f'h{j}';held.append(i);large.append(i);records[i]=dict(writer_id='1',prompt_family='held',text=f'held{j}')
  for w in ['1','2','3','reserved']:
   for j in range(40):
    i=f'{w}a{j}';large.append(i);records[i]=dict(writer_id=w,prompt_family='other',text=f'new{w}{j}')
  records['2a0']['text']='  HELD0  ';records['2a1']['prompt_family']='held'
  return dict(records=records,splits={'large_train':large},test_writers=['reserved'],dev_writers=[]),dict(train=small,unseen_prompt=held)
 def test_nested_and_global_leakage_guards(self):
  m,o=self.fixture();r=select_coverage(m,o,100,3);b=r['arms']['broad1024']
  self.assertTrue(set(o['train'])<=set(r['arms']['writer_all'])<=set(b));self.assertFalse(set(o['unseen_prompt'])&set(b));self.assertNotIn('2a0',b);self.assertNotIn('2a1',b)
  self.assertFalse(any(m['records'][i]['writer_id']=='reserved' for i in b));self.assertEqual(r,select_coverage(m,o,100,3))
 def test_insufficient_data_fails(self):
  m,o=self.fixture()
  with self.assertRaises(ValueError):select_coverage(m,o,1024,3)
 def test_unsafe_original_fails(self):
  m,o=self.fixture();m['records']['s0']['prompt_family']='held'
  with self.assertRaises(ValueError):select_coverage(m,o,100,3)
 def test_zero_writer_is_exact_old_mapping(self):
  torch.manual_seed(1);base=TextLatentDenoiser(channels=8,vocab_size=4,width=16,depth=1,heads=2).eval()
  torch.nn.init.normal_(base.final[-1].weight);model=WriterTextDenoiser(3,**base.config).eval();missing=model.load_state_dict(base.state_dict(),strict=False);self.assertEqual(missing.missing_keys,['writer.weight'])
  x=torch.randn(2,4,8);t=torch.ones(2);text=torch.tensor([[1,2],[3,-1]]);mask=torch.tensor([[True]*4,[True,True,False,False]])
  with torch.no_grad():self.assertTrue(torch.equal(base(x,t,text,mask),model(x,t,text,mask,writer_ids=torch.tensor([0,2]))))
 def test_writer_conditioning_learns_and_masks(self):
  m=WriterTextDenoiser(2,channels=8,vocab_size=4,width=16,depth=1,heads=2);torch.nn.init.normal_(m.final[-1].weight)
  x=torch.zeros(2,4,8);mask=torch.tensor([[True]*4,[True,True,False,False]]);out=m(x,torch.ones(2),torch.ones(2,2,dtype=torch.long),mask,writer_ids=torch.tensor([0,1]));out.sum().backward()
  self.assertGreater(float(m.writer.weight.grad.abs().sum()),0);self.assertTrue(torch.equal(out[1,2:],torch.zeros_like(out[1,2:])))
 def test_explicit_writer_required(self):
  m=WriterTextDenoiser(2,channels=8,vocab_size=4,width=16,depth=1,heads=2)
  with self.assertRaises(ValueError):m(torch.zeros(1,2,8),torch.ones(1),torch.ones(1,2,dtype=torch.long),torch.ones(1,2,dtype=torch.bool))
 def test_budget_and_train_selection(self):
  from iam_tools.generation_coverage_study import validate_budget,score
  for v in [0,999,12001,True,1000.]:
   with self.assertRaises(ValueError):validate_budget(v)
  validate_budget(8000)
  def row(i,value):return dict(sample_id=i,policy='correct',geometry=dict(x_rmse=value,y_rmse=value,first_difference=dict(vector_rmse=value)),pen_aligned_reference=dict(pen_up_f1=1.))
  self.assertEqual(score(dict(lines=[row('train',1.),row('held',999.)]),{'train'}),2.25)
  with self.assertRaises(ValueError):score(dict(lines=[row('held',999.)]),{'train'})
if __name__=='__main__':unittest.main()
