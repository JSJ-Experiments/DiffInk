import unittest
import torch
from iam_tools.generation_cache import CachedLatentPool
from iam_tools.generation_study import collate
class GenerationCacheTests(unittest.TestCase):
 def test_ragged_exact_equivalence(self):
  z={'a':torch.randn(3,384),'b':torch.randn(5,384),'c':torch.randn(2,384)}
  r={'a':{'text':'aba','points':23},'b':{'text':'baaba','points':39},'c':{'text':'b','points':14}}
  v=['a','b'];s={'mean':torch.randn(384),'std':torch.rand(384)+.1}
  p=CachedLatentPool(z,r,v,list(z),s,'cpu')
  for ids in [['c','a'],['b'],['b','a','c'],['a','a']]:
   cached=p.select(ids);original=collate(z,r,v,ids,s,'cpu')
   for a,b in zip(cached,original):self.assertTrue(torch.equal(a,b))
   self.assertEqual(cached[3].sum().item(),sum(r[i]['points'] for i in ids))
 def test_bound(self):
  with self.assertRaises(ValueError):CachedLatentPool({}, {}, [], [], {},'cpu')
if __name__=='__main__':unittest.main()
