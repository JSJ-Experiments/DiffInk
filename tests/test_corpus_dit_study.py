import ast,unittest,torch
from pathlib import Path
from iam_tools.corpus_dit_study import PosteriorPool,schedule
class CorpusDiTStudyTests(unittest.TestCase):
    def test_full_epoch_bucket_schedule_does_not_drop_rare_writers_or_tail(self):
        ids=[str(j) for j in range(161)];lengths={i:int(i)%13+1 for i in ids};batches=list(schedule(ids,lengths,100,32,4));first=batches[:6]
        # 128-window=4 batches, remaining33-window=2 batches; all lines exactly once.
        self.assertEqual(sorted(i for b in first for i in b),sorted(ids));self.assertEqual(batches,list(schedule(ids,lengths,100,32,4)));self.assertTrue(all(1<=len(b)<=32 for b in batches))
    def test_gpu_pool_cpu_reference_whitening_sampling_and_minimal_padding(self):
        items={s:dict(mu=torch.ones(n,384)*v,lv=torch.zeros(n,384),prefix_blocks=1) for s,n,v in [('a',3,2.),('b',6,4.)]};records=dict(a=dict(text='ab'),b=dict(text='a'));stats=dict(mean=torch.ones(384),std=torch.ones(384)*2);p=PosteriorPool(items,records,['a','b'],['a','b'],stats,'cpu')
        torch.manual_seed(9);q,text,mask,prefix=p.select(['a']);torch.manual_seed(9);expected=torch.full((1,3,384),.5)+torch.randn(1,3,384)*.5;torch.testing.assert_close(q,expected);self.assertEqual(tuple(q.shape),(1,3,384));self.assertEqual(text.tolist(),[[0,1]]);self.assertEqual(prefix.tolist(),[1]);self.assertTrue(mask.all());self.assertFalse(p.mask[0,3:].any());self.assertEqual(float(p.std[0,3:].abs().sum()),0.)
    def test_one_durable_child_and_explicit_read_only_original_volume(self):
        source=Path('modal_corpus_dit.py').read_text();tree=ast.parse(source);co=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='coordinate');self.assertEqual([a.arg for a in co.args.args],['relative']);self.assertEqual([n.module for n in ast.walk(co) if isinstance(n,ast.ImportFrom)],['iam_tools.launch_ledger'])
        calls=[n for n in ast.walk(co) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='calls_once'];self.assertEqual(ast.literal_eval(calls[0].args[1]),['baseline']);self.assertIn('with_mount_options(read_only=True)',source);self.assertIn("timeout=120,check=True",source)
if __name__=='__main__':unittest.main()
