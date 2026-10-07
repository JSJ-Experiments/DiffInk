from pathlib import Path
import copy,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_convergence import validate_parent,resumed_schedule,set_lr_only,POOL_SHA,SHA
from iam_tools.ocr_pool_expansion import state_digest
from iam_tools.frozen_ocr_study import bucket_schedule

class ConvergenceTests(unittest.TestCase):
    def test_resume_iterator_matches_suffix_across_partial_epoch(self):
        cache={str(i):dict(mu=torch.zeros(1,4,(i%5)+1)) for i in range(145)};ids=list(cache)
        entire=list(bucket_schedule(cache,ids,37,batch_size=16,seed=43))
        for skip in (0,1,9,13,26):
            state=torch.get_rng_state().clone()
            self.assertEqual(list(resumed_schedule(cache,ids,37-skip,skip=skip)),entire[skip:])
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
        with self.assertRaises(ValueError):resumed_schedule(cache,ids,1,skip=-1)
    def test_lr_override_preserves_all_optimizer_state_except_lr(self):
        head=torch.nn.Linear(2,3);opt=torch.optim.AdamW(head.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=1e-4)
        head(torch.ones(1,2)).square().sum().backward();opt.step();before=copy.deepcopy(opt.state_dict())
        proof=set_lr_only(opt,2e-4);after=copy.deepcopy(opt.state_dict());self.assertTrue(proof['all_moments_counters_other_settings_preserved'])
        self.assertEqual(opt.param_groups[0]['lr'],2e-4)
        for state in (before,after):
            for g in state['param_groups']:g.pop('lr')
        self.assertEqual(state_digest(before),state_digest(after))
        for bad in (0,-.1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):set_lr_only(opt,bad)
    def test_fixed_pool_parent_rejects_changed_ids_features_and_iterator(self):
        splits=dict(small_train=['a','b'],large_train=['a','b'],dev=['d'],held_out=['h'])
        c=dict(pool_manifest_sha256=POOL_SHA,source_sha256=SHA,feature_mode='relative_scaled',attention_radius=None,
            train_ids=['a','b'],dev_ids=['d'],held_out_ids=['h'],feature_calibration_ids=['a','b'],common_train_probe=['a'],posterior_evaluation_ids=['a','d','h'],
            schedule_seed=43,parent_updates=6000,physical_batch=16)
        saved=dict(updates=12000,config=c,optimizer_state_dict=dict(param_groups=[dict(lr=1e-4)]))
        self.assertIs(validate_parent(saved,dict(splits=splits)),c)
        for field,value in [('train_ids',['b','a']),('dev_ids',['a']),('feature_calibration_ids',['b']),('schedule_seed',42),('physical_batch',8),('attention_radius',4)]:
            bad=copy.deepcopy(saved);bad['config'][field]=value
            with self.assertRaises(ValueError):validate_parent(bad,dict(splits=splits))
    def test_same_shapes_different_lrs_preserve_dropout_rng_pairing(self):
        with torch.random.fork_rng():
            torch.manual_seed(8);head=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.1),torch.nn.Linear(4,2))
            saved=copy.deepcopy(head.state_dict());rng=torch.get_rng_state().clone();end=[]
            for lr in (1e-4,2e-4):
                head.load_state_dict(saved);opt=torch.optim.AdamW(head.parameters(),lr=lr);torch.set_rng_state(rng)
                for _ in range(5):
                    opt.zero_grad();head(torch.ones(7,3)).square().mean().backward();opt.step()
                end.append(torch.get_rng_state().clone())
            self.assertTrue(torch.equal(end[0],end[1]))
    def test_modal_launcher_selfcontained(self):
        import ast
        imports=[n.module for n in ast.walk(ast.parse((ROOT/'modal_ocr_convergence.py').read_text())) if isinstance(n,ast.ImportFrom) and n.module]
        self.assertFalse(any(name.startswith('modal_') for name in imports))

if __name__=='__main__':unittest.main()
