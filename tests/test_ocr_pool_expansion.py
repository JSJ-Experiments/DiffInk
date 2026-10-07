from pathlib import Path
import copy,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_pool import assert_extension,build
from iam_tools.ocr_pool_expansion import restore,state_digest,validate_parent,PARENT_POOL,SHA

class ExpansionTests(unittest.TestCase):
    def test_extension_retains_calibration_prefix_records_and_eval(self):
        parent=dict(vocab_sha256='v',splits=dict(small_train=['b'],large_train=['a','b'],dev=['d'],held_out=['h']),records={i:dict(points_sha256=i) for i in 'abdh'})
        splits=copy.deepcopy(parent['splits']);splits['large_train']+=['c'];records=dict(parent['records'],c=dict(points_sha256='c'))
        self.assertTrue(all(assert_extension(parent,records,splits,'v').values()))
        for key in ('small_train','large_train','dev','held_out'):
            bad=copy.deepcopy(splits);bad[key]=list(reversed(bad[key])) if len(bad[key])>1 else ['c']
            with self.assertRaises(ValueError):assert_extension(parent,records,bad,'v')
        with self.assertRaisesRegex(ValueError,'record'):assert_extension(parent,dict(records,a={}),splits,'v')
        with self.assertRaisesRegex(ValueError,'vocabulary'):assert_extension(parent,records,splits,'other')
    def test_restore_keeps_moments_lr_rng_and_parent_immutable(self):
        with torch.random.fork_rng():
            torch.manual_seed(7);head=torch.nn.Linear(3,2);opt=torch.optim.AdamW(head.parameters(),lr=1e-4)
            head(torch.ones(1,3)).square().sum().backward();opt.step()
            saved=copy.deepcopy(dict(ocr_state_dict=head.state_dict(),optimizer_state_dict=opt.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=[]))
            fingerprint=state_digest(saved)
            a=torch.nn.Linear(3,2);oa=torch.optim.AdamW(a.parameters(),lr=.1)
            proof=restore(a,oa,saved);next_a=torch.rand(8)
            b=torch.nn.Linear(3,2);ob=torch.optim.AdamW(b.parameters(),lr=.2)
            self.assertEqual(restore(b,ob,saved),proof);self.assertTrue(torch.equal(next_a,torch.rand(8)))
            self.assertEqual(oa.param_groups[0]['lr'],1e-4)
            for model,optimizer in ((a,oa),(b,ob)):
                optimizer.zero_grad();model(torch.ones(1,3)).square().sum().backward();optimizer.step()
            self.assertEqual(state_digest(a.state_dict()),state_digest(b.state_dict()))
            self.assertEqual(state_digest(oa.state_dict()),state_digest(ob.state_dict()))
            self.assertEqual(state_digest(saved),fingerprint)
    def test_baseline_float_tolerance_cannot_hide_changed_decodes(self):
        from iam_tools.report_ocr_pool_expansion import baseline_equivalence
        a=dict(lines=[dict(sample_id='s',mu=dict(decoded='abc'),sampled=[dict(decoded='abc')],ctc_loss=1.0)])
        b=copy.deepcopy(a);b['lines'][0]['ctc_loss']+=5e-7
        self.assertTrue(baseline_equivalence(a,b)['all_mean_and_sampled_decode_records_identical'])
        for key in ('mu','sampled'):
            bad=copy.deepcopy(b)
            if key=='mu':bad['lines'][0][key]['decoded']='wrong'
            else:bad['lines'][0][key][0]['decoded']='wrong'
            with self.assertRaisesRegex(AssertionError,'transcript'):baseline_equivalence(a,bad)
        bad=copy.deepcopy(b);bad['lines'][0]['ctc_loss']+=.01
        with self.assertRaisesRegex(AssertionError,'CTC'):baseline_equivalence(a,bad)
        with self.assertRaisesRegex(AssertionError,'missing'):baseline_equivalence(a,dict(lines=[]))

    def test_modal_launcher_does_not_import_unbundled_sibling_entrypoint(self):
        import ast
        tree=ast.parse((ROOT/'modal_ocr_pool_expansion.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module]
        self.assertFalse(any(name.startswith('modal_') for name in imports))

    def test_extension_cannot_replace_parent_pool(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=root/'raw';original=root/'original';parent=root/'parent'
            for p in (raw,original,parent):p.mkdir()
            (parent/'sentinel').write_text('preserve')
            with self.assertRaisesRegex(ValueError,'protected'):
                build(raw,original,out=parent,parent_pool=parent)
            with self.assertRaisesRegex(ValueError,'protected'):
                build(raw,original,out=parent/'child',parent_pool=parent)
            self.assertEqual((parent/'sentinel').read_text(),'preserve')

    def test_state_digest_detects_scalar_and_tensor_changes(self):
        a=dict(moment=torch.tensor([1.,2.]),count=4);b=copy.deepcopy(a)
        self.assertEqual(state_digest(a),state_digest(b));b['count']=5;self.assertNotEqual(state_digest(a),state_digest(b))
        b=copy.deepcopy(a);b['moment'][0]=0;self.assertNotEqual(state_digest(a),state_digest(b))
    def test_parent_requires_pinned_calibration_and_optimizer_lr(self):
        parent=dict(vocab_sha256='v',splits=dict(small_train=['a'],large_train=['a'],dev=['d'],held_out=['h']),records={i:{} for i in 'adh'},train_writers=['1'])
        expanded=copy.deepcopy(parent);expanded['splits']['large_train']+=['b'];expanded['records']['b']={};expanded['parent_manifest_sha256']=PARENT_POOL
        c=dict(pool_manifest_sha256=PARENT_POOL,source_sha256=SHA,feature_mode='relative_scaled',train_ids=['a'],feature_calibration_ids=['a'],dev_ids=['d'],held_out_ids=['h'])
        saved=dict(updates=6000,config=c,optimizer_state_dict=dict(param_groups=[dict(lr=1e-4)]))
        self.assertIs(validate_parent(saved,parent,expanded),c)
        bad=copy.deepcopy(saved);bad['config']['feature_calibration_ids']=['b']
        with self.assertRaisesRegex(ValueError,'calibration'):validate_parent(bad,parent,expanded)
        bad=copy.deepcopy(saved);bad['optimizer_state_dict']['param_groups'][0]['lr']=.001
        with self.assertRaisesRegex(ValueError,'LR'):validate_parent(bad,parent,expanded)

if __name__=='__main__':unittest.main()
