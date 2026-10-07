from pathlib import Path
import copy,sys,unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from iam_tools.ocr_augmentation import augment_transport,draw_parameters,mixed_loss,validate_parent,POOL_SHA,SHA
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.ocr_context_features import unpack,make_head
from iam_tools.frozen_ocr_study import collate_latents
from iam_tools.ocr_error_analysis import edit_trace,analyze

class OCRAugmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_identity_pen_and_eoc_synthetic_phases_preserved(self):
        c=frame_cache({'a':line(13)},4)['a'];before=c['mu'].clone()
        x=augment_transport(c['mu'],c['mask'],torch.tensor([[1.,1.,0.,0.]]))
        torch.testing.assert_close(x[:,:20],c['mu'][:,:20],atol=6e-8,rtol=1e-7)
        self.assertTrue(torch.equal(c['mu'],before));self.assertTrue(torch.equal(x[:,20:],torch.zeros_like(x[:,20:])))
        f,r=unpack(x,c['mask'],4);a,_=unpack(c['mu'],c['mask'],4)
        self.assertTrue(torch.equal(f[:,:,2:],a[:,:,2:]));self.assertTrue(torch.equal(f.permute(0,1,3,2)[~r],a.permute(0,1,3,2)[~r]));self.assertEqual(int(r.sum()),13)
    def test_per_line_affine_formula_and_stroke_structure(self):
        cache=frame_cache({'a':line(13),'b':line(19)},4);mu,l,mask=collate_latents(cache,['a','b']);p=torch.tensor([[.9,1.1,.12,.04],[1.1,.9,-.12,-.04]])
        f,r=unpack(mu,mask,4);v=augment_transport(mu,mask,p);g,r2=unpack(v,mask,4)
        for j in range(2):
            sx,sy,sh,dy=p[j];torch.testing.assert_close(g[j,:,0][r[j]],(sx*f[j,:,0]+sh*(f[j,:,1]-.5))[r[j]])
            torch.testing.assert_close(g[j,:,1][r[j]],(.5+sy*(f[j,:,1]-.5)+dy)[r[j]])
        self.assertTrue(torch.equal(r,r2));self.assertTrue(torch.equal(f[:,:,2:],g[:,:,2:]))
    def test_gradient_unused_noise_padding_nan_and_invalid_coefficients(self):
        c=frame_cache({'a':line()},4)['a'];x=c['mu'].clone().requires_grad_(True);a=augment_transport(x,c['mask'],torch.tensor([[1.1,.9,.12,.04]]));a.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all());self.assertEqual(float(x.grad[:,20:].abs().sum()),0.)
        dirty=c['mu'].clone();dirty[:,20:]=float('nan');dirty=torch.cat((dirty,torch.full((1,48,3),float('nan'))),2);mask=torch.cat((c['mask'],torch.zeros(1,3,dtype=torch.bool)),1)
        self.assertTrue(torch.isfinite(augment_transport(dirty,mask,torch.tensor([[1.,1.,0.,0.]]))).all())
        for p in [torch.zeros(1,3),torch.tensor([[0.,1.,0.,0.]]),torch.tensor([[1.,-1.,0.,0.]]),torch.tensor([[1.,1.,float('nan'),0.]])]:
            with self.assertRaises(ValueError):augment_transport(c['mu'],c['mask'],p)
    def test_dedicated_generator_reproducible_bounded_no_dropout_rng_consumption(self):
        state=torch.get_rng_state().clone();a=draw_parameters(100,torch.Generator().manual_seed(901));b=draw_parameters(100,torch.Generator().manual_seed(901))
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for j,(lo,hi) in enumerate([(.9,1.1),(.9,1.1),(-.12,.12),(-.04,.04)]):self.assertTrue(((a[:,j]>=lo)&(a[:,j]<=hi)).all())
        with self.assertRaises(ValueError):draw_parameters(1,None)
        for bad in (0,-1,1.,True):
            with self.assertRaises(ValueError):draw_parameters(bad,torch.Generator())
    def test_two_forwards_pair_dropout_rng_and_finite_backward(self):
        cfg=dict(latent_dim=48,ocr_hidden_dim=16,ocr_num_heads=2,ocr_num_layers=1);cache=frame_cache({'a':line(),'b':line(19)},4);mu,l,m=collate_latents(cache,['a','b']);aug=augment_transport(mu,m,torch.tensor([[.9,1.1,.12,.04],[1.1,.9,-.12,-.04]]))
        rng=torch.get_rng_state().clone();ends=[]
        for use in [False,True]:
            h=make_head(cfg,3,points_per_frame=4);torch.set_rng_state(rng);loss=mixed_loss(h,mu,l,m,aug,use);loss.backward();self.assertTrue(torch.isfinite(loss));ends.append(torch.get_rng_state().clone())
        self.assertTrue(torch.equal(*ends));torch.set_rng_state(rng)
    def test_pinned_parent_all_split_optimizer_and_readout_guards(self):
        splits=dict(large_train=['a','b'],small_train=['a','b'],dev=['d'],held_out=['h']);c=dict(source_sha256=SHA,pool_manifest_sha256=POOL_SHA,points_per_frame=4,feature_mode='relative_scaled',attention_radius=None,train_ids=['a','b'],dev_ids=['d'],held_out_ids=['h'],feature_calibration_ids=['a','b'],common_train_probe=['a'],posterior_evaluation_ids=['a','d','h'],schedule_seed=43,physical_batch=16,seed=42)
        saved=dict(updates=8000,config=c,optimizer_state_dict=dict(param_groups=[dict(lr=1e-4,betas=(.9,.99),weight_decay=1e-4)]));self.assertIs(validate_parent(saved,dict(splits=splits)),c)
        for field,value in [('points_per_frame',2),('feature_mode','global_raw'),('seed',137),('train_ids',['b','a']),('dev_ids',['a']),('common_train_probe',['b']),('schedule_seed',42)]:
            bad=copy.deepcopy(saved);bad['config'][field]=value
            with self.assertRaises(ValueError):validate_parent(bad,dict(splits=splits))
        bad=copy.deepcopy(saved);bad['optimizer_state_dict']['param_groups'][0]['lr']=5e-4
        with self.assertRaises(ValueError):validate_parent(bad,dict(splits=splits))
    def test_launcher_guarded_no_sibling_import(self):
        import ast
        tree=ast.parse((ROOT/'modal_ocr_augmentation.py').read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom) and n.module]
        self.assertFalse(any(m.startswith('modal_') for m in imports));self.assertIn('if not train:',(ROOT/'modal_ocr_augmentation.py').read_text())

class OCRErrorAnalysisTests(unittest.TestCase):
    def test_edit_trace_minimum_counts_empty_and_repeats(self):
        from iam_tools.inkvae import edit_distance
        for a,b in [('',''),('','abc'),('abc',''),('hello','helo'),('a','b'),('ab','ba'),('letter','leter')]:self.assertEqual(len(edit_trace(a,b)),edit_distance(a,b))
        self.assertEqual(edit_trace('a','b'),[('substitute','a','b')])
    def test_character_weighting_writer_and_verbatim_not_normalized_objective(self):
        rows=[dict(sample_id='b',text='abcd',mu=dict(decoded='abc',errors=1)),dict(sample_id='a',text='A.',mu=dict(decoded='a',errors=2))];records={'a':dict(text='A.',writer_id=1),'b':dict(text='abcd',writer_id=2)}
        r=analyze(rows,['a','b'],records);self.assertEqual(r['cer'],.5);self.assertEqual(r['lowercase_cer_same_denominator'],2/6);self.assertEqual(r['lowercase_no_ascii_punctuation_cer'],1/5);self.assertEqual(sum(r['edit_counts'].values()),3)
        for ids,rs in [(['a','a'],rows),(['x'],rows),(['a'],rows+[rows[0]])]:
            with self.assertRaises(ValueError):analyze(rs,ids,records)
        bad=copy.deepcopy(rows);bad[1]['mu']['errors']=0
        with self.assertRaises(ValueError):analyze(bad,['a'],records)

class OCRAugmentationReportTests(unittest.TestCase):
    def fixture(self):
        from iam_tools.report_ocr_augmentation import FIXED,ARMS
        c={k:k for k in FIXED};c['max_updates']=2;c['use_augmentation']=False;c['objective']='0.5clean+0.5clean;two forwards match dropout RNG'
        d=copy.deepcopy(c);d['use_augmentation']=True;d['objective']='0.5clean+0.5augmented'
        r=dict(initial_state={'head':'same','optimizer':'same','rng':'same'},sample_schedule_sha256='schedule',final_dropout_rng_sha256='dropout',final_augmentation_rng_sha256='affine',last_step=2,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True)
        logs=[dict(step=i+1,sample_ids=[str(i)],lr=1e-4,affine_parameters_sha256=str(i)) for i in range(2)]
        return dict(zip(ARMS,[c,d])),{a:copy.deepcopy(r) for a in ARMS},{a:copy.deepcopy(logs) for a in ARMS}
    def test_complete_controls_accept_and_reject_rng_and_exposure_changes(self):
        from iam_tools.report_ocr_augmentation import pairing
        c,r,l=self.fixture();self.assertTrue(all(pairing(c,r,l).values()))
        for key,value in [('final_dropout_rng_sha256','other'),('final_augmentation_rng_sha256','other'),('last_step',1),('stop_reason','wall_limit'),('entire_codec_bitwise_unchanged',False)]:
            bad=copy.deepcopy(r);bad['affine_mix'][key]=value
            with self.assertRaises(ValueError):pairing(c,bad,l)
        for key,value in [('sample_ids',['other']),('lr',5e-4),('affine_parameters_sha256','other'),('step',2)]:
            bad=copy.deepcopy(l);bad['affine_mix'][0][key]=value
            with self.assertRaises(ValueError):pairing(c,r,bad)
    def test_no_hidden_hyperparameter_or_feature_change(self):
        from iam_tools.report_ocr_augmentation import pairing
        c,r,l=self.fixture()
        for key in ['feature_stats','augmentation_limits','schedule_skip','parent_sha256','dropout','base_lr','feature_calibration_ids']:
            bad=copy.deepcopy(c);bad['affine_mix'][key]='different'
            with self.assertRaises(ValueError):pairing(bad,r,l)
        bad=copy.deepcopy(c);bad['affine_mix']['objective']='one_forward'
        with self.assertRaises(ValueError):pairing(bad,r,l)
    def test_robustness_probes_fixed_mean_only_do_not_modify_cache_or_labels(self):
        from unittest.mock import patch
        from iam_tools.report_ocr_augmentation import robustness,PROBES
        cache=frame_cache({'a':line()},4);before=cache['a']['mu'].clone();calls=[]
        def fake(head,changed,texts,splits,vocab,folder,step,posterior_ids):
            self.assertEqual(posterior_ids,());self.assertTrue(torch.equal(changed['a']['labels'],cache['a']['labels']));self.assertTrue(torch.equal(changed['a']['mask'],cache['a']['mask']))
            _,real=unpack(changed['a']['mu'],changed['a']['mask'],4);self.assertEqual(int(real.sum()),13);calls.append(step);return dict(groups={'dev':dict(mu=dict(cer=.1))})
        rng=torch.get_rng_state().clone()
        with patch('iam_tools.report_ocr_augmentation.evaluate',fake):r=robustness(None,cache,{'a':'ab'},{'dev':['a']},['a','b'],Path('.'))
        self.assertEqual(len(calls),8);self.assertEqual(set(r['probes']),set(PROBES));self.assertTrue(torch.equal(before,cache['a']['mu']));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
    def test_report_line_changes_align_by_id(self):
        from iam_tools.report_ocr_augmentation import line_changes
        a=dict(lines=[dict(sample_id='a',mu=dict(errors=2)),dict(sample_id='b',mu=dict(errors=1))]);b=dict(lines=[dict(sample_id='b',mu=dict(errors=2)),dict(sample_id='a',mu=dict(errors=1))])
        r=line_changes(a,b,['a','b']);self.assertEqual((r['improved'],r['tied'],r['worsened']),(1,0,1));self.assertEqual(r['lines'][0]['sample_id'],'a')

if __name__=='__main__':unittest.main()
