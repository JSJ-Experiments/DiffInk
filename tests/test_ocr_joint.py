from pathlib import Path
import copy,sys,unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO));sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_ocr_frames import line
from test_vae_ctc import small_config
from iam_tools.ocr_recurrent import make_head
from iam_tools.ocr_frame_study import frame_cache
from iam_tools.ocr_joint_adapter import PolyphaseGRUOCR,input_gradient_gru,CONTRACT
from iam_tools.frozen_ocr_study import collate_latents
from model.vae import VAE


class JointReaderAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def fixture(self):
        original={'a':line(13),'b':line(25)}
        head=make_head(dict(latent_dim=48,ocr_hidden_dim=16),3,None).eval()
        wrapper=PolyphaseGRUOCR(head,None).eval()
        x,labels,lm=collate_latents(original,['a','b'])
        pm=torch.arange(x.shape[-1]*8)[None]<torch.tensor([13,25])[:,None]
        return original,wrapper,x,labels,lm,pm

    def test_matches_original_standalone_reader(self):
        original,w,x,labels,lm,pm=self.fixture();framed=frame_cache(original,4)
        z,_,mask=collate_latents(framed,['a','b'])
        with torch.no_grad():
            expected=w.head(z,padding_mask=~mask)
            actual=w(x,padding_mask=~lm,point_mask=pm)
        torch.testing.assert_close(actual[:expected.shape[0]],expected,atol=2e-6,rtol=2e-6)

    def test_ctc_input_gradient_but_no_reader_weight_gradient(self):
        _,w,x,l,lm,pm=self.fixture();x.requires_grad_()
        loss=w.get_ocr_loss(x,l,lm,point_mask=pm);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(float(x.grad[:,:40].abs().sum()),0.)
        self.assertEqual(float(x.grad[:,40:].abs().sum()),0.)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in w.parameters()))

    def test_predicted_eoc_cannot_hide_points_or_shrink_ctc_clock(self):
        _,w,x,_,lm,pm=self.fixture();_,mask=w.framed(x,lm,pm)
        x=x.clone();x[:,:40].reshape(2,8,5,-1)[:,:,4]=100.
        _,changed=w.framed(x,lm,pm);torch.testing.assert_close(changed,mask)
        self.assertEqual(mask.sum(1).tolist(),[4,7])

    def test_nan_padding_and_unused_fields_do_not_change_valid_logits(self):
        _,w,x,_,lm,pm=self.fixture()
        with torch.no_grad():expected=w(x,padding_mask=~lm,point_mask=pm)
        x=x.masked_fill(~lm[:,None],float('nan'));x[:,40:]=float('nan')
        with torch.no_grad():actual=w(x,padding_mask=~lm,point_mask=pm)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_missing_inconsistent_nonprefix_masks_rejected(self):
        _,w,x,l,lm,pm=self.fixture()
        with self.assertRaises(ValueError):w.get_ocr_loss(x,l,lm)
        with self.assertRaises(ValueError):w.get_ocr_loss(x,l,~lm,point_mask=pm)
        invalid=pm.clone();invalid[0,1]=False
        with self.assertRaises(ValueError):w.get_ocr_loss(x,l,lm,point_mask=invalid)

    def test_empty_infeasible_targets_rejected_not_silently_zeroed(self):
        _,w,x,_,lm,pm=self.fixture()
        with self.assertRaises(ValueError):w.get_ocr_loss(x,torch.full((2,2),-1),lm,point_mask=pm)
        with self.assertRaises(ValueError):w.get_ocr_loss(x,torch.zeros((2,6),dtype=torch.long),lm,point_mask=pm)

    def test_model_train_cannot_enable_reader_dropout_or_unfreeze(self):
        _,w,*_=self.fixture();w.train(True)
        self.assertFalse(w.training);self.assertFalse(w.head.rnn.training)
        self.assertTrue(all(not p.requires_grad for p in w.parameters()))

    def test_training_backend_no_dropout_same_function_and_restores_flags(self):
        _,w,x,_,lm,pm=self.fixture();f,valid=w.framed(x,lm,pm)
        expected=w.head.logits_from_features(f,valid)
        before=torch.get_rng_state().clone()
        with input_gradient_gru(w.head,True):
            self.assertTrue(w.head.rnn.training);self.assertEqual(w.head.rnn.dropout,0.)
            actual=w.head.logits_from_features(f,valid)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        self.assertFalse(w.head.rnn.training);self.assertEqual(w.head.rnn.dropout,.1)
        with self.assertRaises(RuntimeError):
            with input_gradient_gru(w.head,True):raise RuntimeError('intentional')
        self.assertFalse(w.head.rnn.training);self.assertEqual(w.head.rnn.dropout,.1)

    def test_actual_vae_forward_to_adapter_and_encoder_backward(self):
        from iam_tools.identity_geometry_probe import initialize_identity_geometry
        from iam_tools.ocr_pool_study import single_batch
        cfg=small_config();cfg.hidden_dims=[48,48,48];cfg.latent_dim=48
        cfg.decoder_dims=[48,48,128];cfg.style_classifier_dim=48;cfg.ocr_hidden_dim=16
        cfg.model_input_scale=.01;cfg.trans_dropout=0.;cfg.use_decoder_padding_mask=True;cfg.trans_hidden_dim=16
        m=VAE(cfg).eval();initialize_identity_geometry(m)
        m.ocr_model=PolyphaseGRUOCR(make_head(dict(latent_dim=48,ocr_hidden_dim=16),4,None),None)
        points=torch.zeros(25,5);points[:,:2]=torch.randn(25,2);points[:,2]=1.
        points[-1,2]=0.;points[-1,4]=1.
        raw,pm,labels=single_batch(points,'ab',['a','b','c']);lm=pm.reshape(1,-1,8).any(-1)
        with patch.object(m.ocr_model,'get_ocr_loss',wraps=m.ocr_model.get_ocr_loss) as actual:
            _,ctc,_,_=m(raw,lm,labels,torch.zeros(1,dtype=torch.long),get_style_loss=False,point_mask=pm)
        self.assertEqual(actual.call_count,1);self.assertIs(actual.call_args.kwargs['point_mask'],pm)
        ctc.backward();self.assertTrue(torch.isfinite(ctc))
        self.assertGreater(float(m.encoder.conv_1.weight.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None for p in m.ocr_model.parameters()))
        with self.assertRaisesRegex(ValueError,'explicit real-point lengths'):
            m(raw,lm,labels,torch.zeros(1,dtype=torch.long),get_style_loss=False)

    def test_research_checkpoint_rejected_by_default_loader(self):
        m=VAE(small_config());saved=dict(config=dict(research_ocr_contract=CONTRACT))
        with self.assertRaisesRegex(ValueError,'dedicated adapter'):m.apply_checkpoint_contract(saved)
        m.apply_checkpoint_contract(saved,allow_research_ocr=True)

    def test_latent_vs_rendered_diagnostic_runs_with_real_frozen_reader(self):
        from iam_tools.identity_geometry_probe import initialize_identity_geometry
        from iam_tools.ocr_pool_study import single_batch
        from iam_tools.report_ocr_joint import latent_vs_drawing
        cfg=small_config();cfg.hidden_dims=[48,48,48];cfg.latent_dim=48
        cfg.decoder_dims=[48,48,128];cfg.style_classifier_dim=48;cfg.ocr_hidden_dim=16
        cfg.trans_hidden_dim=16;cfg.model_input_scale=.01;cfg.trans_dropout=0.
        m=VAE(cfg).eval();initialize_identity_geometry(m)
        m.ocr_model=PolyphaseGRUOCR(make_head(dict(latent_dim=48,ocr_hidden_dim=16),4,None),None)
        m.requires_grad_(False)
        points=torch.zeros(25,5);points[:,:2]=torch.randn(25,2);points[:,2]=1.
        points[-1,2]=0.;points[-1,4]=1.
        batch=single_batch(points,'ab',['a','b','c'])
        result=latent_vs_drawing(m,{'a':batch},{'a':dict(text='ab')},['a','b','c'],['a'])
        row=result['lines'][0]
        self.assertEqual(row['input_gradient_energy_fractions']['unused'],0.)
        self.assertAlmostEqual(sum(row['input_gradient_energy_fractions'].values()),1.,places=5)
        self.assertLess(abs(row['ctc']['latent']-row['ctc']['rendered']),.001)
        self.assertTrue(all(p.grad is None for p in m.parameters()))


class JointStudyGuardTests(unittest.TestCase):
    def test_completed_cpu_cache_reuses_only_bound_data_and_vocab(self):
        import tempfile
        from types import SimpleNamespace
        from iam_tools.report_ocr_joint import cached_cpu_evaluation
        class Fake(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
                self.ocr_model=SimpleNamespace(stats=None)
        model=Fake();batches={'a':(torch.zeros(1,5,8),torch.ones(1,8,dtype=torch.bool),torch.zeros(1,1,dtype=torch.long))}
        records={'a':dict(text='a',writer_id='1')};splits={'probe':['a']};row=dict(lines=[dict(sample_id='a')])
        with tempfile.TemporaryDirectory() as tmp,patch('iam_tools.report_ocr_joint.evaluate',return_value=row) as ev:
            a=cached_cpu_evaluation(model,batches,records,['a','b'],splits,tmp,0,0,False)
            b=cached_cpu_evaluation(model,batches,records,['a','b'],splits,tmp,0,0,False)
            self.assertEqual(a,b);self.assertEqual(ev.call_count,1)
            cached_cpu_evaluation(model,batches,records,['b','a'],splits,tmp,0,0,False)
            self.assertEqual(ev.call_count,2)
            batches['a'][0][0,0,0]=1.
            cached_cpu_evaluation(model,batches,records,['b','a'],splits,tmp,0,0,False)
            self.assertEqual(ev.call_count,3)

    def test_named_curve_pool_exclusions_explicitly_require_full_eight_supplement(self):
        from iam_tools.ocr_joint_study import named_scope,LEGACY_EIGHT
        records={i:{} for i in LEGACY_EIGHT if i!='a07-421z-02'}
        scope=named_scope(records)
        self.assertEqual(scope['outside_pool'],['a07-421z-02'])
        self.assertEqual(len(scope['legacy_eight']),8)
        self.assertIn('all8',scope['required_supplement'])

    def test_rendered_repacking_preserves_chronology_pens_and_xy_gradients(self):
        from iam_tools.report_ocr_joint import pack_rendered_points
        from iam_tools.ocr_frame_study import split_tensor
        xy=torch.arange(64,dtype=torch.float32).reshape(2,16,2).requires_grad_()
        states=torch.arange(32).reshape(2,16)%3
        packed=pack_rendered_points(xy,states,48)
        four=split_tensor(packed,4)[:,:20].reshape(2,4,5,-1).permute(0,3,1,2).reshape(2,16,5)
        torch.testing.assert_close(four[:,:,:2],xy)
        torch.testing.assert_close(four[:,:,2:].argmax(2),states)
        self.assertEqual(float(packed[:,40:].detach().abs().sum()),0.)
        four[:,:,:2].sum().backward();torch.testing.assert_close(xy.grad,torch.ones_like(xy))
        with self.assertRaises(ValueError):pack_rendered_points(xy[:,:15],states[:,:15],48)

    def fixture(self):
        from test_latent_integration import fake_row
        row=fake_row()
        for r in row['lines']:
            for v in [r['mu']]+r['sampled']:
                v['geometry']['target_corner_turn_error_degrees']=dict(p90=5.)
                v['geometry']['target_shallow_turn_error_degrees']=dict(p90=2.)
        return row

    def test_single_corner_regression_rejected_even_if_xy_unchanged(self):
        from iam_tools.ocr_joint_study import gate
        ref=self.fixture();row=copy.deepcopy(ref);ids=[r['sample_id'] for r in ref['lines']]
        self.assertTrue(gate(row,ref,ids)['passed'])
        row['lines'][0]['mu']['geometry']['target_corner_turn_error_degrees']['p90']=9.
        self.assertFalse(gate(row,ref,ids)['passed'])

    def test_one_sampled_boundary_error_rejected(self):
        from iam_tools.ocr_joint_study import gate
        ref=self.fixture();row=copy.deepcopy(ref)
        row['lines'][1]['sampled'][0]['pen']['non_final_false_eoc_count']=1
        self.assertFalse(gate(row,ref,['1'])['passed'])

    def test_missing_line_or_draw_cannot_silently_pass(self):
        from iam_tools.ocr_joint_study import gate
        ref=self.fixture();row=copy.deepcopy(ref);row['lines'].pop()
        with self.assertRaises(ValueError):gate(row,ref,['7'])
        row=copy.deepcopy(ref);row['lines'][0]['sampled']=[]
        with self.assertRaises(ValueError):gate(row,ref,['0'])

    def test_selection_ignores_dev_scores(self):
        from iam_tools.ocr_joint_study import selection
        row=dict(groups=dict(train_probe=dict(mu=dict(cer=.1,mean_ctc_loss=.3,x_rmse=.001,y_rmse=.002)),dev=dict(mu=dict(cer=.9))))
        a=selection(row);row['groups']['dev']['mu']['cer']=0.;self.assertEqual(a,selection(row))

    def test_cli_default_and_bad_pool_do_not_request_gpu(self):
        import ast
        tree=ast.parse((ROOT/'modal_ocr_joint.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');main.decorator_list=[]
        calls=[];namespace=dict(research=type('Fake',(),{'remote':lambda *a:calls.append(a)})(),print=lambda *a:None)
        exec(compile(ast.Module(body=[main],type_ignores=[]),'jointcli','exec'),namespace)
        namespace['main']();self.assertEqual(calls,[])
        with self.assertRaises(ValueError):namespace['main'](train=True,pool_sha='bad')
        with self.assertRaises(ValueError):namespace['main'](train=True,steps=401)
        self.assertEqual(calls,[])


if __name__=='__main__':unittest.main()
