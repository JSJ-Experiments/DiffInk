from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import unittest
import numpy as np
import torch
from test_vae_ctc import VAE,small_config,REPO
from model.gmm import get_mixture_coef,get_mixture_coef_max,sample_from_params
from model.losses import bounded_pen,mixture_expectation,calibrate_anchor
from trainer.vae_trainer import train_vae_one_epoch
from dataset.transform import Transform
from iam_tools.pen_refit import refit_loss,refit_states
from iam_tools.eightline import check_config
from iam_tools.objective_study import arm_config,gradient_diagnostics
from iam_tools.lbfgs_geometry import geometry_loss
import yaml


class EnglishContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def model(self):
        cfg=small_config();cfg.trans_dropout=0;cfg.model_input_scale=.01;cfg.use_decoder_padding_mask=True
        return VAE(cfg),cfg

    def batch(self):
        data=torch.zeros(1,24,5);data[:,:,0]=torch.linspace(1,100,24);data[:,:,1]=torch.sin(torch.arange(24))*20
        data[:,:,2]=1;data[:,10,2:]=torch.tensor([0.,1.,0.]);data[:,21,2:]=torch.tensor([0.,0.,1.]);data[:,22:]=torch.tensor([0.,0.,0.,0.,1.])
        return data,torch.arange(24)[None]<22,torch.tensor([[0,1,2]]),[],torch.tensor([0])

    def test_model_adapter_public_encode_roundtrip_and_legacy_optout(self):
        model,cfg=self.model();model.eval();data=self.batch()[0].transpose(1,2)
        scaled=model.to_model_space(data)
        torch.testing.assert_close(scaled[:,:2],data[:,:2]*.01);torch.testing.assert_close(scaled[:,2:],data[:,2:])
        before=data.clone();_,mu,_=model.encode(data);_,other,_=model.encode(scaled,input_is_model_space=True)
        self.assertTrue(torch.equal(mu,other));self.assertTrue(torch.equal(data,before))
        torch.testing.assert_close(model.to_data_space(scaled.transpose(1,2)),data.transpose(1,2))
        checkpoint={'config':{'model_input_scale':.02,'trans_dropout':.2,'use_decoder_padding_mask':True}}
        model.apply_checkpoint_contract(checkpoint);self.assertEqual(cfg.model_input_scale,.02)
        self.assertEqual(model.transformer_decoder.transformer.layers[0].self_attn.dropout,.2)
        cfg.language='en'
        with self.assertRaises(ValueError):model.apply_checkpoint_contract({})

    def test_decoder_dropout_zero_and_full_point_mask_polarity(self):
        model,cfg=self.model();model.train();data,mask,labels,_,writers=self.batch()
        capture=[];hook=model.transformer_decoder.register_forward_pre_hook(lambda m,a,k:capture.append(k['padding_mask'].clone()),with_kwargs=True)
        with patch.object(model,'reparameterize',side_effect=lambda mu,logvar:mu):
            a=model(data.transpose(1,2),torch.ones(1,3),labels,writers,False,False,point_mask=mask)[0]
            b=model(data.transpose(1,2),torch.ones(1,3),labels,writers,False,False,point_mask=mask)[0]
        hook.remove();self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(capture[0],~mask))
        self.assertEqual(sum(isinstance(x,torch.nn.GroupNorm) for x in model.modules()),48)

    def test_kl_averages_valid_elements_not_sequence_length(self):
        model,_=self.model()
        for channels,length in [(8,3),(16,3),(8,9)]:
            mu=torch.ones(2,channels,length);logvar=torch.zeros_like(mu)
            self.assertAlmostEqual(float(model.kl_divergence_new(mu,logvar,torch.ones(2,length))),.5)
        mu=torch.ones(1,8,4,requires_grad=True);lv=torch.zeros_like(mu,requires_grad=True)
        with torch.no_grad():mu[:,:,3]=float('nan');lv[:,:,3]=float('nan')
        loss=model.kl_divergence_new(mu,lv,torch.tensor([[1,1,1,0]]));loss.backward()
        self.assertEqual(float(loss.detach()),.5);self.assertTrue(torch.isfinite(mu.grad).all());self.assertTrue(torch.isfinite(lv.grad).all())

    def test_gmm_sigma_parity_and_no_fake_greedy_randomness(self):
        torch.manual_seed(1);output=torch.randn(1,123,5)
        a,b=get_mixture_coef(output,20),get_mixture_coef_max(output,20)
        for x,y in zip(a,b):self.assertTrue(torch.equal(x,y))
        params=[torch.tensor([[.25]*5,[.75]*5]),torch.tensor([[1.]*5,[3.]*5]),torch.tensor([[2.]*5,[4.]*5]),
                torch.ones(2,5),torch.ones(2,5),torch.zeros(2,5),torch.tensor([[1.]*5,[0.]*5,[0.]*5])]
        with patch('torch.distributions.Categorical.sample',side_effect=AssertionError('deterministic decode must not sample')):
            expected=sample_from_params(params,max_seq_len=3,mode='expectation')
            greedy=sample_from_params(params,max_seq_len=3,greedy=True)
        self.assertEqual(expected.shape,(3,5));np.testing.assert_allclose(expected[:,:2],[[2.5,3.5]]*3)
        np.testing.assert_allclose(greedy[:,:2],[[3.,4.]]*3)
        with patch('torch.distributions.Categorical.sample',return_value=torch.tensor(0)) as sampled:
            random=sample_from_params(params,temp=0,max_seq_len=3,mode='sample')
        self.assertEqual(sampled.call_count,3);np.testing.assert_allclose(random[:,:2],[[1.,2.]]*3)

    def test_binary_is_explicit_known_length_and_padding_not_supervision(self):
        targets=torch.tensor([0,0,1,0,2]);logits=torch.randn(5,3,requires_grad=True)
        a=refit_loss(logits,targets,'binary_forced_final');a.backward()
        self.assertEqual(logits.grad[-1].count_nonzero(),0);self.assertEqual(logits.grad[:,2].count_nonzero(),0)
        states=refit_states(logits,'binary_forced_final');self.assertEqual(int(states[-1]),2);self.assertFalse((states[:-1]==2).any())
        mask=torch.tensor([[True]*5+[False]*3]);pad=torch.cat([logits.detach(),torch.full((3,3),float('nan'))])[None]
        labels=torch.tensor([[0,0,1,0,2,2,2,2]])
        actual=bounded_pen(pad,labels,mask)
        reference=refit_loss(logits.detach(),targets,'bounded_three_state')
        torch.testing.assert_close(actual,reference)

    def test_rotation_can_be_disabled_without_disabling_scaling(self):
        points=self.batch()[0][0].numpy()
        np.testing.assert_array_equal(Transform(1000,prob=1,rotation_degrees=0,scaling=False)(points),points)
        scaled=Transform(1000,prob=1,rotation_degrees=0,scaling=True)(points)
        np.testing.assert_array_equal(scaled[:,2:],points[:,2:]);self.assertFalse(np.array_equal(scaled[:,:2],points[:,:2]))

    def training_cfg(self,cfg):
        for k,v in {'gmm_logspace':True,'gmm_weight':1,'pen_weight':1,'expected_xy_weight':.1,'ctc_weight':0,'style_weight':0,'kl_weight':0,
                    'pen_policy':'bounded_three_state','gradient_accumulation_steps':2,'grad_clip':5,'train_batch_size':1,'padding_strategy':'microbatch'}.items():setattr(cfg,k,v)
        return cfg

    def test_actual_trainer_accumulation_tail_and_scaled_targets(self):
        model,cfg=self.model();self.training_cfg(cfg);batch=self.batch()
        optimizer=torch.optim.SGD(model.parameters(),lr=1e-5);steps=[]
        with patch.object(model,'get_ocr_loss',side_effect=AssertionError('CTC off')),patch.object(model,'get_style_loss',side_effect=AssertionError('style off')):
            rows=train_vae_one_epoch(model,cfg,[batch]*3,optimizer,None,0,1,'cpu',on_optimizer_step=steps.append)
        self.assertEqual([r['microbatches'] for r in rows],[2,1]);self.assertEqual(len(steps),2)
        self.assertTrue(all(np.isfinite(r['loss']) for r in rows))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_accumulated_metrics_are_effective_batch_means_not_last_sample(self):
        model,cfg=self.model();self.training_cfg(cfg);optimizer=torch.optim.SGD(model.parameters(),lr=0)
        values=iter([1.,3.,5.])
        def fake_terms(*args):return {'gmm':next(values)+next(model.parameters()).sum()*0}
        with patch('trainer.vae_trainer.loss_terms',side_effect=fake_terms):
            rows=train_vae_one_epoch(model,cfg,[self.batch()]*3,optimizer,None,0,1,'cpu',on_optimizer_step=lambda row:None)
        self.assertEqual([r['loss'] for r in rows],[2.,5.])
        self.assertEqual(rows[0]['last_microbatch_loss'],3.)
        self.assertEqual(rows[0]['loss_log_scope'],'effective_batch_mean')

    def test_anchor_calibration_does_not_step_or_populate_grad_buffers(self):
        model,cfg=self.model();self.training_cfg(cfg);before=deepcopy(model.state_dict())
        result=calibrate_anchor(model,[self.batch()],cfg,'cpu',.15)
        self.assertGreater(result['weight'],0)
        self.assertAlmostEqual(result['lines'][0]['actual_initial_weighted_ratio'],.15)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_eight_line_gpu_guard_caps_and_auxiliaries(self):
        cfg=yaml.safe_load((REPO/'configs/engineering_english.yaml').read_text());check_config(cfg)
        for k,v in [('trans_dropout',.1),('ctc_weight',1),('max_optimizer_updates',201),('train_batch_size',8),('sampled_z_evaluations',1)]:
            bad=deepcopy(cfg);bad[k]=v
            with self.assertRaises(ValueError):check_config(bad)

    def test_objective_arms_change_only_gmm_coefficient(self):
        base=yaml.safe_load((REPO/'configs/engineering_english.yaml').read_text())
        a,b=arm_config(base,'xy_pen'),arm_config(base,'gmm_xy_pen')
        self.assertEqual({k for k in a if a[k]!=b[k]},{'gmm_weight','objective_arm'})
        self.assertEqual(a['max_optimizer_updates'],1000)
        self.assertEqual(base['max_optimizer_updates'],200)
        for key in ('ctc_weight','kl_weight','style_weight'):self.assertEqual(a[key],0)
        with self.assertRaises(ValueError):arm_config(base,'unknown')

    def test_objective_gradient_diagnostic_is_no_update_and_no_buffers(self):
        model,cfg=self.model();self.training_cfg(cfg);before=deepcopy(model.state_dict())
        rows=gradient_diagnostics(model,[self.batch()],vars(cfg),'cpu')
        self.assertTrue(-1.001<=rows[0]['gmm_xy_cosine']<=1.001)
        self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_lbfgs_deterministic_closure_has_gradients_without_latent_sampling(self):
        model,cfg=self.model();model.eval();raw,mask,*_=self.batch();raw=raw.transpose(1,2)
        optimizer=torch.optim.LBFGS(model.parameters(),max_iter=2,history_size=2,line_search_fn='strong_wolfe')
        def closure():
            optimizer.zero_grad(set_to_none=True);loss=geometry_loss(model,raw,mask);loss.backward();return loss
        with patch.object(model,'reparameterize',side_effect=AssertionError('mean diagnostic must not sample')):
            before=float(geometry_loss(model,raw,mask).detach());optimizer.step(closure)
            after=float(geometry_loss(model,raw,mask).detach())
        self.assertLess(after,before)
        self.assertTrue(all(p.grad is None for p in model.conv_logvar.parameters()))

    def test_cached_ctc_head_training_does_not_call_encoder_or_decoder(self):
        model,cfg=self.model();model.requires_grad_(False);model.ocr_model.requires_grad_(True)
        features=torch.randn(1,cfg.latent_dim,8);labels=torch.tensor([[0,1,1]])
        with patch.object(model,'encode',side_effect=AssertionError('frozen cached latent')),patch.object(model,'decode',side_effect=AssertionError('geometry must not run')):
            loss=model.get_ocr_loss(features,labels,torch.ones(1,8));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(loss.detach()),0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.ocr_model.parameters()))

if __name__=='__main__':unittest.main()
