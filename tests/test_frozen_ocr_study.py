"""Padding-safe OCR head refitting, no raw trajectory minibatch leakage."""
from pathlib import Path
import sys
import unittest
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from model.ocr import ChineseHandwritingOCR
from iam_tools.frozen_ocr_study import bucket_schedule,collate_latents,summarize,batch_parity,ablate_inactive_projection,head_optimizer,head_training_loss


class FrozenOCRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def setUp(self):
        torch.manual_seed(9);self.head=ChineseHandwritingOCR(4,8,2,2,4,dropout=0).eval()
        self.cache={str(i):dict(mu=torch.randn(1,4,3+i),lv=torch.full((1,4,3+i),-8.),
            labels=torch.tensor([[0,1,2]]),mask=torch.ones(1,3+i,dtype=torch.bool)) for i in range(6)}
    def test_masked_padded_batch_matches_individual_logits_and_ctc(self):
        class Wrapper:
            def __init__(self,h):self.ocr_model=h
            def get_ocr_loss(self,*args):return self.ocr_model.get_ocr_loss(*args)
        result=batch_parity(Wrapper(self.head),self.cache,list(self.cache));self.assertLess(result['max_valid_logit_difference'],2e-5)
    def test_padded_nan_features_have_no_output_or_gradient_effect(self):
        features=torch.randn(1,4,6,requires_grad=True);mask=torch.tensor([[True,True,True,False,False,False]])
        with torch.no_grad():features[:,:,3:]=float('nan')
        out=self.head(features,padding_mask=~mask)
        self.assertTrue(torch.isfinite(out).all());out[:3].square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all());self.assertEqual(float(features.grad[:,:,3:].abs().sum()),0)
        torch.testing.assert_close(out[:3],self.head(features[:,:,:3]),atol=2e-6,rtol=2e-6)
    def test_optional_mask_and_explicit_all_valid_agree(self):
        x=self.cache['0']['mu'];labels=self.cache['0']['labels']
        torch.testing.assert_close(self.head.get_ocr_loss(x,labels),self.head.get_ocr_loss(x,labels,self.cache['0']['mask']))
    def test_released_binary_float_mask_matches_boolean(self):
        c=self.cache['0']
        torch.testing.assert_close(self.head.get_ocr_loss(c['mu'],c['labels'],c['mask']),self.head.get_ocr_loss(c['mu'],c['labels'],c['mask'].float()))
    def test_invalid_mask_shapes_types_empty_or_nonprefix_rejected(self):
        x=self.cache['0']['mu'];y=self.cache['0']['labels']
        for mask in (torch.ones(1,2,dtype=torch.bool),torch.full((1,3),.5),torch.zeros(1,3,dtype=torch.bool),torch.tensor([[True,False,True]])):
            with self.assertRaises(ValueError):self.head.get_ocr_loss(x,y,mask)
    def test_collation_only_pads_frozen_latents_and_preserves_labels(self):
        ids=['0','5'];z,labels,mask=collate_latents(self.cache,ids)
        self.assertEqual(z.shape,(2,4,8));self.assertEqual(int(mask[0].sum()),3)
        torch.testing.assert_close(z[0,:,:3],self.cache['0']['mu'][0]);self.assertEqual(float(z[0,:,3:].abs().sum()),0)
        self.assertFalse(z.requires_grad);self.assertTrue(torch.equal(labels,self.cache['0']['labels'].expand(2,-1)))
        before={i:self.cache[i]['mu'].clone() for i in ids};collate_latents(self.cache,ids,sampled=True)
        for i in ids:self.assertTrue(torch.equal(before[i],self.cache[i]['mu']))
    def test_bucket_schedule_is_train_only_paired_and_complete_epochs(self):
        ids=list(self.cache)[:4];a=list(bucket_schedule(self.cache,ids,8,batch_size=2));b=list(bucket_schedule(self.cache,ids,8,batch_size=2))
        self.assertEqual(a,b)
        for j in range(0,8,2):self.assertEqual(sorted(a[j]+a[j+1]),ids)
        with self.assertRaises(ValueError):list(bucket_schedule(self.cache,['0','0'],3))
    def test_inactive_noise_ablation_requires_analytical_zero_means(self):
        from types import SimpleNamespace
        codec=SimpleNamespace(conv_mu=torch.nn.Conv1d(4,4,1),ocr_model=self.head)
        with torch.no_grad():codec.conv_mu.weight[2:]=0.;codec.conv_mu.bias[2:]=0.
        x=torch.randn(1,4,6);x[:,2:]=0.;before=self.head(x).detach();active=self.head.input_proj.weight[:,:2].detach().clone()
        info=ablate_inactive_projection(codec,2)
        self.assertEqual(info['inactive_channels'],2)
        torch.testing.assert_close(self.head(x),before,rtol=0,atol=0)
        self.assertTrue(torch.equal(active,self.head.input_proj.weight[:,:2]))
        self.assertEqual(int(torch.count_nonzero(self.head.input_proj.weight[:,2:])),0)
        with torch.no_grad():codec.conv_mu.bias[2]=1e-9
        with self.assertRaises(ValueError):ablate_inactive_projection(codec,2)
    def test_mean_training_gives_no_gradient_to_pure_noise_input_columns(self):
        x=torch.randn(1,4,6);x[:,2:]=0.
        self.head(x).square().sum().backward()
        self.assertGreater(float(self.head.input_proj.weight.grad[:,:2].abs().sum()),0)
        self.assertEqual(float(self.head.input_proj.weight.grad[:,2:].abs().sum()),0)
    def test_restored_head_moments_and_step_are_not_reset(self):
        import copy
        optimizer,_=head_optimizer(self.head,{})
        x=self.cache['0']['mu'];optimizer.zero_grad();self.head(x).square().mean().backward();optimizer.step()
        parent=dict(ctc_head_updates=1000,ocr_optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
        restored,previous=head_optimizer(self.head,parent)
        self.assertEqual(previous,1000);self.assertEqual(restored.param_groups[0]['lr'],1e-4)
        for p,state in optimizer.state.items():
            for key,value in state.items():torch.testing.assert_close(restored.state[p][key],value,rtol=0,atol=0)
        self.assertEqual(parent['ocr_optimizer_state_dict']['param_groups'][0]['lr'],5e-4)
        with self.assertRaises(ValueError):head_optimizer(self.head,dict(ctc_head_updates=1))
    def test_paired_mean_and_stochastic_arms_consume_identical_rng(self):
        class Recorder:
            def __init__(self):self.features=[];self.masks=[]
            def get_ocr_loss(self,x,y,m):
                self.features.append(x.clone());self.masks.append(m.clone())
                return torch.nn.functional.dropout(x,p=.1,training=True).square().mean()
        states=[];models=[]
        for sampled in (False,True):
            torch.manual_seed(771);model=Recorder()
            head_training_loss(model,self.cache,['0','5'],resumed=True,sampled=sampled)
            states.append(torch.get_rng_state());models.append(model)
        self.assertTrue(torch.equal(states[0],states[1]))
        torch.testing.assert_close(models[0].features[0],models[1].features[0],rtol=0,atol=0)
        torch.testing.assert_close(models[0].features[0],models[0].features[1],rtol=0,atol=0)
        self.assertFalse(torch.equal(models[1].features[0],models[1].features[1]))
    def test_approximate_inactive_ablation_requires_explicit_research_optin(self):
        from types import SimpleNamespace
        codec=SimpleNamespace(conv_mu=torch.nn.Conv1d(4,4,1),ocr_model=self.head)
        with self.assertRaises(ValueError):ablate_inactive_projection(codec,2)
        info=ablate_inactive_projection(codec,2,require_exact=False)
        self.assertFalse(info['mu_weight_and_bias_exactly_zero']);self.assertFalse(info['require_exact'])
    def test_character_weighted_cer_repeat_groups(self):
        rows=[]
        for i,n,error in [('a',2,1),('b',8,0)]:
            v=dict(errors=error,characters=n,blank_frame_fraction=.5)
            rows.append(dict(sample_id=i,adjacent_repeats=i=='a',mu=v,sampled=[v,v],ctc_loss=1.))
        result=summarize(rows,['a','b']);self.assertEqual(result['mu']['cer'],.1)
        self.assertEqual(result['sampled']['cer'],.1);self.assertEqual(result['mu']['repeats']['cer'],.5)
        self.assertEqual(result['mu']['exact_lines'],1)

if __name__=='__main__':unittest.main()
