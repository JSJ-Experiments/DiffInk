import unittest,sys
from pathlib import Path
from unittest.mock import patch
import torch
from iam_tools.compact_dit import loss,to_x0,coefficients,X0Adapter,matched_initial_state,ARMS
from iam_tools.corpus_dit import sample,training_loss
REPO=Path(__file__).resolve().parents[1]/'third_party/DiffInk'
if not REPO.is_dir():REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from model.dit import DiT
from utils.utils import ModelConfig

class Fixed(torch.nn.Module):
    def __init__(self,out):super().__init__();self.out=torch.nn.Parameter(out);self.seen=[]
    def forward(self,**kw):self.seen.append(kw);return self.out

class CompactDiTTests(unittest.TestCase):
    def inputs(self):
        g=torch.Generator().manual_seed(143);clean=torch.randn(2,6,384,generator=g);eps=torch.randn(2,6,384,generator=g)
        alpha=torch.tensor([.9999,.8,.2,.0001]);t=torch.tensor([0,3]);mask=torch.arange(6)[None]<torch.tensor([6,4])[:,None]
        return clean,eps,alpha,t,mask,torch.tensor([2,1]),torch.tensor([[1,2],[2,3]])
    def test_v_rotation_inverts_at_all_noise_levels(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();a,b=coefficients(alpha,t,clean);noisy=a*clean+b*eps;v=a*eps-b*clean
        torch.testing.assert_close(to_x0(v,noisy,a,b,'v'),clean)
        torch.testing.assert_close(to_x0(clean,noisy,a,b,'x0'),clean)
        with self.assertRaises(ValueError):to_x0(v,noisy,a,b,'eps')
    def test_perfect_targets_mask_and_gradient_all_four_arms(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();a,b=coefficients(alpha,t,clean)
        for channels,pred in ARMS.values():
            target=clean[...,:channels] if pred=='x0' else (a*eps-b*clean)[...,:channels]
            output=target.clone();output[0,:2]+=100;output[1,:1]+=100;output[1,4:]+=100
            model=Fixed(output);r=loss(model,clean,text,mask,prefix,t,alpha,1000.,True,False,channels,pred,eps)
            self.assertEqual(float(r['loss']),0.);self.assertEqual(int(r['active_mask'].sum()),7)
            self.assertLess(float(r['active40_x0_mse']),1e-12);r['loss'].backward();self.assertEqual(float(model.out.grad.abs().sum()),0.)
            torch.testing.assert_close(model.seen[0]['x'][0,:2],clean[0,:2,:channels])
    def test_text_drop_removes_reference_and_padding_is_excluded(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();m=Fixed(torch.zeros(2,6,40))
        r=loss(m,clean,text,mask,prefix,t,alpha,1000.,True,True,40,'x0',eps)
        self.assertTrue(torch.equal(r['active_mask'],mask));self.assertTrue(m.seen[0]['drop_cond']);self.assertIsNone(r['unused344_objective'])
        expected=clean[...,:40][mask]
        torch.testing.assert_close(r['loss'],expected.square().mean())
    def test_field_metrics_follow_polyphase_xy_layout_not_first16(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();pred=clean[...,:40].clone()
        indices=[j for j in range(40) if j%5<2];pred[...,indices]+=2
        r=loss(Fixed(pred),clean,text,mask,prefix,t,alpha,1000.,False,False,40,'x0',eps)
        self.assertEqual(float(r['xy16_x0_mse']),4.);self.assertEqual(float(r['pen24_x0_mse']),0.)
    def test_shared_diffusion_noise_is_identical_across_channel_arms(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();rows=[]
        for channels in [40,384]:
            torch.manual_seed(123);r=loss(Fixed(torch.zeros(2,6,channels)),clean,text,mask,prefix,t,alpha,1000.,False,False,channels,'x0');rows.append((r['noisy'],torch.get_rng_state()))
        torch.testing.assert_close(rows[0][0],rows[1][0][...,:40]);self.assertTrue(torch.equal(rows[0][1],rows[1][1]))
    def test_adapter_uses_noisy_suffix_for_v_and_zero_unused_decode(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();a,b=coefficients(alpha,t,clean);noisy=a*clean+b*eps;v=(a*eps-b*clean)[...,:40]
        model=X0Adapter(Fixed(v),alpha,40,'v').eval();out=model(x=clean,noise=noisy,text=text,time=t.float()/1000,mask=mask,drop_text=False,drop_cond=True)
        torch.testing.assert_close(out[...,:40][mask],clean[...,:40][mask]);self.assertEqual(float(out[...,40:].abs().sum()),0.);self.assertEqual(float(out[~mask].abs().sum()),0.)
    def test_v_zero_model_has_low_noise_identity_not_a_learning_claim(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();a,b=coefficients(alpha,t,clean);noisy=a*clean+b*eps
        out=to_x0(torch.zeros_like(clean),noisy,a,b,'v')
        self.assertLess(float((out[0]-clean[0]).square().mean()),.001)
        self.assertGreater(float((out[1]-clean[1]).square().mean()),.5)
    def test_matched_init_keeps_backbone_and_all_text_columns(self):
        base=dict(dim=32,latent_dim=384,num_text_embedding=12,text_dim=16,text_mask_padding=True,conv_layers=1,dim_head=8,depth=2,heads=4,ff_mult=2,dropout=0.,long_skip_connection=False)
        torch.manual_seed(4);a=DiT(ModelConfig(base));base['latent_dim']=40;b=DiT(ModelConfig(base));s=matched_initial_state(a.state_dict(),b.state_dict());b.load_state_dict(s)
        for key,value in b.state_dict().items():
            if key not in ['input_embed.proj.weight','proj_out.weight','proj_out.bias']:torch.testing.assert_close(value,a.state_dict()[key])
        torch.testing.assert_close(b.input_embed.proj.weight[:,40:],a.input_embed.proj.weight[:,384:])
    def test_cfg_conversion_commutes_with_v_guidance(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs();a,b=coefficients(alpha,t,clean);noisy=a*clean+b*eps
        one=torch.ones_like(clean);two=2*one;g=2.5
        lhs=to_x0(one+g*(two-one),noisy,a,b,'v');u=to_x0(one,noisy,a,b,'v');c=to_x0(two,noisy,a,b,'v')
        torch.testing.assert_close(lhs,u+g*(c-u))
    def test_full_x0_matches_original_diffusion_noise_layout_and_rng(self):
        from model.diffusion import Diffusion
        clean,eps,alpha,t,mask,prefix,text=self.inputs();d=Diffusion(device='cpu',schedule_type='cosine');m=Fixed(torch.zeros_like(clean))
        torch.manual_seed(41);old=training_loss(m,clean,text,mask,prefix,t,d,1000.,True,False);oldnoise=m.seen[-1]['noise'].clone();oldrng=torch.get_rng_state()
        torch.manual_seed(41);new=loss(m,clean,text,mask,prefix,t,d.alpha_hat,1000.,True,False,384,'x0');newrng=torch.get_rng_state()
        torch.testing.assert_close(oldnoise,new['noisy'],rtol=0,atol=0);self.assertTrue(torch.equal(oldrng,newrng));torch.testing.assert_close(old['loss'],new['loss'],rtol=0,atol=0)
    def test_decoder_gate_accepts_negligible_coupling_but_rejects_visible_change(self):
        from iam_tools.compact_dit_study import unused_gate
        class Codec(torch.nn.Module):
            def __init__(self,weight):super().__init__();self.p=torch.nn.Parameter(torch.tensor(float(weight)))
            def decode(self,z):
                out=torch.zeros(len(z),5,z.shape[-1]);out[:,0]=1;out[:,3:5]=z[:,40:42]*self.p;return out
        items={'a':{'mu':torch.zeros(4,384)}};stats={'mean':torch.zeros(384),'std':torch.ones(384)}
        with patch('model.losses.mixture_expectation',side_effect=lambda x:x[:,3:5].transpose(1,2)):
            small=unused_gate(Codec(1e-5),items,stats,['a']);self.assertTrue(small['passed']);self.assertGreater(small['rows'][1]['max_xy_change'],0.)
            with self.assertRaises(ValueError):unused_gate(Codec(.1),items,stats,['a'])
    def test_shape_timestep_and_empty_suffix_fail_closed(self):
        clean,eps,alpha,t,mask,prefix,text=self.inputs()
        for bad in [t.float(),torch.tensor([-1,2]),torch.tensor([0,4])]:
            with self.assertRaises(ValueError):coefficients(alpha,bad,clean)
        with self.assertRaises(ValueError):loss(Fixed(torch.zeros(2,6,40)),clean,text,mask,torch.tensor([6,4]),t,alpha,1000.,True,False,40,'v',eps)
        with self.assertRaises(ValueError):loss(Fixed(torch.zeros(2,6,40)),clean,text,mask,prefix,t,alpha,0.,True,False,40,'v',eps)
if __name__=='__main__':unittest.main()
