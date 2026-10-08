import unittest,torch
from unittest.mock import patch
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1]/'third_party/DiffInk'
if not REPO.is_dir():REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from model.dit import DiT,TextEmbedding,InputEmbedding
from model.modules import TimestepEmbedding,precompute_freqs_cis,Attention,AttnProcessor
from x_transformers.x_transformers import RotaryEmbedding,apply_rotary_pos_emb
from utils.utils import ModelConfig
class DiTEnglishContractTests(unittest.TestCase):
    def cfg(self):return ModelConfig(dict(dim=32,latent_dim=40,num_text_embedding=12,text_dim=16,text_mask_padding=True,conv_layers=1,dim_head=8,depth=2,heads=4,ff_mult=2,dropout=0.,long_skip_connection=False))
    def test_integer_timesteps_do_not_quantize_sinusoidal_features(self):
        torch.manual_seed(21);m=TimestepEmbedding(32);t=torch.tensor([1,100,500,999]);torch.testing.assert_close(m(t),m(t.float()),rtol=0,atol=0)
        self.assertGreater(float((m(t)[0]-m(t)[-1]).abs().max().detach()),.01)
    def test_constructor_is_device_agnostic_and_optional_audio_not_needed(self):
        m=DiT(self.cfg());self.assertTrue(all(p.device.type=='cpu' for p in m.parameters()));self.assertEqual(precompute_freqs_cis(16,10).device.type,'cpu')
        x=torch.randn(2,9,40);text=torch.tensor([[1,2,3,-1],[1,3,4,5]]);out=m(x,x,text,torch.tensor([10,200]),mask=None);self.assertEqual(tuple(out.shape),(2,9,40));self.assertTrue(torch.isfinite(out).all())
    def test_positional_convolution_cannot_see_padding_noise_or_projection_bias(self):
        torch.manual_seed(23);m=InputEmbedding(40,16,32).eval();x=torch.randn(1,8,40);t=torch.randn(1,8,16);mask=torch.ones(1,8,dtype=torch.bool)
        a=m(x,x,t,mask=mask);xp=torch.cat((x,torch.randn(1,8,40)*100),1);tp=torch.cat((t,torch.randn(1,8,16)*100),1);mp=torch.arange(16)[None]<8
        b=m(xp,xp,tp,mask=mp);torch.testing.assert_close(a,b[:,:8],atol=1e-6,rtol=1e-6);self.assertEqual(float(b[:,8:].abs().sum().detach()),0.)
    def test_full_backbone_padding_and_train_backward_nontrivial_readout(self):
        torch.manual_seed(24);m=DiT(self.cfg()).eval()
        # Zero-initialized output alone would make a bogus padding test pass.
        with torch.no_grad():m.proj_out.weight.normal_(std=.1);m.proj_out.bias.fill_(.1)
        for block in m.transformer_blocks:
            with torch.no_grad():block.attn_norm.linear.weight.normal_(std=.1);block.attn_norm.linear.bias.normal_(std=.1)
        x=torch.randn(1,8,40);t=torch.tensor([[1,2,3]]);mask=torch.ones(1,8,dtype=torch.bool);time=torch.tensor([100.])
        a=m(x,x,t,time,mask);xp=torch.cat((x,torch.randn(1,8,40)*100),1);mp=torch.arange(16)[None]<8
        b=m(xp,xp,t,time,mp);torch.testing.assert_close(a,b[:,:8],atol=1e-5,rtol=1e-5);self.assertEqual(float(b[:,8:].abs().sum().detach()),0.)
        m.train();m(x,x,t,time,mask).square().mean().backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
    def test_rotary_frequencies_apply_to_every_head_not_concatenated_prefix(self):
        torch.manual_seed(27);att=Attention(AttnProcessor(),dim=32,heads=4,dim_head=8,dropout=0.);x=torch.randn(1,5,32);rope=RotaryEmbedding(8).forward_from_seq_len(5);seen={}
        def capture(q,k,v,**kw):seen.update(q=q,k=k);return torch.zeros_like(v)
        with patch('model.modules.F.scaled_dot_product_attention',side_effect=capture):att(x,rope=rope)
        expected=apply_rotary_pos_emb(att.to_q(x).view(1,5,4,8).transpose(1,2),rope[0],rope[1])
        torch.testing.assert_close(seen['q'],expected,rtol=0,atol=0)
        raw=att.to_q(x).view(1,5,4,8).transpose(1,2)
        for head in range(4):self.assertGreater(float((seen['q'][0,head,1:]-raw[0,head,1:]).abs().max().detach()),.01)
if __name__=='__main__':unittest.main()
