import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch,yaml
root=Path(__file__).resolve().parents[1];repo=root/'third_party/DiffInk' if (root/'third_party/DiffInk').is_dir() else root
sys.path.insert(0,str(repo))
from model.vae import VAE
from model.losses import mixture_expectation
from iam_tools.identity_geometry_probe import initialize_identity_geometry

class IdentityProbeTests(unittest.TestCase):
    def model(self):
        cfg=yaml.safe_load((repo/'configs/engineering_english.yaml').read_text())
        cfg.update(hidden_dims=[16,24,48],latent_dim=48,decoder_dims=[48,32,128],trans_hidden_dim=32,
                   trans_num_layers=1,ocr_hidden_dim=16,ocr_num_heads=4,ocr_num_layers=1,
                   num_text_embedding=5,num_writer=8,style_classifier_dim=48)
        m=VAE(SimpleNamespace(**cfg)).eval();initialize_identity_geometry(m);return m
    def raw(self,n=29,padded=32):
        p=torch.linspace(0,1,n);raw=torch.zeros(1,5,padded)
        raw[0,0,:n]=(p*8-3)*100;raw[0,1,:n]=(p*10).sin()*100
        raw[:,2,:n]=1;raw[:,2,12]=0;raw[:,3,12]=1;raw[:,2,n-1]=0;raw[:,4,n-1:]=1
        return raw,torch.arange(padded)[None,:]<n
    def test_all_eight_polyphase_slots_pack_and_reconstruct(self):
        m=self.model();raw,mask=self.raw()
        with torch.no_grad():
            z,mu,lv=m.encode(raw)
            packed=mu[:,:40].transpose(1,2).reshape(1,32,5).transpose(1,2)
            torch.testing.assert_close(packed,m.to_model_space(raw),atol=0,rtol=0)
            output=m.decode(mu,padding_mask=~mask);xy=mixture_expectation(output)
            torch.testing.assert_close(xy,m.to_model_space(raw)[:,:2].transpose(1,2),atol=1e-5,rtol=1e-5)
            # Nested inference may zero invalid padding. Only real states are
            # reconstructed; this control makes no padded stop-supervision claim.
            torch.testing.assert_close(output[:,:3].argmax(1)[mask],raw[:,2:].argmax(1)[mask],atol=0,rtol=0)
            self.assertTrue((mu[:,40:]==0).all());self.assertTrue((lv[:,40:]==0).all())
    def test_unused_latent_noise_and_extra_padding_do_not_distort_real_geometry(self):
        m=self.model();raw,mask=self.raw();long,longmask=self.raw(padded=64)
        with torch.no_grad():
            _,mu,_=m.encode(raw);a=mixture_expectation(m.decode(mu,padding_mask=~mask))[:,:29]
            noisy=mu.clone();noisy[:,40:]=torch.randn_like(noisy[:,40:])*5
            b=mixture_expectation(m.decode(noisy,padding_mask=~mask))[:,:29]
            _,other,_=m.encode(long);c=mixture_expectation(m.decode(other,padding_mask=~longmask))[:,:29]
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6);torch.testing.assert_close(a,c,atol=1e-6,rtol=1e-6)
    def test_invalid_init_parameters_do_not_silently_mutate_model(self):
        m=self.model();before={k:v.clone() for k,v in m.state_dict().items()}
        for kw in [dict(S=1),dict(S=float('nan')),dict(S=float('inf')),dict(std=0),dict(std=.5)]:
            with self.assertRaises(ValueError):initialize_identity_geometry(m,**kw)
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()))

if __name__=='__main__':unittest.main()
