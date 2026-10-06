import unittest
import torch
from iam_tools.conditioning import ChannelNorm1d,install_channel_norm,condition_batches,restore_xy

class ConditioningTests(unittest.TestCase):
    def test_channel_norm_is_independent_of_padding_and_matches_layernorm(self):
        torch.manual_seed(7);x=torch.randn(2,6,11);layer=ChannelNorm1d(6)
        expected=torch.nn.functional.layer_norm(x.transpose(1,2),(6,)).transpose(1,2)
        torch.testing.assert_close(layer(x),expected)
        padded=torch.cat([x,torch.randn(2,6,9)*100],-1)
        torch.testing.assert_close(layer(padded)[:,:,:11],layer(x))
        self.assertGreater(float((torch.nn.GroupNorm(1,6)(padded)[:,:,:11]-torch.nn.GroupNorm(1,6)(x)).abs().max().detach()),.1)
    def test_replacement_preserves_state_keys_affine_and_gradients(self):
        model=torch.nn.Sequential(torch.nn.GroupNorm(1,5),torch.nn.Conv1d(5,5,1),torch.nn.GroupNorm(1,5))
        before={k:v.clone() for k,v in model.state_dict().items()}
        self.assertEqual(install_channel_norm(model),2);self.assertEqual(model.state_dict().keys(),before.keys())
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))
        model(torch.randn(1,5,8)).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(install_channel_norm(model),0)
    def test_center_roundtrip_preserves_states_padding_and_differences(self):
        raw=torch.zeros(1,5,8);raw[0,:2,:5]=torch.tensor([[0,2,3,4,6],[0,1,0,-1,0.]])
        raw[:,2,:4]=1;raw[:,4,4:]=1;mask=torch.arange(8)[None,:]<5;labels=torch.tensor([[1]])
        before=raw.clone();batches,offsets=condition_batches({'x':(raw,mask,labels)},'center')
        centered=batches['x'][0];restored=restore_xy(centered[0,:2,:5].T*.01,offsets['x'])
        torch.testing.assert_close(restored,raw[0,:2,:5].T*.01)
        torch.testing.assert_close(centered[0,:2,1:5]-centered[0,:2,:4],raw[0,:2,1:5]-raw[0,:2,:4])
        self.assertTrue(torch.equal(raw,before));self.assertTrue(torch.equal(centered[:,2:],raw[:,2:]));self.assertTrue(torch.equal(centered[:,:,5:],raw[:,:,5:]))
    def test_edge_padding_never_changes_real_points_or_pen_states(self):
        raw=torch.randn(1,5,8);mask=torch.arange(8)[None,:]<5;labels=torch.zeros(1,1)
        batch,offset=condition_batches({'x':(raw,mask,labels)},'edge_pad');x=batch['x'][0]
        torch.testing.assert_close(x[:,:,:5],raw[:,:,:5]);torch.testing.assert_close(x[:,2:],raw[:,2:])
        torch.testing.assert_close(x[0,:2,5:],raw[0,:2,4,None].expand(2,3))
        self.assertTrue((offset['x']==0).all())
    def test_invalid_inputs_fail(self):
        with self.assertRaises(ValueError):condition_batches({},'unknown')
        with self.assertRaises(ValueError):ChannelNorm1d(4)(torch.randn(1,3,8))
        raw=torch.zeros(1,5,8);mask=torch.tensor([[True,False,True,False,False,False,False,False]])
        with self.assertRaises(ValueError):condition_batches({'x':(raw,mask,torch.zeros(1,1))},'center')

class FullSetJointTests(unittest.TestCase):
    def test_joint_loss_backpropagates_geometry_and_pen_not_logvar(self):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        from iam_tools.fullset_joint import joint_loss,run,PEN_WEIGHT,DELTA_WEIGHT
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Conv1d(5,4,8,stride=8)
                self.conv_mu=torch.nn.Conv1d(4,4,1);self.conv_logvar=torch.nn.Conv1d(4,4,1)
                self.fc=torch.nn.Linear(4,123)
            def to_model_space(self,x):return x
            def decode(self,z,padding_mask=None):return self.fc(z.repeat_interleave(8,dim=-1).transpose(1,2)).transpose(1,2)
        torch.manual_seed(9);m=Tiny();raw=torch.zeros(1,5,16);raw[:,0]=torch.linspace(0,1,16);raw[:,2]=1
        raw[:,2,7]=0;raw[:,3,7]=1;raw[:,2,-1]=0;raw[:,4,-1]=1
        mask=torch.ones(1,16,dtype=torch.bool)
        t=joint_loss(m,(raw,mask,torch.zeros(1,1)));(t['point']+DELTA_WEIGHT*t['delta']+PEN_WEIGHT*t['pen']).backward()
        self.assertTrue(torch.isfinite(m.encoder.weight.grad).all());self.assertIsNone(m.conv_logvar.weight.grad)
        self.assertGreater(float(m.fc.weight.grad[:3].abs().sum()),0);self.assertGreater(float(m.fc.weight.grad[3:63].abs().sum()),0)
        self.assertTrue((m.fc.weight.grad[63:]==0).all())
        with self.assertRaises(ValueError):run('missing','missing',steps=301)


class ResearchContractTests(unittest.TestCase):
    def test_standard_trainer_rejects_research_contract_before_model_use(self):
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        from trainer.vae_trainer import train_vae_one_epoch
        with self.assertRaisesRegex(ValueError,'recorded study runner'):
            train_vae_one_epoch(None,SimpleNamespace(research_contract='channel normalization'),[],None,None,0,1,'cpu')

    def test_standard_checkpoint_loader_rejects_incompatible_conditioning(self):
        import sys
        from pathlib import Path
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        from model.vae import VAE
        dummy=SimpleNamespace(config=SimpleNamespace(language='en',trans_dropout=0.),
            transformer_decoder=SimpleNamespace(transformer=SimpleNamespace(layers=[])))
        for mode in ('center','channel','edge_pad'):
            checkpoint=dict(config=dict(conditioning_mode=mode,model_input_scale=.01))
            with self.assertRaisesRegex(ValueError,'dedicated loader'):
                VAE.apply_checkpoint_contract(dummy,checkpoint)
            VAE.apply_checkpoint_contract(dummy,checkpoint,allow_research_conditioning=True)
        VAE.apply_checkpoint_contract(dummy,dict(config=dict(conditioning_mode='control')))
