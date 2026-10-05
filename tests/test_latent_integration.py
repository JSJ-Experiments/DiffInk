import copy
import json
import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from iam_tools.latent_integration import geometry_gate,summary,run,DELTA_WEIGHT


def fake_row():
    g=dict(x_rmse=.001,y_rmse=.001,point_distance_p95=.002,point_distance_max=.003,
           first_difference={'relative_rms_error':.05},second_difference={'relative_rms_error':.08})
    for k in ('tangent_angle_error_degrees','turn_angle_error_degrees','target_corner_turn_error_degrees','target_shallow_turn_error_degrees'):
        g[k]=dict(median=2.,p90=5.,p99=10.)
    pen=dict(pen_up_f1=1.,final_eoc_correct=True,non_final_false_eoc_count=0)
    return dict(lines=[dict(sample_id=str(i),mu=dict(kind='mu',geometry=copy.deepcopy(g),pen=copy.deepcopy(pen)),
                            sampled=[dict(kind='z-0',geometry=copy.deepcopy(g),pen=copy.deepcopy(pen))]) for i in range(8)])


class IntegrationGateTests(unittest.TestCase):
    def test_identical_reference_passes(self):
        row=fake_row();self.assertTrue(geometry_gate(row,row)['passed'])
    def test_single_bad_line_not_hidden_by_average(self):
        ref=fake_row();row=copy.deepcopy(ref);row['lines'][0]['mu']['geometry']['turn_angle_error_degrees']['p90']=10.
        self.assertFalse(geometry_gate(row,ref)['passed'])
    def test_single_bad_sampled_pen_not_hidden(self):
        ref=fake_row();row=copy.deepcopy(ref);row['lines'][0]['sampled'][0]['pen']['non_final_false_eoc_count']=1
        self.assertFalse(geometry_gate(row,ref)['passed'])
    def test_sampled_regression_rejected_even_if_mean_perfect(self):
        ref=fake_row();row=copy.deepcopy(ref)
        for l in row['lines']:l['sampled'][0]['geometry']['y_rmse']=.002
        self.assertFalse(geometry_gate(row,ref)['passed'])
    def test_summary_all_draws(self):
        row=fake_row();row['lines'][0]['sampled'].append(copy.deepcopy(row['lines'][0]['sampled'][0]))
        row['lines'][0]['sampled'][-1]['pen']['pen_up_f1']=.8
        self.assertEqual(summary(row)['sampled_min_pen_f1'],.8)
    def test_run_invalid_chunk_before_allocating_gpu(self):
        with self.assertRaises(ValueError):run('invalid','invalid',steps=1001)
        with self.assertRaises(ValueError):run('invalid','invalid',stages=('style',))
        with self.assertRaises(ValueError):run('invalid','invalid',stages=())
    def test_runtime_metadata_safe_checkpoint_roundtrip(self):
        import io
        from iam_tools.latent_integration import runtime_metadata
        metadata=runtime_metadata();self.assertIs(type(metadata['torch_version']),str)
        buffer=io.BytesIO();torch.save(dict(config=metadata),buffer);buffer.seek(0)
        self.assertEqual(torch.load(buffer,weights_only=True)['config'],metadata)

    def test_standard_trainer_rejects_unsupported_mean_anchor(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        from trainer.vae_trainer import train_vae_one_epoch
        with self.assertRaisesRegex(ValueError,'silently drop'):
            train_vae_one_epoch(None,SimpleNamespace(mean_xy_anchor_weight=1000),[],None,None,0,1,'cpu')

    def test_target_difference_scalar_preserved(self):
        self.assertAlmostEqual(DELTA_WEIGHT,.20471838744633777)

class IntegrationForwardTests(unittest.TestCase):
    def test_float_downsample_mask_becomes_boolean_and_real_joint_backward(self):
        from iam_tools.latent_integration import encoded,terms
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Conv1d(5,4,8,stride=8)
                self.conv_mu=torch.nn.Conv1d(4,4,1);self.conv_logvar=torch.nn.Conv1d(4,4,1)
                self.decoder=torch.nn.Conv1d(4,123,1)
            def to_model_space(self,x):return x
            def decode(self,z,padding_mask=None):return self.decoder(z).repeat_interleave(8,dim=2)
            def kl_divergence_new(self,mu,lv,mask):
                return (-.5*(1+lv-mu.square()-lv.exp()))[mask[:,None].expand_as(mu)].mean()
            def get_ocr_loss(self,z,labels,mask):return z.square().mean()
        torch.manual_seed(43);m=Tiny();raw=torch.zeros(1,5,16)
        raw[:,0]=torch.linspace(0,1,16);raw[:,1]=torch.linspace(1,0,16);raw[:,2]=1;raw[:,2,-1]=0;raw[:,4,-1]=1
        mask=torch.ones(1,16,dtype=torch.bool)
        self.assertEqual(encoded(m,raw,mask)[-1].dtype,torch.bool)
        t=terms(m,(raw,mask,torch.tensor([[0,1]])),use_ocr=True,epsilon=torch.full((1,4,2),.01))
        loss=100*(t['sampled_geometry']+t['mean_geometry'])+t['pen']+1e-6*t['kl']+.01*t['ctc']
        loss.backward();self.assertTrue(torch.isfinite(loss))
        for p in m.parameters():self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(m.conv_logvar.weight.grad.abs().sum()),0)


if __name__=='__main__':unittest.main()
