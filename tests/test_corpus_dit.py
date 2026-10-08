import copy,json,tempfile,unittest,torch
from pathlib import Path
from unittest.mock import patch
from iam_tools.corpus_dit_contract import scope,validate_scope,duration_model,duration
from iam_tools.corpus_dit import training_loss,sample,requested_inputs,fit_posterior_whitening
import sys
REPO=Path(__file__).resolve().parents[1]/'third_party/DiffInk'
if not REPO.is_dir():REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from utils.utils import ModelConfig
from model.dit import DiT
from model.diffusion import Diffusion
class CorpusDiTTests(unittest.TestCase):
    def manifest(self):
        records={};train=[]
        for f in range(6):
            for w in ['a','b']:
                for j in range(3):
                    sid=f'f{f}-{w}-{j}';train.append(sid);records[sid]=dict(text=f'family{f} sentence{j}',prompt_family=f'family{f}',writer_id=w,points=300+j*8)
        for w,f in [('dev','devfamily'),('a','heldfamily')]:records[f]=dict(text=f+' unseen text',prompt_family=f,writer_id=w,points=400)
        return dict(records=records,splits=dict(large_train=train,dev=['devfamily'],held_out=['heldfamily']),test_writers=['test'],dev_writers=['dev'])
    def test_metadata_reservation_blocks_whole_families_and_duplicate_texts(self):
        m=self.manifest();history={i:r for i,r in m['records'].items() if r['prompt_family'] in ('family0','family1')};s=scope(m,history);self.assertEqual(len(s['splits']['fresh_confirmation']),16);self.assertEqual(len(s['fresh_reserved_families']),4)
        self.assertTrue(all(m['records'][i]['prompt_family'] not in s['fresh_reserved_families'] for i in s['splits']['train']))
        validate_scope(m['records'],s['splits'],history,m['test_writers'],m['dev_writers'])
        bad=copy.deepcopy(s['splits']);bad['train'].append(bad['fresh_confirmation'][0])
        with self.assertRaises(ValueError):validate_scope(m['records'],bad,history,m['test_writers'],m['dev_writers'])
        # Case and whitespace changes do not evade transcript blocking.
        records=copy.deepcopy(m['records']);records[s['splits']['train'][0]]['text']='  '+records[s['splits']['fresh_confirmation'][0]]['text'].upper()+'  '
        with self.assertRaises(ValueError):validate_scope(records,s['splits'],history,m['test_writers'],m['dev_writers'])
    def test_duration_and_generation_use_no_target_trajectory_length(self):
        m=self.manifest();train=m['splits']['large_train'];d=duration_model(m['records'],train);texts=['family1 sentence1','a long entirely new sentence'];noise,text,mask,length=requested_inputs(texts,list(set(''.join(texts))),d,42,['a','b'],'cpu');self.assertTrue(all(n>=len(t) for t,n in zip(texts,length)));self.assertEqual(tuple(noise.shape),(2,max(length),384));self.assertEqual(int(mask[0].sum()),length[0])
        d2=copy.deepcopy(d);self.assertEqual(duration(d,texts[0]),duration(d2,texts[0]));self.assertEqual(d['train_ids'],train)
    def test_whitening_includes_posterior_variance_all_channels_train_only(self):
        mu=torch.stack((torch.zeros(384),torch.ones(384)*2));lv=torch.zeros_like(mu);s=fit_posterior_whitening([(mu,lv)]);torch.testing.assert_close(s['mean'],torch.ones(384));torch.testing.assert_close(s['std'],torch.full((384,),2**.5));self.assertEqual(s['points'],2)
        with self.assertRaises(ValueError):fit_posterior_whitening([])
    def test_masked_sampled_x0_loss_excludes_padding_and_kept_reference(self):
        cfg=ModelConfig(dict(dim=32,latent_dim=384,num_text_embedding=12,text_dim=16,text_mask_padding=True,conv_layers=1,dim_head=8,depth=2,heads=4,ff_mult=2,dropout=0.,long_skip_connection=False));m=DiT(cfg);clean=torch.randn(2,6,384);mask=torch.arange(6)[None]<torch.tensor([5,4])[:,None];text=torch.tensor([[1,2],[2,3]]);prefix=torch.tensor([2,1]);t=torch.tensor([300,999]);d=Diffusion(device='cpu',schedule_type='cosine')
        with patch.object(d,'noise_images',return_value=(torch.ones_like(clean),torch.ones_like(clean))):r=training_loss(m,clean,text,mask,prefix,t,d,1000.,True,False)
        expected=mask.clone();expected[0,:2]=False;expected[1,:1]=False;self.assertTrue(torch.equal(expected,r['active_mask']));torch.testing.assert_close(r['loss'],clean[expected].square().mean());r['loss'].backward();self.assertTrue(torch.isfinite(m.proj_out.weight.grad).all())
        with patch.object(d,'noise_images',return_value=(torch.ones_like(clean),torch.ones_like(clean))):r=training_loss(m,clean,text,mask,prefix,t,d,1000.,True,True)
        self.assertTrue(torch.equal(r['active_mask'],mask))
    def test_sampler_uses_actual_noise_and_text_cfg_not_reference_only_cfg(self):
        class Fake(torch.nn.Module):
            def forward(self,x,noise,text,time,mask,drop_text,drop_cond):
                self.seen.append((time.clone(),drop_text,drop_cond));return torch.full_like(x,0. if drop_text else .3).masked_fill(~mask[...,None],0.)
        m=Fake().eval();m.seen=[];noise=torch.randn(1,5,384);mask=torch.tensor([[True,True,True,False,False]]);text=torch.tensor([[1,2]]);d=Diffusion(device='cpu',schedule_type='cosine');a=sample(m,noise,text,mask,d.alpha_hat,steps=5,guidance=2.)
        torch.testing.assert_close(a[:,:3],torch.full_like(a[:,:3],.6));self.assertEqual(float(a[:,3:].abs().sum()),0.);self.assertTrue(all(bool((t>=0).all()&(t<1).all()) and cond for t,drop,cond in m.seen));self.assertEqual(sum(drop for t,drop,cond in m.seen),5)
if __name__=='__main__':unittest.main()
