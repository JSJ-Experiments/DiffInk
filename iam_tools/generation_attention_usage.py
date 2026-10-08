"""CPU-only frozen TRAIN last-layer usage audit, never held prompt tuning."""
import json
from pathlib import Path
import h5py,numpy as np,torch
from .attention_usage import MODES,last_layer_intervention
from .generation_prefix_contract import PrefixContractWriter
from .generation_cache import CachedLatentPool
from .generation_capacity import DATA
from .generation_coverage_study import writer_tensor
from .generation_study import decode_sample
from .latent_diffusion import transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .ocr_context_study import tensor_digest
from .pen_ab import file_sha

FIXED=['k04-309z-09','c03-109z-03','r07-568z-04','a02-130z-03','d08-586z-01','p10-249z-01','e07-425z-02','n05-514z-02']


@torch.no_grad()
def audit(directory,repo,root='data'):
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());ids=data['training_ids']['control'];records=data['records']
    if ids!=data['training_ids']['weak_alignment'] or not set(FIXED)<=set(ids):raise ValueError('actual sharedTRAIN-only audit required')
    torch.set_num_threads(2);out=p/'attention-usage';out.mkdir(exist_ok=False)
    for name in ['attention_usage.py','generation_attention_usage.py','weak_alignment.py']:(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    latents={};targets={}
    with h5py.File(root/DATA/'source.h5') as f:
        for s in ids:latents[s]=torch.tensor(f[s]['latent_mean'][:]);targets[s]=f[s]['target'][:]
    stats=torch.load(root/DATA/'whitening.pt',weights_only=True);pool=CachedLatentPool(latents,records,cfg['vocab'],ids,stats,device='cpu')
    codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None)
    codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);cd=tensor_digest(codec.state_dict());rd=tensor_digest(reader.state_dict());results={}
    with h5py.File(out/'outputs.h5','w') as dest:
        for arm in ['control','weak_alignment']:
            r=json.loads((p/arm/'result.json').read_text());cp=p/arm/'checkpoint-best.pt'
            if file_sha(cp)!=r['selected_sha256']:raise ValueError('frozen TRAIN-selected checkpoint guard')
            saved=torch.load(cp,map_location='cpu',weights_only=False);m=PrefixContractWriter(**cfg['models'][arm]).eval();m.load_state_dict(saved['model_state_dict']);md=tensor_digest(m.state_dict());rng=torch.get_rng_state().clone();rows=[];mode_rows={}
            for start in range(0,len(ids),8):
                batch=ids[start:start+8];clean,mask,text,_=pool.select(batch);wi=writer_tensor(batch,records,cfg['writers'],'cpu');args=(torch.zeros_like(clean),torch.ones(len(batch)),text,mask)
                base=m(*args,writer_ids=wi);rawbase=transform(base,stats,True)[...,:40].reshape(len(batch),-1,5)
                for mode in ('baseline',)+MODES:
                    if mode=='baseline':pred=base
                    else:
                        with last_layer_intervention(m,text,mode):pred=m(*args,writer_ids=wi)
                    z=transform(pred,stats,True);raw=z[...,:40].reshape(len(batch),-1,5)
                    for j,s in enumerate(batch):
                        n=records[s]['points'];delta=(raw[j,:n,:2]-rawbase[j,:n,:2]).numpy()
                        row=dict(sample_id=s,mode=mode,x_drift=float(np.sqrt((delta[:,0]**2).mean())),y_drift=float(np.sqrt((delta[:,1]**2).mean())),max_latent_drift=float((pred[j,:len(latents[s])]-base[j,:len(latents[s])]).abs().max()),raw_pen_changes=int((raw[j,:n,2:].argmax(-1)!=rawbase[j,:n,2:].argmax(-1)).sum()))
                        g=dest.create_group(arm+'/'+mode+'/'+s);g.create_dataset('latent',data=z[j,:len(latents[s])].numpy(),compression='gzip')
                        if s in FIXED:
                            points,metrics=decode_sample(codec,reader,z[j,:len(latents[s])],records[s],cfg['vocab']);row['decoded_fixed_train']=metrics;g.create_dataset('points',data=points,compression='gzip')
                        g.attrs['row']=json.dumps(row);rows.append(row);mode_rows.setdefault(mode,[]).append(row)
            if tensor_digest(m.state_dict())!=md or not torch.equal(rng,torch.get_rng_state()):raise ValueError('frozen audit model/RNG mutation')
            if max(r['max_latent_drift'] for r in mode_rows['manual_control'])>1e-4:raise ValueError('manual control numerical mismatch')
            aggregate={}
            for mode,rs in mode_rows.items():
                q=[r['decoded_fixed_train'] for r in rs if 'decoded_fixed_train' in r];aggregate[mode]=dict(train_drift_lines=len(rs),mean_x_drift=float(np.mean([r['x_drift'] for r in rs])),mean_y_drift=float(np.mean([r['y_drift'] for r in rs])),raw_pen_changes=sum(r['raw_pen_changes'] for r in rs),max_latent_drift=max(r['max_latent_drift'] for r in rs),fixed_train_reader_lines=len(q),fixed_train_cer=sum(r['free_errors'] for r in q)/sum(r['characters'] for r in q),fixed_train_exact=sum(r['free_errors']==0 for r in q))
            results[arm]=dict(checkpoint_sha256=file_sha(cp),selected_step=saved['step'],state_rng_unchanged=True,aggregate=aggregate,lines=rows);print(dict(arm=arm,aggregate=aggregate),flush=True)
    if tensor_digest(codec.state_dict())!=cd or tensor_digest(reader.state_dict())!=rd:raise ValueError('frozen reader/codec drift')
    s=dict(arms=results,fixed_train_ids=FIXED,packed_h5_sha256=file_sha(out/'outputs.h5'),definitions='All256TRAIN: physical polyphase40 latent-field XY/pen drift, NOT an opaque generic VAE metric. Actual frozen-codec decode/free-stop reader on8fixedTRAIN only. Last-layer value interventions leave queries/keys/all earlier layers/full text unchanged. Character embedding rotation within each real transcript, BOS and positional values unchanged. Zero contexts preserve out-projection bias. Manual same-weight control independently verifies arithmetic.',limitations='TRAIN-only frozen mechanism probe, not generalization scores or proof of exact glyph boundaries. All previous layers still see correct text; a small last-layer effect does not imply the whole model ignores conditioning. Counterfactual ablations are not production output. No updates, candidate tuning, held prompt scoring, latent sampling, KL/style/CTC or smoothing.')
    (out/'summary.json').write_text(json.dumps(s,indent=2)+'\n');render(out,targets,records,s);return str(out/'index.html')


def render(out,targets,records,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    pages=[]
    with h5py.File(out/'outputs.h5') as f:
        for arm in s['arms']:
            for start in range(0,len(FIXED),4):
                ids=FIXED[start:start+4];modes=['baseline','local_token_rotate','global_token_rotate','local_zero','global_zero','all_zero'];fig,axs=plt.subplots(len(ids),7,figsize=(42,3*len(ids)),squeeze=False)
                for j,sid in enumerate(ids):
                    true=targets[sid];draw(axs[j,0],split_xy(true[:,:2],true[:,2:].argmax(-1)));axs[j,0].set_title(sid+' / source\n'+records[sid]['text'],fontsize=8)
                    for ax,mode in zip(axs[j,1:],modes):
                        g=f[arm+'/'+mode+'/'+sid];r=json.loads(g.attrs['row']);points=g['points'][:r['decoded_fixed_train']['generated_points_at_stop']];draw(ax,split_xy(points[:,:2],points[:,2:].argmax(-1)));ax.set_title(arm+'/'+mode+'\nreader: '+r['decoded_fixed_train']['free_decoded'],fontsize=8)
                fig.tight_layout();name=f'{arm}-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append(name)
    body='<!doctype html><meta charset="utf-8"><title>Frozen last-layer character-value usage</title><style>body{font:17px system-ui;max-width:2400px;margin:30px auto}img{width:100%}th,td{border:1px solid #aaa;padding:8px}</style><h1>Does better weak timing actually carry character information?</h1><p>'+s['definitions']+'</p><p>'+s['limitations']+'</p><a href="summary.json">All256 drift rows and actual8TRAIN decoded-reader metrics, hashes</a><table><tr><th>Arm/mode</th><th>All256 mean X/Y drift</th><th>All256 raw pen changes</th><th>8fixedTRAIN CER</th></tr>'
    for a,r in s['arms'].items():
        for mode,q in r['aggregate'].items():body+=f'<tr><td>{a}/{mode}</td><td>{q["mean_x_drift"]:.5f}/{q["mean_y_drift"]:.5f}</td><td>{q["raw_pen_changes"]}</td><td>{100*q["fixed_train_cer"]:.2f}%</td></tr>'
    body+='</table>'
    for name in pages:body+=f'<img loading="lazy" src="{name}">'
    (out/'index.html').write_text(body)
