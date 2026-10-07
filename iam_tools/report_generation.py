"""Packed generation gallery, frozen-codec provenance and independent CPU sampling."""
import hashlib,html,json,collections
from pathlib import Path
import h5py,numpy as np,torch
from .generation_study import ARMS,SOURCE,SHA,READER_SHA,READER_REL,decode_sample,collate,noise_for
from .latent_diffusion import TextLatentDenoiser,transform,cosine_schedule,ddim_sample
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .pen_ab import file_sha


def verify_pairing(directory,config,results):
    logs={a:[json.loads(l) for l in (Path(directory)/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    if any(len(logs[a])!=results[a]['last_step'] for a in ARMS):raise ValueError('incomplete training logs')
    paired=all(x['sample_ids']==y['sample_ids'] and x['noise_sha256']==y['noise_sha256'] for x,y in zip(logs['text'],logs['no_text']))
    full=all(r['last_step']==config['max_updates'] for r in results.values())
    frozen=all(r['codec_reader_unchanged'] for r in results.values())
    same_rng=not full or len({r['final_noise_rng_sha256'] for r in results.values()})==1
    if not paired or not frozen or not same_rng:raise AssertionError('matched noise/codec guards failed')
    for a,r in results.items():
        for name,key in [('checkpoint-best.pt','selected_sha256'),('checkpoint-last.pt','last_sha256')]:
            if file_sha(Path(directory)/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
    return dict(matched_batches_noise=paired,full_budget=full,codec_reader_frozen=frozen,same_final_rng= same_rng)


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    cfg=json.loads((directory/'config.json').read_text());data=json.loads((directory/'dataset.json').read_text());results=json.loads((directory/'result.json').read_text());paired=verify_pairing(directory,cfg,results)
    if file_sha(root/SOURCE)!=SHA or file_sha(root/READER_REL)!=READER_SHA or file_sha(directory/'source.h5')!=cfg['source_h5_sha256'] or file_sha(directory/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('immutable source/reader/cache SHA drift')
    for q in (directory/'source-code').rglob('*.py'):
        rel=q.relative_to(directory/'source-code');current=Path(__file__).with_name(rel.name) if rel.parts[0]=='iam_tools' else Path(repo)/rel
        if q.read_bytes()!=current.read_bytes():raise ValueError('as-run code drift: '+str(rel))
    out=directory/'report';out.mkdir(exist_ok=True);(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    stats=torch.load(directory/'whitening.pt',map_location='cpu',weights_only=True);vocab=cfg['vocab'];records=data['records'];latents={};targets={}
    with h5py.File(directory/'source.h5') as hf:
        for sid in records:latents[sid]=torch.from_numpy(hf[sid]['latent_mean'][:]);targets[sid]=hf[sid]['target'][:]
    # Small independent sampler/decoder reload on CPU, not a new training run.
    codec,_,_,codec_cfg,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,codec_cfg)
    ids=data['splits']['train'][:4]+data['splits']['unseen_prompt'][:4];_,mask,text=collate(latents,records,vocab,ids,stats,'cpu');noise=noise_for(ids,latents,9142,'cpu');checks={}
    for arm,r in results.items():
        model=TextLatentDenoiser(**cfg['model']);saved=torch.load(directory/arm/'checkpoint-last.pt',map_location='cpu',weights_only=True)
        if saved['config']!=dict(cfg,arm=arm) or saved['step']!=r['last_step']:raise ValueError('denoiser checkpoint contract drift')
        model.load_state_dict(saved['model_state_dict']);model.eval().requires_grad_(False)
        # Use actual saved GPU epsilon, NOT a CPU RNG with the same seed.
        with h5py.File(directory/arm/f'evaluation-{r["last_step"]}.h5') as hf:
            for j,sid in enumerate(ids):noise[j,:len(latents[sid])]=torch.from_numpy(hf[f'correct/9142/{sid}/initial_noise'][:])
        z=transform(ddim_sample(model,noise,text,mask,cosine_schedule(),steps=50,drop_text=arm=='no_text'),stats,inverse=True);rows=[]
        with h5py.File(directory/arm/f'evaluation-{r["last_step"]}.h5') as hf,h5py.File(out/f'cpu-reload-{arm}.h5','w') as saved_hf:
            for j,sid in enumerate(ids):
                points,metrics=decode_sample(codec,reader,z[j,:len(latents[sid])],records[sid],vocab);gpu=hf[f'correct/9142/{sid}/points'][:];previous=json.loads(hf[f'correct/9142/{sid}'].attrs['row'])
                rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),cpu_free_decoded=metrics['free_decoded'],gpu_free_decoded=previous['free_decoded'],cpu_window_decoded=metrics['oracle_window_decoded'],gpu_window_decoded=previous['oracle_window_decoded']))
                saved_hf.create_dataset(sid,data=points,compression='gzip')
        checks[arm]=dict(lines=rows,max_xy_difference=max(v['max_xy_difference'] for v in rows),pen_mismatches=sum(v['pen_mismatches'] for v in rows),free_transcript_differences=sum(v['cpu_free_decoded']!=v['gpu_free_decoded'] for v in rows),window_transcript_differences=sum(v['cpu_window_decoded']!=v['gpu_window_decoded'] for v in rows))
        # Diffusion iterates amplify roundoff: REPORT differences, do not hide
        # them or impose the deterministic-codec1e-4 gate on a50step sampler.
        print(dict(independent_cpu_sampler=arm,checks=checks[arm]),flush=True);del model
    duration_counts=collections.Counter((records[i]['points']+7)//8 for i in data['splits']['train'])
    duration_identity=dict(train_distinct_lengths=len(duration_counts),train_singleton_length_lines=sum(v==1 for v in duration_counts.values()),train_total=len(data['splits']['train']),counts=dict(duration_counts),note='Oracle duration can leak training-line identity; same-length collisions, matched no-text control and fixed-duration same-noise text interventions are required interpretation controls.')
    char_train=set(''.join(records[i]['text'] for i in data['splits']['train']));char_unseen=set(''.join(records[i]['text'] for i in data['splits']['unseen_prompt']))
    summaries={};evals={}
    for arm,r in results.items():
        evals[arm]={}
        for e in r['history']:
            row=json.loads((directory/arm/f'eval-{e["step"]}.json').read_text());q=directory/arm/f'evaluation-{e["step"]}.h5'
            if file_sha(q)!=row['packed_h5_sha256']:raise ValueError('generation artifact drift')
            evals[arm][e['step']]=row
        summaries[arm]=r['history'][-1]['groups']
    summary=dict(duration_identity=duration_identity,unseen_characters_not_in_training=sorted(char_unseen-char_train),config=cfg,dataset=data,results=results,paired=paired,independent_cpu_reload=checks,final_groups=summaries,
        limitations='Oracle latent duration; one fixed writer; generator TRAIN32/form-heldout8 from frozenreaderTRAINcorpus. Codec initialized polyphase40 not learned semantic VAE. No reference prefix/target trajectory at sampling. Reader metrics not an independent OCR benchmark. No final promotion; visually inspect, especially truncated free-stop versus oracle window.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');pages=[]
    # Full oracle window displayed AND genuine first-EOC stop; no true pen states.
    versions={a:results[a]['last_step'] for a in ARMS}
    for split,group in data['splits'].items():
        for start in range(0,len(group),4):
            selection=group[start:start+4];fig,axes=plt.subplots(len(selection),5,figsize=(25,3.0*len(selection)),squeeze=False)
            with h5py.File(directory/'text'/f'evaluation-{versions["text"]}.h5') as th,h5py.File(directory/'no_text'/f'evaluation-{versions["no_text"]}.h5') as nh:
                for j,sid in enumerate(selection):
                    ref=targets[sid];conditional=th[f'correct/9142/{sid}'];nullo=th[f'null/9142/{sid}'];unconditional=nh[f'correct/9142/{sid}'];cp=conditional['points'][:];row=json.loads(conditional.attrs['row']);stop=row['generated_points_at_stop']
                    variants=[(ref,'IAM/RDP reference'),(cp[:len(ref)],'text / oracle window'),(cp[:stop],'text / predicted stop'),(nullo['points'][:len(ref)],'SAME model / text dropped'),(unconditional['points'][:len(ref)],'matched NO-TEXT model')]
                    for ax,(q,title) in zip(axes[j],variants):
                        draw(ax,split_xy(q[:,:2],q[:,2:].argmax(1)));ax.set_title(sid+' | '+title+'\n'+records[sid]['text'],fontsize=8)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=135);plt.close(fig);pages.append(name)
    # Ablation galleries include both seeds and swapped text/CFG, not only the
    # visually preferred seed. Four TRAIN and ALL8 unseen prompts.
    probe=data['splits']['train'][:4]+data['splits']['unseen_prompt']
    for seed in (9142,9143):
        for start in range(0,len(probe),4):
            selection=probe[start:start+4];fig,axes=plt.subplots(len(selection),5,figsize=(25,3*len(selection)),squeeze=False)
            with h5py.File(directory/'text'/f'evaluation-{versions["text"]}.h5') as hf:
                for j,sid in enumerate(selection):
                    for ax,policy in zip(axes[j,1:],['correct','null','swapped','correct_cfg3']):
                        q=hf[f'{policy}/{seed}/{sid}'];row=json.loads(q.attrs['row']);points=q['points'][:];draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(policy+' | free reader: '+row['free_decoded']+'\nwindow reader: '+row['oracle_window_decoded'],fontsize=8)
                    draw(axes[j,0],split_xy(targets[sid][:,:2],targets[sid][:,2:].argmax(1)));axes[j,0].set_title(sid+' | reference\n'+records[sid]['text'],fontsize=8)
            fig.tight_layout();name=f'ablations-seed{seed}-{start//4+1}.png';fig.savefig(out/name,dpi=135);plt.close(fig);pages.append(name)
    fig,axes=plt.subplots(1,3,figsize=(18,5))
    for arm,r in results.items():
        h=r['history'];steps=[e['step'] for e in h]
        axes[0].plot(steps,[e['denoising']['999']['xy'] for e in h],'o-',label=arm);axes[1].plot(steps,[e['groups']['train']['correct']['oracle_window_cer'] for e in h],'o-',label=arm);axes[2].plot(steps,[e['groups']['unseen_prompt']['correct']['oracle_window_cer'] for e in h],'o-',label=arm)
    for ax,title in zip(axes,['TRAIN terminal-noise XY MSE','TRAIN generated window CER','UNSEEN generated window CER']):ax.set_title(title);ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    rows=''
    for arm,r in results.items():
        for split,policies in r['history'][-1]['groups'].items():
            for policy,q in policies.items():rows+=f'<tr><td>{arm}</td><td>{split}</td><td>{policy}</td><td>{100*q["free_cer"]:.2f}%</td><td>{100*q["oracle_window_cer"]:.2f}%</td><td>{q["free_exact"]}/{q["evaluations"]}</td><td>{q["missing_eoc"]}</td></tr>'
    body=f'''<!doctype html><meta charset="utf-8"><title>Frozen codec text-to-latent generation gate</title><style>body{{font:16px system-ui;margin:30px auto;max-width:1800px;padding:15px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:7px}}pre{{white-space:pre-wrap}}</style><h1>Generation, not reconstruction: can text control a frozen faithful codec?</h1><p>Small cross-attention x0 diffusion prototype; NOT released InkDiT/paper reproduction. Same initialized weights, minibatches/noise, TRAIN32 and unseen whole-form8, writer10160. Full384-channel latent means whitened from TRAIN only. Frozen codec/reader. Sampling starts from pure noise; no trajectory/reference prefix. Oracle latent length remains a known limitation.</p><p><a href="summary.json">All configs/metrics/provenance/CPU reload</a> · <a href="../config.json">Exact as-run configuration</a> · <a href="../dataset.json">Split/source codec preflight</a></p><p>Reader is trained on the broader IAM corpus containing these samples, but generator has never seen the8held-out prompts or their form. This is a same-writer, reader-familiar small-data diagnostic—not independent text/style generalization certification. Aligned point error only measures TRAIN memorization, NOT whether another plausible unseen handwriting trajectory is valid.</p><h2>Generated reading, both genuine first-EOC stop and oracle window</h2><table><tr><th>Model</th><th>Split</th><th>Condition</th><th>Free-stop CER</th><th>Oracle-window CER</th><th>Free exact</th><th>No EOC</th></tr>{rows}</table><img src="learning.png"><h2>All40 reference/generated comparisons</h2><p>Marker-free, PREDICTED pen states; not true boundaries. Reference/oracle-window/predicted-stop views separated. All models see oracle latent length, not target coordinates. No automatic final promotion.</p>'''
    for name in pages:body+=f'<h3>{name}</h3><img loading="lazy" src="{name}">'
    body+='<h2>Independent CPU sampler reload</h2><pre>'+html.escape(json.dumps(checks,indent=2))+'</pre><h2>Oracle-length identity leakage</h2><pre>'+html.escape(json.dumps(duration_identity,indent=2))+'</pre><h2>Pairing</h2><pre>'+html.escape(json.dumps(paired,indent=2))+'</pre>'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),paired=paired,groups=summaries,cpu=checks)
