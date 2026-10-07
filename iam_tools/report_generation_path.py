"""All40native comparisons, full readout matrix, paired guards and CPU reload."""
import json,html
from pathlib import Path
import h5py,numpy as np,torch
from .generation_path import ARMS,NATIVE,generate
from .generation_path_study import PARENT,PARENT_SHA,DATA_PARENT
from .generation_study import collate,decode_sample
from .latent_diffusion import TextLatentDenoiser,cosine_schedule,transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .kl_tradeoff import SOURCE,SHA
from .pen_ab import file_sha
from .report_resources import report as resource_report


def verify_run(directory,config,results):
    arms=config.get('arms',ARMS);directory=Path(directory);logs={a:[json.loads(l) for l in (directory/a/'metrics.jsonl').read_text().splitlines()] for a in arms}
    if any(len(logs[a])!=results[a]['last_step'] for a in arms):raise ValueError('incomplete logs')
    if len({r['last_step'] for r in results.values()})!=1:raise ValueError('different arm budgets; inspect separately instead of claiming paired comparison')
    for a in arms[1:]:
        if any(x['sample_ids']!=y['sample_ids'] or x.get('noise_rng_sha256')!=y.get('noise_rng_sha256') for x,y in zip(logs[arms[0]],logs[a])):raise ValueError('paired schedule/RNG mismatch')
    if len({r['final_noise_rng_sha256'] for r in results.values()})!=1 or not all(r['codec_reader_unchanged'] for r in results.values()):raise ValueError('frozen/RNG guard failed')
    for a,r in results.items():
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(directory/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
    return dict(matched_batches_rng=True,codec_reader_frozen=True,equal_budgets=True,full_budget=all(r['last_step']==config['max_updates'] for r in results.values()))


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);data_parent=root/DATA_PARENT;cfg=json.loads((directory/'config.json').read_text());data=json.loads((data_parent/'dataset.json').read_text());results=json.loads((directory/'result.json').read_text());guards=verify_run(directory,cfg,results);arms=cfg['arms'];readouts=cfg['native_readouts'];source_mode=cfg.get('source_readout','ddim50')
    if file_sha(root/cfg['parent'])!=cfg['parent_sha256'] or file_sha(data_parent/'source.h5')!=cfg['source_h5_sha256'] or file_sha(data_parent/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('immutable parent/data drift')
    for q in (directory/'source-code').rglob('*.py'):
        rel=q.relative_to(directory/'source-code');current=Path(repo)/rel if rel.parts[0]=='model' else Path(__file__).with_name(q.name)
        if q.read_bytes()!=current.read_bytes():raise ValueError('as-run code drift: '+str(rel))
    out=directory/'report';out.mkdir(exist_ok=True);resource_report(directory);(out/'index.html').replace(out/'resources.html');(out/'report-source.py').replace(out/'resource-report-source.py');(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    stats=torch.load(data_parent/'whitening.pt',weights_only=True);targets={};latents={};evaluations={};files={}
    with h5py.File(data_parent/'source.h5') as f:
        for sid in data['records']:targets[sid]=f[sid]['target'][:];latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    probe=json.loads((directory/'source_probe/eval-0.json').read_text());files['source']=directory/'source_probe/evaluation-0.h5';evaluations['source']=probe
    for a,r in results.items():evaluations[a]=json.loads((directory/a/f'eval-{r["last_step"]}.json').read_text());files[a]=directory/a/f'evaluation-{r["last_step"]}.h5'
    for a,e in evaluations.items():
        if file_sha(files[a])!=e['packed_h5_sha256']:raise ValueError('packed trajectory SHA drift')
    # Validate same initial epsilon for allreadouts/arms, not just RNGstate.
    same_noise=True
    with h5py.File(files['source']) as p:
        for arm in arms:
            with h5py.File(files[arm]) as f:
                for sid in sorted(data['records']):
                    for seed in (9142,9143):
                        same_noise &= np.array_equal(p[f'{source_mode}_correct/{seed}/{sid}/initial_noise'][:],f[f'{readouts[arm]}_correct/{seed}/{sid}/initial_noise'][:])
    if not same_noise:raise ValueError('sampler noise pairing failed')
    # IndependentCPUreload native predictions2TRAIN/2held from actualGPUepsilon.
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc)
    ids=data['splits']['train'][:2]+data['splits']['unseen_prompt'][:2];_,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');checks={}
    for arm,r in results.items():
        model=TextLatentDenoiser(**cfg['model']);model.load_state_dict(torch.load(directory/arm/'checkpoint-last.pt',map_location='cpu',weights_only=False)['model_state_dict']);model.eval();epsilon=torch.zeros(len(ids),mask.shape[1],384);rows=[]
        with h5py.File(files[arm]) as f:
            for j,sid in enumerate(ids):epsilon[j,:len(latents[sid])]=torch.tensor(f[f'{readouts[arm]}_correct/9142/{sid}/initial_noise'][:])
            z=transform(generate(model,epsilon,text,mask,cosine_schedule(),readouts[arm]),stats,True)
            for j,sid in enumerate(ids):
                points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=f[f'{readouts[arm]}_correct/9142/{sid}'];gpu=q['points'][:];old=json.loads(q.attrs['row'])
                rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),free_transcript_equal=m['free_decoded']==old['free_decoded']))
        checks[arm]=rows
    # Zero-inputseed invariance: deterministic duplicate observations reported.
    direct_invariant=True
    with h5py.File(files[next(a for a in arms if readouts[a]=='zero')]) as f:
        for sid in data['records']:
            direct_invariant &= np.array_equal(f[f'zero_correct/9142/{sid}/points'][:],f[f'zero_correct/9143/{sid}/points'][:])
    pages=[]
    for split,ids in data['splits'].items():
        for seed in ([9142] if split=='train' else [9142,9143]):
            for start in range(0,len(ids),4):
                group=ids[start:start+4];fig,axs=plt.subplots(len(group),2+len(arms),figsize=(5*(2+len(arms)),3*len(group)),squeeze=False)
                for j,sid in enumerate(group):
                    draw(axs[j,0],split_xy(targets[sid][:,:2],targets[sid][:,2:].argmax(1)));axs[j,0].set_title(sid+' | IAM/RDP\n'+data['records'][sid]['text'],fontsize=8)
                    for ax,(label,mode) in zip(axs[j,1:],[('source',source_mode)]+[(a,readouts[a]) for a in arms]):
                        with h5py.File(files[label]) as f:q=f[f'{mode}_correct/{seed}/{sid}'];p=q['points'][:];row=json.loads(q.attrs['row'])
                        p=p[:row['generated_points_at_stop']];draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_title(label+' / '+mode+'\nreader: '+row['free_decoded'],fontsize=8)
                fig.tight_layout();name=f'{split}-seed{seed}-{start//4+1}.png';fig.savefig(out/name,dpi=135);plt.close(fig);pages.append(name)
    # Source sampler readouts directly juxtaposed on fixed first4TRAIN/first4held.
    for split in (('train','unseen_prompt') if 'ddim100_correct' in probe['aggregate']['train'] else ()):
        ids=data['splits'][split][:4];fig,axs=plt.subplots(4,5,figsize=(25,12))
        with h5py.File(files['source']) as f:
            for j,sid in enumerate(ids):
                draw(axs[j,0],split_xy(targets[sid][:,:2],targets[sid][:,2:].argmax(1)));axs[j,0].set_title(sid+' | reference',fontsize=8)
                for ax,mode in zip(axs[j,1:],['zero','terminal','ddim2','ddim100']):
                    q=f[f'{mode}_correct/9142/{sid}'];p=q['points'][:];row=json.loads(q.attrs['row']);p=p[:row['generated_points_at_stop']];draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_title('source '+mode+'\nreader: '+row['free_decoded'],fontsize=8)
        fig.tight_layout();name='source-readout-'+split+'.png';fig.savefig(out/name,dpi=135);plt.close(fig);pages.append(name)
    summary=dict(config=cfg,results=results,guards=guards,matched_sampling_epsilon=same_noise,direct_seed_invariant=direct_invariant,source_probe=probe['aggregate'],final_matrix={a:evaluations[a]['aggregate'] for a in arms},cpu_native_reload=checks,
                 limitations='32TRAIN/8held,singlewriter,oracle duration,generatorunseenformbutreaderfamiliarcorpus; no paper/productionpromotion. Directzero two seeds duplicate deterministic observations. TRAINpairedgeometry measures memorization; heldtrajectory not unique valid writing. Indexdifferences not physicalvelocity/curvature. Angle stats are mean per-line p90s.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Text mapping versus denoising and sampling</title><style>body{font:16px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style><h1>Controlled conditional-generation diagnosis</h1><p>Matched source model+Adam+RNG,32TRAIN/8whole-formheld,LR1e-4. Fullyfrozenfaithfulcodec/reader,unchangedarchitectureandXY/targetsegmentanchors. See exactconfig below for the isolated intervention and eacharm native readout. No target/reference trajectory at sampling; oracle duration remains. No generic smoothing,KL,OCR/styletraining or productionpromotion.</p><p><a href="summary.json">Full metrics/config/CPUreload/provenance</a> · <a href="../config.json">As-run config</a> · <a href="resources.html">Resource telemetry</a></p>'
    body+='<h2>Exact intervention</h2><pre>'+html.escape(json.dumps({k:cfg[k] for k in ('profile','parent','parent_step','max_updates','arms','native_readouts','intervention','matching','training_inputs','text_drop_by_arm','null_evaluation_note') if k in cfg},indent=2))+'</pre>'
    def table(matrix,title,only_native=False):
        b='<h2>'+title+'</h2><table><tr><th>Model</th><th>Split</th><th>Readout/text</th><th>FreeCER</th><th>Exact</th><th>X/YRMSE</th><th>segment / seconddiff</th><th>tangent/turnp90mean°</th><th>minpenF1</th></tr>'
        for label,splits in matrix.items():
            for split,policies in splits.items():
                for policy,r in policies.items():
                    if only_native and policy!=(readouts.get(label,source_mode)+'_correct'):continue
                    b+=f'<tr><td>{label}</td><td>{split}</td><td>{policy}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.5f} / {r["y_rmse"]:.5f}</td><td>{r["segment_vector_rmse"]:.5f} / {r["second_difference_vector_rmse"]:.5f}</td><td>{r["tangent_p90_mean"]:.1f} / {r["turn_p90_mean"]:.1f}</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
        return b+'</table>'
    body+=table({'source':probe['aggregate'],**summary['final_matrix']},'Native correct-text comparison',True)
    body+='<p>Predictedpenstates and genuinefirstEOCstop used in ALLrenders,not true boundaries. Directzero seed labels repeat deterministic predictions; no claim of64independentTRAINdraws. IndependentCPUsampler/native decoder reloaded4linesperarm.</p>'
    for name in pages:body+='<h2>'+name+'</h2><img loading="lazy" src="'+name+'">'
    body+=table({'source':probe['aggregate']},'No-training source readout sweep')+table(summary['final_matrix'],'Full trained readout/text matrix')
    body+='<h2>CPU reload and guards</h2><pre>'+html.escape(json.dumps(dict(cpu=checks,guards=guards,noise_pairing=same_noise,direct_seed_invariant=direct_invariant),indent=2))+'</pre>'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),native={a:evaluations[a]['aggregate']['train'][readouts[a]+'_correct'] for a in arms},cpu=checks)
