"""Immutable composition review: matched finals, sealed confirmation, full galleries."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .generation_alignment import AlignedWriterDenoiser
from .generation_alignment_study import ARMS
from .generation_alignment_review import verify_sources
from .generation_capacity import DATA
from .generation_confirmation import confirm
from .generation_study import collate,decode_sample
from .generation_coverage_study import writer_tensor
from .generation_attention import capture,describe
from .latent_diffusion import transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .pen_ab import file_sha


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());results=json.loads((p/'result.json').read_text())
    if (p/'report').exists() or (p/'confirmation').exists():raise ValueError('refuse overwrite completed or partial review/confirmation')
    guards,pool_path=verify_sources(p,repo,root,cfg,data,results)
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);reader.eval().requires_grad_(False)
    source=root/DATA;stats=torch.load(source/'whitening.pt',weights_only=True);targets={};latents={}
    with h5py.File(source/'source.h5') as f:
        for sid in data['records']:targets[sid]=f[sid]['target'][:];latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    confirmation=confirm(p,cfg,data,results,pool_path,codec,reader,stats)
    from .report_resources import report as resource_report
    resource_report(p);out=p/'report';(out/'index.html').replace(out/'resources.html');(out/'report-source.py').replace(out/'resource-report-source.py')
    for name in ('report_generation_alignment.py','generation_alignment_review.py'):(out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    ids=data['splits']['retained_train'][:2]+data['splits']['new_train128'][:2]+data['splits']['expansion256'][:2]+data['splits']['unseen_prompt'][:2]
    clean,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');wi=writer_tensor(ids,data['records'],cfg['writers'],'cpu');checks={};attention={};evaluations={};duration={}
    with h5py.File(out/'attention.h5','w') as af:
        for a in ARMS:
            model=AlignedWriterDenoiser(**cfg['models'][a]);saved=torch.load(p/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval();step=results[a]['last_step']
            evaluations[a]=json.loads((p/a/f'eval-{step}.json').read_text());duration[a]=json.loads((p/a/f'duration-eval-{step}.json').read_text())
            with torch.no_grad():pred=model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi);z=transform(pred,stats,True)
            checks[a]=[]
            with h5py.File(p/a/f'evaluation-{step}.h5') as f:
                for j,sid in enumerate(ids):
                    points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);g=f['correct/'+sid];old=json.loads(g.attrs['row']);gpu=g['points'][:]
                    checks[a].append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),transcript_equal=m['free_decoded']==old['free_decoded']))
            attention[a]={}
            for sid in ids[::2]:
                c,m,t=collate(latents,data['records'],cfg['vocab'],[sid],stats,'cpu');w=writer_tensor([sid],data['records'],cfg['writers'],'cpu')
                with torch.no_grad():normal=model(torch.zeros_like(c),torch.ones(1),t,m,writer_ids=w);observed,weights=capture(model,torch.zeros_like(c),torch.ones(1),t,m,writer_ids=w)
                rows={name:describe(value[0],len(latents[sid]),len(data['records'][sid]['text'])) for name,value in weights.items()}
                attention[a][sid]=dict(observer_output_max_difference=float((normal-observed).abs().max()),layers=rows)
                for name,v in weights.items():af.create_dataset(a+'/'+sid+'/'+name,data=v,compression='gzip')
                final_name=list(weights)[-1];v=weights[final_name][0,:,:len(data['records'][sid]['text'])+1];fig,axs=plt.subplots(1,4,figsize=(16,5),squeeze=False)
                for h,ax in enumerate(axs[0]):ax.imshow(v[h,:,1:],aspect='auto',origin='lower',interpolation='nearest');ax.set_title(f'{a} head{h}');ax.set_xlabel('text token index');ax.set_ylabel('ink block index')
                fig.suptitle(sid+' / '+data['records'][sid]['text'],fontsize=9);fig.tight_layout();fig.savefig(out/f'attention-{a}-{sid}.png',dpi=110);plt.close(fig)
    # Fresh outputs equal across arms; continuations must equal their own parent.
    if cfg.get('continuation_of'):
        parent=root/cfg['continuation_of']
        for a in ARMS:
            with h5py.File(p/a/'evaluation-0.h5') as f,h5py.File(parent/a/f'evaluation-{cfg["parent_step"]}.h5') as g:
                if set(f['correct'])!=set(g['correct']) or any(not np.array_equal(f['correct/'+i+'/points'][:],g['correct/'+i+'/points'][:]) for i in f['correct']):raise ValueError('own-parent step0 output equality guard')
        guards['own_parent_initial_decoded_outputs_bitwise_identical']=True
    else:
        with h5py.File(p/'global/evaluation-0.h5') as f,h5py.File(p/'soft_gaussian/evaluation-0.h5') as g:
            if set(f['correct'])!=set(g['correct']) or any(not np.array_equal(f['correct/'+i+'/points'][:],g['correct/'+i+'/points'][:]) for i in f['correct']):raise ValueError('identical step0 prediction guard')
        guards['initial_decoded_outputs_bitwise_identical']=True
    fig,axs=plt.subplots(1,3,figsize=(18,5))
    original_results=json.loads((root/cfg['continuation_of']/'result.json').read_text()) if cfg.get('continuation_of') else None
    for a,r in results.items():
        offset=cfg['parent_step'];history=[dict(h,step=offset+h['step']) for h in r['history']];dh=[dict(h,step=offset+h['step']) for h in r['duration_history']]
        if original_results:history=original_results[a]['history'][:-1]+history;dh=original_results[a]['duration_history'][:-1]+dh
        xs=[h['step'] for h in history]
        axs[0].plot(xs,[100*h['aggregate']['all_train256']['correct']['free_cer'] for h in history],label=a)
        axs[1].plot(xs,[100*h['aggregate']['unseen_prompt']['correct']['free_cer'] for h in history],label=a)
        axs[2].plot([h['step'] for h in dh],[100*h['aggregate']['estimated_correct']['free_cer'] for h in dh],label=a)
    for ax,title in zip(axs,['TRAIN256 / oracle duration','Development8 / oracle duration','Development8 / predicted duration']):ax.set_title(title);ax.set_xlabel('total updates from fresh initialization');ax.set_ylabel('CER %');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning-curves.png',dpi=120);plt.close(fig)
    pages=[]
    def gallery(label,records,ids,columns,reference=None):
        for start in range(0,len(ids),4):
            group=ids[start:start+4];offset=int(reference is not None);fig,axs=plt.subplots(len(group),len(columns)+offset,figsize=(8*(len(columns)+offset),3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                if reference is not None:
                    truth=reference[sid];draw(axs[j,0],split_xy(truth[:,:2],truth[:,2:].argmax(1)));axs[j,0].set_title(sid+' / reference\n'+records[sid]['text'],fontsize=8)
                for ax,(title,path,policy) in zip(axs[j,offset:],columns):
                    with h5py.File(path) as f:g=f[policy+'/'+sid];row=json.loads(g.attrs['row']);points=g['points'][:row['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(title+'\n'+records[sid]['text']+'\nreader: '+row['free_decoded'],fontsize=8)
            fig.tight_layout();name=f'{label}-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append((label,name))
    finals=[(a+' / matched final',p/a/f'evaluation-{results[a]["last_step"]}.h5','correct') for a in ARMS]
    for split in ('retained_train','added_same_writer','new_train128','expansion256','unseen_prompt'):gallery(split,data['records'],data['splits'][split],finals,targets)
    dev_cols=[(a+' / predicted duration',p/a/f'duration-evaluation-{results[a]["last_step"]}.h5','estimated_correct') for a in ARMS]
    gallery('development-predicted-duration',data['records'],data['splits']['unseen_prompt'],dev_cols,targets)
    seal=data['confirmation_seal'];ct={}
    with h5py.File(p/'confirmation/source.h5') as f:
        for sid in seal['ids']:ct[sid]=f[sid]['target'][:]
    columns=[]
    for a in ARMS:
        step=results[a]['best_step'];folder=p/'confirmation'/a
        columns.extend([(a+' / selected oracle',folder/f'evaluation-{step}.h5','correct'),(a+' / selected predicted duration',folder/f'duration-evaluation-{step}.h5','estimated_correct')])
    gallery('sealed-paired',seal['records'],seal['ids'],columns,ct)
    synth=data['synthetic_seal']['records'];columns=[(a+' / selected predicted duration',p/'confirmation'/a/'synthetic'/f'duration-evaluation-{results[a]["best_step"]}.h5','estimated_correct') for a in ARMS]
    gallery('sealed-synthetic',synth,list(synth),columns)
    summary=dict(config=cfg,guards=guards,results=results,original_results=original_results,matched_final_oracle={a:e['aggregate'] for a,e in evaluations.items()},matched_final_estimated={a:e['aggregate'] for a,e in duration.items()},confirmation=confirmation,cpu_reload=checks,attention=attention,attention_sha256=file_sha(out/'attention.h5'),preflight=data['preflight'],limitations=cfg['caveats'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'attention.json').write_text(json.dumps(attention,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Text composition alignment control</title><style>body{font:16px system-ui;max-width:2600px;margin:30px auto;padding:20px}img{width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap}</style><h1>Fresh global vs soft-Gaussian text alignment</h1><p>Standalone deterministic mapper with frozen polyphase transport codec and corpus-familiar reader. NOT released InkDiT, semantic InkVAE, or a TrInk reproduction. Original study: identical fresh weights, empty Adam, RNG,256TRAIN,32writers,24000 updates, batch8 and LR schedule. Any convergence continuation is labeled below. No CTC/style/KL. Same fixed XY and target-difference anchors. Intervention bundles finite Gaussian attention bias (3/4 heads, one global) + trainable positional-encoding amplitudes. Single initialization; not a causal isolation of each component.</p><p>Durations at inference are either explicit oracle targets or estimated solely from requested text/writer using a TRAIN-only ridge fit. No forced final EOC. Swapped/NULL controls keep duration fixed and are scored against ORIGINAL request, not the swapped string.</p><p><a href="summary.json">Metrics/provenance</a> · <a href="attention.json">Descriptive attention diagnostics</a> · <a href="resources.html">Resources</a></p>'
    if cfg.get('continuation_of'):body+='<p><strong>Convergence continuation:</strong> both arms restored their own completed fresh24000-update model/full Adam/RNG, then trained24000 more at constant1e-5. Total48000updates each. Triggered solely by TRAIN CER>10%, before opening confirmation. Original fresh pairing verified separately; continuation initial weights differ legitimately. Galleries/tables below use continuation finals.</p>'
    body+='<h2>Matched-final oracle durations (not different best checkpoints)</h2><table><tr><th>Arm</th><th>Split / policy</th><th>CER</th><th>Exact</th><th>X/Y RMSE</th><th>segment error</th></tr>'
    for a,e in evaluations.items():
        for split in ('all_train256','unseen_prompt'):
            for policy,r in e['aggregate'][split].items():body+=f'<tr><td>{a}</td><td>{split}/{policy}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.5f}/{r["y_rmse"]:.5f}</td><td>{r["segment_vector_rmse"]:.5f}</td></tr>'
    body+='</table><h2>Development predicted-duration controls</h2>'
    def duration_table(arms):
        s='<table><tr><th>Arm / policy</th><th>CER</th><th>Exact</th><th>Missing EOC</th><th>Duration relative error</th></tr>'
        for a,groups in arms.items():
            for policy,r in groups.items():s+=f'<tr><td>{a}/{policy}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["missing_eoc"]}</td><td>{r["mean_absolute_relative_length_error"]}</td></tr>'
        return s+'</table>'
    body+=duration_table({a:e['aggregate'] for a,e in duration.items()})
    body+='<h2>TRAIN-selected checkpoints: familiar text without oracle duration</h2>'+duration_table({a:r['train_estimated'] for a,r in confirmation['arms'].items()})+'<p>This target-free duration check on all256 TRAIN texts is a shortcut diagnostic, not a held benchmark. Models were trained with oracle lengths; changing that at inference can cause distribution shift as well as remove an identity cue.</p>'
    body+='<h2>Sealed confirmation, TRAIN-selected frozen checkpoints</h2><p>16 paired IAM transcripts/8known writers and16 genuinely new unpaired synthetic sentences. No training/selection on these outputs. Paired IDs/transcripts are outside all1032 previous packed generation examples; forms excluded from CURRENT fresh TRAIN/dev, but other lines from these forms occurred in historical1024 runs. Reader may have seen corpus trajectories. Reader failures are NOT excluded. Checkpoint steps: '+html.escape(str({a:r['best_step'] for a,r in results.items()}))+'. Paired source reader ceiling: '+html.escape(str(confirmation['preflight']))+'.</p><h3>Paired / oracle duration</h3><table><tr><th>Arm/policy</th><th>CER</th><th>Exact</th></tr>'
    for a,r in confirmation['arms'].items():
        for policy,q in r['oracle']['confirmation'].items():body+=f'<tr><td>{a}/{policy}</td><td>{100*q["free_cer"]:.2f}%</td><td>{q["free_exact"]}/{q["evaluations"]}</td></tr>'
    body+='</table><h3>Paired / predicted duration</h3>'+duration_table({a:r['estimated'] for a,r in confirmation['arms'].items()})+'<h3>Synthetic / predicted duration ONLY</h3><p>No reference trajectory exists, so no oracle duration or aligned RMSE is invented. Reader CER is a proxy; inspect actual writing.</p>'+duration_table({a:r['synthetic'] for a,r in confirmation['arms'].items()})
    body+='<h2>Learning curves</h2><img src="learning-curves.png"><h2>Marker-free galleries</h2><p>All264 current lines and all sealed prompts; predicted pens and first-EOC stopping. No smoothing, point markers or postprocessing. Nonuniform index differences are NOT physical velocities. RMSE versus a held IAM line is only a paired diagnostic, not a necessary criterion for valid alternative handwriting.</p>'
    for label,name in pages:body+='<h3>'+html.escape(label)+'</h3><img loading="lazy" src="'+name+'">'
    body+='<h2>Attention observations (not ground-truth alignment)</h2><p>Last layer, all4heads. need_weights switches attention kernel; its output drift is measured, not used in training/selection.</p>'
    for a in ARMS:
        for sid in attention[a]:body+='<img loading="lazy" src="attention-'+a+'-'+sid+'.png">'
    body+='<h2>Independent CPU reload and protocol guards</h2><pre>'+html.escape(json.dumps(dict(cpu_reload=checks,guards=guards),indent=2))+'</pre>'
    (out/'index.html').write_text(body)
    return dict(report=str(out/'index.html'),matched_final={a:e['aggregate']['unseen_prompt']['correct']['free_cer'] for a,e in evaluations.items()},confirmation=confirmation['arms'])
