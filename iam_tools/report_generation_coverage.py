"""Marker-free all73 coverage comparisons; independent frozen-codec CPU reload."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .generation_coverage import WriterTextDenoiser
from .generation_coverage_study import PARENT_SHA,writer_tensor
from .generation_study import collate,decode_sample
from .latent_diffusion import transform
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .pen_ab import file_sha
from .report_resources import report as resources_report


def verify(directory,cfg,results,data):
    p=Path(directory);trains=data['training_ids'];records=data['records'];held=data['splits']['unseen_prompt'];forms={records[i]['prompt_family'] for i in held}
    from .ocr_pool import normalized_text
    texts={normalized_text(records[i]['text']) for i in held}
    if cfg['parent_sha256']!=PARENT_SHA or len(results)!=3:raise ValueError('pinnedthree-armparent guard')
    for a,r in results.items():
        if r['last_step']!=cfg['max_updates'] or not r['codec_reader_unchanged']:raise ValueError('incomplete budget/frozen guard')
        rows=[json.loads(s) for s in (p/a/'metrics.jsonl').read_text().splitlines()]
        if len(rows)!=r['last_step'] or any(not set(row['sample_ids'])<=set(trains[a]) for row in rows):raise ValueError('training scope/log guard')
        if any(records[i]['prompt_family'] in forms or normalized_text(records[i]['text']) in texts for i in trains[a]):raise ValueError('heldform/transcript leakage')
        for name,key in [('checkpoint-last.pt','last_sha256'),('checkpoint-best.pt','selected_sha256')]:
            if file_sha(p/a/name)!=r[key]:raise ValueError('checkpoint SHA drift')
    if not set(trains['small32'])<=set(trains['writer_all'])<=set(trains['broad1024']):raise ValueError('nestedtraining guard')
    return dict(full_budgets=True,codec_reader_frozen=True,actual_train_scope=True,global_held_forms_and_texts_excluded=True,nested_training=True)


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);cfg=json.loads((directory/'config.json').read_text());data=json.loads((directory/'dataset.json').read_text());results=json.loads((directory/'result.json').read_text());guards=verify(directory,cfg,results,data)
    if file_sha(root/cfg['parent'])!=cfg['parent_sha256'] or file_sha(directory/'source.h5')!=cfg['source_h5_sha256'] or file_sha(directory/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('parentdata drift')
    for p in (directory/'source-code').rglob('*.py'):
        rel=p.relative_to(directory/'source-code');q=Path(repo)/rel
        if p.read_bytes()!=q.read_bytes():raise ValueError('as-run code drift:'+str(rel))
    out=directory/'report'
    if out.exists():raise ValueError('refuse overwrite completed report')
    resources_report(directory);(out/'index.html').replace(out/'resources.html');(out/'report-source.py').replace(out/'resource-report-source.py');(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    latents={};targets={};evaluations={}
    with h5py.File(directory/'source.h5') as f:
        for sid in data['records']:latents[sid]=torch.tensor(f[sid]['latent_mean'][:]);targets[sid]=f[sid]['target'][:]
    for a,r in results.items():
        e=json.loads((directory/a/f'eval-{r["last_step"]}.json').read_text());assert file_sha(directory/a/f'evaluation-{r["last_step"]}.h5')==e['packed_h5_sha256'];evaluations[a]=e
    # Identical zero-init source outputs across all arms, before different data gradients.
    initial_equal=True;optimizer_groups={}
    for a in results:
        saved=torch.load(directory/a/'checkpoint-initial.pt',map_location='cpu',weights_only=False)
        optimizer_groups[a]=[{k:g[k] for k in ('lr','betas','weight_decay','eps')} for g in saved['optimizer_state_dict']['param_groups']]
    if any(g!=optimizer_groups['small32'] for g in optimizer_groups.values()):raise ValueError('initialoptimizerpolicy mismatch')
    with h5py.File(directory/'small32/evaluation-0.h5') as f:
        for a in ('writer_all','broad1024'):
            with h5py.File(directory/a/'evaluation-0.h5') as g:
                for sid in f['correct']:initial_equal &= np.array_equal(f['correct/'+sid+'/points'][:],g['correct/'+sid+'/points'][:])
    if not initial_equal:raise ValueError('initial trajectory pairing failed')
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,cfg['source_rel'],cfg['source_sha256'],writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc);stats=torch.load(directory/'whitening.pt',weights_only=True);checks={};attention={}
    ids=data['splits']['retained_train'][:2]+data['splits']['unseen_prompt'][:2];clean,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');wi=writer_tensor(ids,data['records'],cfg['writers'],'cpu')
    for a,r in results.items():
        model=WriterTextDenoiser(**cfg['model']);model.load_state_dict(torch.load(directory/a/'checkpoint-last.pt',map_location='cpu',weights_only=False)['model_state_dict']);model.eval()
        with torch.no_grad():z=transform(model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi),stats,True)
        checks[a]=[]
        from .generation_attention import capture,describe
        observed,weights=capture(model,torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi)
        with torch.no_grad():reference=model(torch.zeros_like(clean),torch.ones(len(ids)),text,mask,writer_ids=wi)
        attention[a]=dict(kernel_max_standardized_latent_drift=float((observed-reference).abs().max()),lines={})
        for j,sid in enumerate(ids):
            layers={name:describe(w[j],len(latents[sid]),len(data['records'][sid]['text'])) for name,w in weights.items()}
            attention[a]['lines'][sid]=layers
            fig,axs=plt.subplots(1,len(layers),figsize=(20,5),squeeze=False)
            for ax,(name,description) in zip(axs[0],layers.items()):
                ax.imshow(description['mean_weights'],origin='lower',aspect='auto',cmap='magma');ax.set_title(name,fontsize=9);ax.set_xlabel('character token index');ax.set_ylabel('latent point-block index')
            fig.suptitle(a+' | '+sid+' | '+data['records'][sid]['text'],fontsize=9);fig.tight_layout();fig.savefig(out/('attention-'+a+'-'+sid+'.png'),dpi=110);plt.close(fig)
        with h5py.File(directory/a/f'evaluation-{r["last_step"]}.h5') as f:
            for j,sid in enumerate(ids):
                points,m=decode_sample(codec,reader,z[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=f['correct/'+sid];old=json.loads(q.attrs['row']);gpu=q['points'][:]
                checks[a].append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),transcript_equal=m['free_decoded']==old['free_decoded']))
    pages=[]
    for split,ids in data['splits'].items():
        for start in range(0,len(ids),4):
            group=ids[start:start+4];fig,axs=plt.subplots(len(group),4,figsize=(24,3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                draw(axs[j,0],split_xy(targets[sid][:,:2],targets[sid][:,2:].argmax(1)));axs[j,0].set_title(sid+' | reference\n'+data['records'][sid]['text'],fontsize=9)
                for ax,a in zip(axs[j,1:],('small32','writer_all','broad1024')):
                    with h5py.File(directory/a/f'evaluation-{results[a]["last_step"]}.h5') as f:
                        q=f['correct/'+sid];row=json.loads(q.attrs['row']);points=q['points'][:row['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(a+' / '+('TRAIN' if sid in data['training_ids'][a] else 'untrained')+'\nreader: '+row['free_decoded'],fontsize=9)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append(name)
    summary=dict(config=cfg,guards=guards,initial_trajectories_bitwise_identical=initial_equal,actual_initial_optimizer_groups=optimizer_groups,results=results,final_matrix={a:e['aggregate'] for a,e in evaluations.items()},cpu_reload=checks,preflight=data['preflight'],attention=attention,limitations=cfg['caveats'])
    (out/'attention-source.py').write_bytes(Path(__file__).with_name('generation_attention.py').read_bytes())
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Nested text coverage and writer conditioning</title><style>body{font:16px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style><h1>Text coverage:32→41samewriter→1024/32writers</h1><p>Frozen faithful polyphase40 transport, not semantic VAE or released InkDiT. Same no-drop parent/model+Adam, zero-initialized writer bias in ALLarms. Deterministic zero input, oracle duration, no trajectory/reference at inference. Different datasets mean different minibatches; broad arm changes textcoverage AND writerdiversity. Eight held-form prompts and exact normalized texts excluded globally. Reader is corpus-familiar; not an independent benchmark.</p><p><a href="summary.json">Full metrics/provenance</a> · <a href="../config.json">As-run config</a> · <a href="resources.html">Telemetry</a></p><h2>Final matched update budget</h2><table><tr><th>Arm</th><th>Split / condition</th><th>CER</th><th>Exact</th><th>X/YRMSE</th><th>segment/second-index-diff</th><th>minpenF1</th></tr>'
    for a,e in evaluations.items():
        for split,policies in e['aggregate'].items():
            for policy,r in policies.items():
                body+=f'<tr><td>{a}</td><td>{split} / {policy}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["x_rmse"]:.5f}/{r["y_rmse"]:.5f}</td><td>{r["segment_vector_rmse"]:.5f}/{r["second_difference_vector_rmse"]:.5f}</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
    body+='</table><p>Added samewriter/otherwriter lines are untrained for arms where they are absent; do not average those into TRAIN quality. Matching a held reference pointwise is not necessary for valid handwriting; inspect readability too. All renders use predicted pens and genuinefirstEOC stopping, no markers or smoothing. Index differences are NOT physical velocity/curvature.</p>'
    for name in pages:body+='<h2>'+name+'</h2><img loading="lazy" src="'+name+'">'
    body+='<h2>Descriptive cross-attention (not actual character alignment)</h2><p>First two retained/twoheld lines, all arms/layers. This observer changes the attention kernel only during inspection; numerical drift reported. Attention maps do not prove causal alignment failure.</p>'
    for a,rows in attention.items():
        for sid in rows['lines']:body+='<h3>'+a+' '+sid+'</h3><img loading="lazy" src="attention-'+a+'-'+sid+'.png">'
    body+='<h2>CPU reload and frozen/split guards</h2><pre>'+html.escape(json.dumps(dict(cpu=checks,guards=guards,initial_equal=initial_equal),indent=2))+'</pre>'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),metrics=summary['final_matrix'],cpu_reload=checks)
