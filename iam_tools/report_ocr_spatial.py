"""Independent reload, verbatim error audit and marker-free frozen-codec gallery."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .ocr_spatial import PARENT,PARENT_SHA,PARENT_UPDATES,ARMS,validate_parent
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_convergence import POOL_SHA
from .ocr_frame_study import frame_cache,paired_posterior_sampler
from .ocr_spatial_features import make_head
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .pen_ab import file_sha
from .report_pointer import publish_pointer
from .ocr_error_analysis import analyze
from .report_ocr_pool_expansion import baseline_equivalence

FIXED=('profile','torch_version','cfg','parent_rel','parent_sha256','parent_updates','source_rel','source_sha256','pool_rel','pool_manifest_sha256','points_per_frame','feature_mode','feature_stats','feature_calibration_ids','train_ids','dev_ids','held_out_ids','common_train_probe','posterior_evaluation_ids','posterior_draws','physical_batch','encoder_physical_batch','attention_radius','dropout','blank_bias','base_lr','betas','weight_decay','clip','seed','schedule_seed','schedule_skip','schedule_sha256','spatial_definition','objective','max_updates','max_wall_seconds','eval_every','train_latent','initial_state','entire_codec_frozen','selection','checkpoint_contract')


def pairing(configs,results,logs):
    if set(configs)!=set(ARMS) or set(results)!=set(ARMS) or set(logs)!=set(ARMS):raise ValueError('complete local/spatial paired study required')
    a,b=ARMS;c,d=configs[a],configs[b];r,s=results[a],results[b]
    for field in FIXED:
        if c[field]!=d[field]:raise ValueError('uncontrolled paired change: '+field)
    if c['spatial_features'] is not False or d['spatial_features'] is not True:raise ValueError('local-only/spatial interventions required')
    if c['objective']!='0.5clean+0.5clean;two identical forwards in BOTH arms':raise ValueError('paired two-forward objectives required')
    checks=dict(same_parent_moments_rng=r['initial_state']==s['initial_state'],same_batches=[v['sample_ids'] for v in logs[a]]==[v['sample_ids'] for v in logs[b]],same_lr=[v['lr'] for v in logs[a]]==[v['lr'] for v in logs[b]],same_schedule=r['sample_schedule_sha256']==s['sample_schedule_sha256'],same_final_dropout_rng=r['final_dropout_rng_sha256']==s['final_dropout_rng_sha256'],equal_completed_budget=all(v['last_step']==c['max_updates'] and v['stop_reason']=='budget_completed' for v in results.values()),entire_codec_frozen=all(v['entire_codec_bitwise_unchanged'] for v in results.values()))
    if not all(checks.values()):raise ValueError('paired budget/RNG/exposure/state guard failed')
    if len(logs[a])!=c['max_updates'] or any(v['step']!=i+1 for i,v in enumerate(logs[a])) or any(v['step']!=i+1 for i,v in enumerate(logs[b])):raise ValueError('complete ordered optimizer logs required')
    return checks


def line_changes(left,right,ids):
    a={r['sample_id']:r for r in left['lines']};b={r['sample_id']:r for r in right['lines']}
    rows=[dict(sample_id=i,parent_errors=a[i]['mu']['errors'],new_errors=b[i]['mu']['errors']) for i in ids]
    return dict(improved=sum(r['new_errors']<r['parent_errors'] for r in rows),tied=sum(r['new_errors']==r['parent_errors'] for r in rows),worsened=sum(r['new_errors']>r['parent_errors'] for r in rows),lines=rows)


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    from .ctc_beam import audit_head
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());configs={a:json.loads((directory/a/'config.json').read_text()) for a in results};logs={a:[json.loads(l) for l in (directory/a/'metrics.jsonl').read_text().splitlines()] for a in results}
    paired=pairing(configs,results,logs);pool,m,vocab=load_pool(root,POOL_SHA)
    if file_sha(root/PARENT)!=PARENT_SHA or (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise ValueError('parent/pool fingerprint drift')
    parent=torch.load(root/PARENT,map_location='cpu',weights_only=True);prev=validate_parent(parent,m);cfg=prev['cfg'];ids=prev['posterior_evaluation_ids']
    for c in configs.values():
        for k in ('cfg','feature_stats','train_ids','dev_ids','held_out_ids','feature_calibration_ids','common_train_probe','posterior_evaluation_ids','points_per_frame','feature_mode'):
            if c[k]!=prev[k]:raise ValueError('parent feature/data drift: '+k)
        if c['source_sha256']!=SHA or c['parent_sha256']!=PARENT_SHA:raise ValueError('pinned source drift')
    base,_,_,loaded,alphabet,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if cfg!=loaded or alphabet!=vocab:raise ValueError('codec/vocab drift')
    base.eval().requires_grad_(False);digest=tensor_digest(base.state_dict());subset=dict(m,records={i:m['records'][i] for i in ids})
    original,texts,audit=cache_corpus(base,pool,subset,vocab,directory,device='cpu',save_geometry=False);cache=frame_cache(original,4);sampler=paired_posterior_sampler(original,cache,4)
    splits=dict(train=prev['common_train_probe'],dev=prev['dev_ids'],held_out=prev['held_out_ids'],common_train_probe=prev['common_train_probe']);cpu={};checks={};maps={};errors={};beam={};gpu={};baseline={}
    for name in ('parent8000',*ARMS):
        folder=directory/name;folder.mkdir(exist_ok=True);saved=parent if name=='parent8000' else torch.load(folder/'head-best.pt',map_location='cpu',weights_only=True)
        c=prev if name=='parent8000' else configs[name]
        if name!='parent8000':
            r=results[name]
            if saved['config']!=c or saved['updates']!=PARENT_UPDATES+r['best_step'] or file_sha(folder/'head-best.pt')!=r['selected_sha256']:raise ValueError('selected checkpoint drift')
        head=make_head(cfg,len(vocab)+1,prev['feature_stats'],spatial=False if name=='parent8000' else c['spatial_features'],seed=42);head.load_state_dict(saved['ocr_state_dict']);head.eval()
        row=evaluate(head,cache,texts,splits,vocab,folder,'cpu-reload',ids,posterior_sampler=sampler);cpu[name]=row;maps[name]={r['sample_id']:r for r in row['lines']}
        errors[name]={g:analyze(row['lines'],group_ids,m['records']) for g,group_ids in splits.items()}
        beam[name]=audit_head(head,cache,texts,dict(dev=prev['dev_ids'],held_out=prev['held_out_ids']),vocab,width=10)
        (folder/'beam-mean-only.json').write_text(json.dumps(beam[name],indent=2)+'\n')
        if name=='parent8000':
            initial_eval=json.loads((directory/ARMS[0]/'ocr-0.json').read_text());gm={r['sample_id']:r for r in initial_eval['lines']}
            diff=[dict(sample_id=r['sample_id'],cpu=r['mu']['decoded'],gpu=gm[r['sample_id']]['mu']['decoded']) for r in row['lines'] if r['mu']['decoded']!=gm[r['sample_id']]['mu']['decoded']]
            checks[name]=dict(mean_cpu_gpu_transcript_differences=diff,cpu_scope='192eval/probe only, parent initial GPU baseline',cpu_groups=row['groups'],posterior_device_rng_unpaired=True)
        if name!='parent8000':
            original_eval=json.loads((folder/f'ocr-{results[name]["best_step"]}.json').read_text());gpu[name]=original_eval;gm={r['sample_id']:r for r in original_eval['lines']}
            diff=[dict(sample_id=r['sample_id'],cpu=r['mu']['decoded'],gpu=gm[r['sample_id']]['mu']['decoded']) for r in row['lines'] if r['mu']['decoded']!=gm[r['sample_id']]['mu']['decoded']]
            checks[name]=dict(mean_cpu_gpu_transcript_differences=diff,cpu_scope='192eval/probe only, NOT8192TRAIN',cpu_groups=row['groups'],posterior_device_rng_unpaired=True)
            first=json.loads((folder/'ocr-0.json').read_text());baseline[name]=baseline_equivalence(json.loads((directory/ARMS[0]/'ocr-0.json').read_text()),first)
    if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA:raise AssertionError('CPU codec mutation')
    preflight=json.loads((directory/'codec-preflight.json').read_text());rows=preflight['lines']
    if not preflight['passed'] or len(rows)!=len(m['records']):raise ValueError('incomplete8352 GPU codec gate')
    geometry=dict(lines=len(rows),mean_per_line_x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in rows])),mean_per_line_y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in rows])),mean_per_line_turn_p90=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] or 0 for r in rows])),maximum_packed_xy_difference=preflight['maximum_packed_xy_difference'],all_pen_boundaries_and_final_eoc_perfect=True)
    summary=dict(results=results,configs=configs,parent_rel=PARENT,parent_sha256=PARENT_SHA,source_sha256=SHA,pool_manifest_sha256=POOL_SHA,paired=paired,initial_baseline_equivalence=baseline,codec_gpu_preflight=geometry,codec_cpu_eval192=audit,cpu_reload=checks,cpu_errors=errors,beam_mean_only=beam,
      selected_vs_parent={a:{g:line_changes(cpu['parent8000'],cpu[a],prev[k]) for g,k in [('dev','dev_ids'),('held_out','held_out_ids')]} for a in ARMS},spatial_vs_local={g:line_changes(cpu[ARMS[0]],cpu[ARMS[1]],prev[k]) for g,k in [('dev','dev_ids'),('held_out','held_out_ids')]},
      caveats='Same reused128DEV/fivewriters and32report; codec prior reportpromptoverlap, not independentIAM/paper semanticVAE. SpatialX readout-only feature. Same common unused-column zeroing, optimizer/counters and two-forward dropout. Coordinates/order/masks/labels unchanged. CPU posterior RNG differs fromGPU; beam width10 mean-only noLM neverselects. No drawing-quality change or jointVAE/CTC/KL/style/InkDiT update.')
    out=directory/'report';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for a,r in results.items():
        for ax,g in zip(axes,('train','dev','held_out')):
            ax.plot([h['step'] for h in r['history']],[100*h['groups'][g]['mu']['cer'] for h in r['history']],label=a,marker='.');ax.axvline(r['best_step'],alpha=.2);ax.set_title(g+'CER');ax.set_xlabel('additionalupdates');ax.set_ylabel('%');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig);pages=[]
    with h5py.File(pool/'lines.h5') as target_file,h5py.File(directory/'geometry-source.h5') as fixed_file:
        for group,k in [('held_out','held_out_ids'),('dev','dev_ids')]:
            for page,page_ids in enumerate(paginate_ids(prev[k],limit=8),1):
                fig,axes=plt.subplots(len(page_ids),2,figsize=(15,2.8*len(page_ids)),squeeze=False)
                for j,sid in enumerate(page_ids):
                    target=target_file[sid]['point_seq'][:];target[:,:2]*=.01;fixed=fixed_file[sid]['mean'][:];lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                    caption='SharedFROZEN reconstruction\n'+'\n'.join(a+': '+maps[a][sid]['mu']['decoded'] for a in ('parent8000',*ARMS))
                    for ax,p,title in [(axes[j,0],target,'IAM/RDP target: '+texts[sid]),(axes[j,1],fixed,caption)]:
                        draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
                fig.tight_layout();f=f'{group}-{page}.png';fig.savefig(out/f,dpi=130);plt.close(fig);pages.append((group,page,f))
    esc=html.escape;chunks=['<meta charset="utf-8"><title>Four-point OCR spatial position</title><style>body{font-family:sans-serif;max-width:1450px;margin:2em auto}img{max-width:100%}td,th{padding:.5em}pre{white-space:pre-wrap}</style><h1>Frozen handwriting: OCR-only spatialX cue</h1>',
      '<h2>What to look for</h2><p>The drawing is unchanged. Target left, shared frozen reconstruction right; compare parent/local/spatial OCR captions. Four normalized horizontal position features are appended ONLY in the spatial reader. No changed coordinates, handwriting, codec, labels or time ordering. Authentic corners/hooks are not smoothed.</p>',
      '<p>Same4-point parent8000 and AdamWmoments/RNG/LR1e-4, same subsequent4000batch updates. Both do two clean forwards. A local-only; B appends4spatialX phases in columns20:24. Common zeroing ofpreviously-unused projectioncolumns preservesparentfunction; momentsexactly0/counterspreserved. Position=(x-minRealX)/max(realXspan,.01), masks/syntheticphases preserved. No labels/alignment/augmentation/rotation/interpolation/time-reindex or newparameters. CleanTRAIN192 calibration unchanged. DEV only selects includingparent0;report32 neverselects. Whole8352codecmean/pen gate unchanged, CPU192 selected reload and20posterior draws, not CPU8192 or all-posterior trajectory certification.</p>',
      '<p>Research initializedpolyphase transport, NOT ordinarysemanticVAE or paperreproduction; fiveDEVwriters/repeateduse and codecpriorreportpromptoverlap limit claims. Beam10 noLM/lexicon is separate mean-only diagnostic, NEVER selects/tunes a head.</p><p><a href="summary.json">Exact metrics/error breakdown/provenance</a> · <a href="../pool-manifest.json">Pinned dataset</a></p>',
      '<table border="1"><tr><th>head/additionalstep</th><th>TRAIN CER</th><th>DEV mean/posterior CER</th><th>report mean/posterior CER</th></tr>']
    first=results[ARMS[0]]['history'][0]['groups']
    for name in ('parent8000',*ARMS):
        v=first if name=='parent8000' else results[name]['selected'];step=0 if name=='parent8000' else results[name]['best_step']
        chunks.append(f'<tr><td>{name}/{step}</td><td>{v["train"]["mu"]["cer"]:.4%}</td><td>{v["dev"]["mu"]["cer"]:.4%}/{v["dev"]["sampled"]["cer"]:.4%}</td><td>{v["held_out"]["mu"]["cer"]:.4%}/{v["held_out"]["sampled"]["cer"]:.4%}</td></tr>')
    chunks.append('</table><img src="learning.png"><h2>Paired controls/geometry/reload</h2><pre>'+esc(json.dumps(dict(paired=paired,geometry=geometry,reload={a:len(v['mean_cpu_gpu_transcript_differences']) for a,v in checks.items()}),indent=2))+'</pre>')
    chunks.append('<h2>Per-writer DEV errors</h2><pre>'+esc(json.dumps({a:errors[a]['dev']['per_writer'] for a in errors},indent=2))+'</pre><h2>CTC beam10 mean-only</h2><pre>'+esc(json.dumps({a:b['groups'] for a,b in beam.items()},indent=2))+'</pre>')
    for g,page,f in pages:chunks.append(f'<h2>{g}page{page}</h2><img loading="lazy" src="{f}">')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes());(out/'index.html').write_text('\n'.join(chunks));publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),selected={a:dict(additionalstep=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for a,r in results.items()},reload_disagreements={a:len(v['mean_cpu_gpu_transcript_differences']) for a,v in checks.items()})
