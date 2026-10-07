"""Independent reload, verbatim error audit and marker-free frozen-codec gallery."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
ARMS=("transformer","bigru")
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_convergence import POOL_SHA
from .ocr_frame_study import frame_cache,paired_posterior_sampler
from .ocr_context_features import make_head as transformer_head
from .ocr_recurrent import make_head as recurrent_head
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .pen_ab import file_sha
from .report_pointer import publish_pointer
from .ocr_error_analysis import analyze

FIXED=('profile','torch_version','cfg','source_rel','source_sha256','pool_rel','pool_manifest_sha256','points_per_frame','feature_mode','feature_stats','feature_calibration_ids','train_ids','dev_ids','held_out_ids','common_train_probe','posterior_evaluation_ids','posterior_draws','physical_batch','encoder_physical_batch','attention_radius','dropout','blank_bias','base_lr','final_lr','lr_drop_step','betas','weight_decay','clip','seed','schedule_seed','schedule_sha256','geometry_gate_mode','intervention','max_updates','max_wall_seconds','eval_every','train_latent','selection','checkpoint_contract')


def pairing(configs,results,logs):
    if set(configs)!=set(ARMS) or set(results)!=set(ARMS) or set(logs)!=set(ARMS):raise ValueError('complete freshTransformer/BiGRU pair required')
    a,b=ARMS;c,d=configs[a],configs[b]
    for k in FIXED:
        if c[k]!=d[k]:raise ValueError('uncontrolled architectural change: '+k)
    if c['architecture']!=a or d['architecture']!=b:raise ValueError('explicit architectures required')
    checks=dict(same_batches=[r['sample_ids'] for r in logs[a]]==[r['sample_ids'] for r in logs[b]],same_lr=[r['lr'] for r in logs[a]]==[r['lr'] for r in logs[b]],same_schedule=results[a]['sample_schedule_sha256']==results[b]['sample_schedule_sha256'],equal_completed_budget=all(r['last_step']==c['max_updates'] and r['stop_reason']=='budget_completed' for r in results.values()),entire_codec_frozen=all(r['entire_codec_bitwise_unchanged'] for r in results.values()),initialization_and_dropout_not_paired=True)
    if not all(checks.values()):raise ValueError('architectural pairing guard failed')
    for arm in ARMS:
        if len(logs[arm])!=c['max_updates'] or any(r['step']!=j+1 for j,r in enumerate(logs[arm])):raise ValueError('complete optimizer logs required')
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
    prev=configs[ARMS[0]];cfg=prev['cfg'];ids=prev['posterior_evaluation_ids']
    if (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise ValueError('pool fingerprint drift')
    for c in configs.values():
        if c['source_sha256']!=SHA or c['pool_manifest_sha256']!=POOL_SHA:raise ValueError('pinned source drift')
        for k,key in [('train_ids','large_train'),('dev_ids','dev'),('held_out_ids','held_out'),('feature_calibration_ids','small_train')]:
            if c[k]!=m['splits'][key]:raise ValueError('split/calibration drift')
    base,_,_,loaded,alphabet,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if cfg!=loaded or alphabet!=vocab:raise ValueError('codec/vocab drift')
    base.eval().requires_grad_(False);digest=tensor_digest(base.state_dict());subset=dict(m,records={i:m['records'][i] for i in ids})
    original,texts,audit=cache_corpus(base,pool,subset,vocab,directory,device='cpu',save_geometry=False);cache=frame_cache(original,4);sampler=paired_posterior_sampler(original,cache,4)
    splits=dict(train=prev['common_train_probe'],dev=prev['dev_ids'],held_out=prev['held_out_ids'],common_train_probe=prev['common_train_probe']);cpu={};checks={};maps={};errors={};beam={}
    for name in ARMS:
        folder=directory/name;saved=torch.load(folder/'head-best.pt',map_location='cpu',weights_only=True);c=configs[name];r=results[name]
        if saved['config']!=c or saved['updates']!=r['best_step'] or file_sha(folder/'head-best.pt')!=r['selected_sha256']:raise ValueError('selected checkpoint drift')
        head=(transformer_head(cfg,len(vocab)+1,'relative_scaled',prev['feature_stats'],seed=c['seed'],points_per_frame=4) if name=='transformer' else recurrent_head(cfg,len(vocab)+1,prev['feature_stats'],seed=c['seed']))
        head.load_state_dict(saved['ocr_state_dict'],strict=True);head.eval()
        if sum(p.numel() for p in head.parameters())!=c['parameter_count']:raise ValueError('reader architecture/parameter drift')
        row=evaluate(head,cache,texts,splits,vocab,folder,'cpu-reload',ids,posterior_sampler=sampler);cpu[name]=row;maps[name]={r['sample_id']:r for r in row['lines']}
        errors[name]={g:analyze(row['lines'],group_ids,m['records']) for g,group_ids in splits.items()}
        beam[name]=audit_head(head,cache,texts,dict(dev=prev['dev_ids'],held_out=prev['held_out_ids']),vocab,width=10)
        (folder/'beam-mean-only.json').write_text(json.dumps(beam[name],indent=2)+'\n')
        original_eval=json.loads((folder/f'ocr-{r["best_step"]}.json').read_text());gm={x['sample_id']:x for x in original_eval['lines']}
        diff=[dict(sample_id=x['sample_id'],cpu=x['mu']['decoded'],gpu=gm[x['sample_id']]['mu']['decoded']) for x in row['lines'] if x['mu']['decoded']!=gm[x['sample_id']]['mu']['decoded']]
        checks[name]=dict(mean_cpu_gpu_transcript_differences=diff,cpu_scope='192eval/probe only, NOT8192TRAIN',cpu_groups=row['groups'],posterior_device_rng_unpaired=True)
    if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA:raise AssertionError('CPU codec mutation')
    preflight=json.loads((directory/'codec-preflight.json').read_text());rows=preflight['lines']
    if not preflight['passed'] or len(rows)!=len(m['records']):raise ValueError('incomplete8352 GPU codec gate')
    geometry=dict(lines=len(rows),mean_per_line_x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in rows])),mean_per_line_y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in rows])),mean_per_line_turn_p90=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] or 0 for r in rows])),maximum_packed_xy_difference=preflight['maximum_packed_xy_difference'],all_pen_boundaries_and_final_eoc_perfect=True)
    reuse=json.loads((directory/'geometry-gate-reuse.json').read_text())
    if reuse['source_sha256']!=SHA or reuse['pool_manifest_sha256']!=POOL_SHA or reuse['new_encoder_checks']!=len(m['records']) or file_sha(directory/'codec-preflight.json')!=reuse['prior_decoder_gate_sha256'] or file_sha(directory/'geometry-source.h5')!=reuse['prior_geometry_h5_sha256']:raise ValueError('reused geometry provenance drift')
    summary=dict(results=results,configs=configs,source_sha256=SHA,pool_manifest_sha256=POOL_SHA,paired=paired,geometry_gate_reuse=reuse,codec_prior_gpu_preflight=geometry,codec_cpu_eval192=audit,cpu_reload=checks,cpu_errors=errors,beam_mean_only=beam,
      recurrent_vs_transformer={g:line_changes(cpu[ARMS[0]],cpu[ARMS[1]],prev[k]) for g,k in [('dev','dev_ids'),('held_out','held_out_ids')]},
      caveats='Whole architecture comparison, not identical initialweights/dropoutstreams/parametercount. Same reused128DEV/fivewriters and32report; priorcodec reportpromptoverlap. Prior bound8352mean decodergate reused, all8352encoder/pointchecks new; CPU192mean decode gate independently repeated. No newall-posteriortrajectory certification. Initializedpolyphase transport research, NOTpapersemanticVAE. Beam10 mean-only/noLM/neverselects. No drawing-quality/geometry/KL/style/InkDiT change.')
    out=directory/'report';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for a,r in results.items():
        for ax,g in zip(axes,('train','dev','held_out')):
            ax.plot([h['step'] for h in r['history']],[100*h['groups'][g]['mu']['cer'] for h in r['history']],label=a,marker='.');ax.axvline(r['best_step'],alpha=.2);ax.set_title(g+'CER');ax.set_xlabel('updates');ax.set_ylabel('%');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig);pages=[]
    with h5py.File(pool/'lines.h5') as target_file,h5py.File(directory/'geometry-source.h5') as fixed_file:
        for group,k in [('held_out','held_out_ids'),('dev','dev_ids')]:
            for page,page_ids in enumerate(paginate_ids(prev[k],limit=8),1):
                fig,axes=plt.subplots(len(page_ids),2,figsize=(15,2.8*len(page_ids)),squeeze=False)
                for j,sid in enumerate(page_ids):
                    target=target_file[sid]['point_seq'][:];target[:,:2]*=.01;fixed=fixed_file[sid]['mean'][:];lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                    caption='SharedFROZEN reconstruction\n'+'\n'.join(a+': '+maps[a][sid]['mu']['decoded'] for a in ARMS)
                    for ax,p,title in [(axes[j,0],target,'IAM/RDP target: '+texts[sid]),(axes[j,1],fixed,caption)]:
                        draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
                fig.tight_layout();f=f'{group}-{page}.png';fig.savefig(out/f,dpi=130);plt.close(fig);pages.append((group,page,f))
    esc=html.escape;chunks=['<meta charset="utf-8"><title>Four-point OCR recurrent comparison</title><style>body{font-family:sans-serif;max-width:1450px;margin:2em auto}img{max-width:100%}td,th{padding:.5em}pre{white-space:pre-wrap}</style><h1>Frozen handwriting: Transformer versus packed BiGRU</h1>',
      '<h2>What to look for</h2><p>Drawings unchanged: target left, shared frozen reconstruction right. Compare Transformer and BiGRU OCR captions. Same4-point relative-scaled inputs, calibration and chronological point grouping; no smoothing, resampling or label changes.</p>',
      '<p>Fresh seeded readers; same8192TRAIN/128DEV/32report,8000updates/samplebatches/LRs/CTC objective. Transformer384/3layers/4heads/sinusoidalpositions versus packed3-layerbidirectionalGRU320/direction,5.50M versus5.25Mparameters. Whole reader architecture intervention, NOT same initialization tensors or dropout streams. Padding omitted from backward recurrence; inputprojection and4-point outputclock unchanged. LR5e-4→1e-4 at6000,clip5,AdamWbetas.9/.99/wd1e-4/dropout.1. DEVonlyselects, report32neverselects.</p>',
      '<p>Exact priorcodec/config/pool8352meangeometry gate reused; all8352encoders/pointchecks refreshed. CPU192independentdecoder/reload/20posteriorOCRdraws. NOT newfull-pool sampledtrajectory certification or papersemanticVAE reproduction. ReusedfiveDEVwriters/reportpromptoverlap remain. Beam10/noLM mean-only diagnostic neverselects.</p>',
      '<p><a href="summary.json">Exact metrics/provenance</a> · <a href="../geometry-gate-reuse.json">Bound geometry reuse</a> · <a href="../pool-manifest.json">Pinned dataset</a></p>',
      '<table border="1"><tr><th>reader/step</th><th>TRAIN CER</th><th>DEV mean/posterior CER</th><th>report mean/posterior CER</th></tr>']
    for name in ARMS:
        v=results[name]['selected'];step=results[name]['best_step']
        chunks.append(f'<tr><td>{name}/{step}</td><td>{v["train"]["mu"]["cer"]:.4%}</td><td>{v["dev"]["mu"]["cer"]:.4%}/{v["dev"]["sampled"]["cer"]:.4%}</td><td>{v["held_out"]["mu"]["cer"]:.4%}/{v["held_out"]["sampled"]["cer"]:.4%}</td></tr>')
    chunks.append('</table><img src="learning.png"><h2>Paired controls/geometry/reload</h2><pre>'+esc(json.dumps(dict(paired=paired,geometry=geometry,reload={a:len(v['mean_cpu_gpu_transcript_differences']) for a,v in checks.items()}),indent=2))+'</pre>')
    chunks.append('<h2>Per-writer DEV errors</h2><pre>'+esc(json.dumps({a:errors[a]['dev']['per_writer'] for a in errors},indent=2))+'</pre><h2>CTC beam10 mean-only</h2><pre>'+esc(json.dumps({a:b['groups'] for a,b in beam.items()},indent=2))+'</pre>')
    for g,page,f in pages:chunks.append(f'<h2>{g}page{page}</h2><img loading="lazy" src="{f}">')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes());(out/'index.html').write_text('\n'.join(chunks));publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),selected={a:dict(step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for a,r in results.items()},reload_disagreements={a:len(v['mean_cpu_gpu_transcript_differences']) for a,v in checks.items()})
