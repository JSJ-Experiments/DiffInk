"""Independent CPU reload and all160 marker-free panels for4/8 or2/4-point readers."""
import html,json
from pathlib import Path
import h5py,numpy as np,torch
from .ocr_frame_study import frame_cache,paired_posterior_sampler,POOL_SHA
from .ocr_pool_study import load_pool,cache_corpus,evaluate,SOURCE,SHA
from .ocr_context_features import fit_stats,make_head
from .ocr_context_study import tensor_digest
from .writer_expansion import load
from .pen_ab import file_sha
from .report_pointer import publish_pointer


def reader_pair(results):
    if set(results)=={'points8','points4'}:return (8,4)
    if set(results)=={'points4','points2'}:return (4,2)
    raise ValueError('both completed8/4 or4/2 reader arms required')


def compare_rows(a,b,ids,frames=(8,4)):
    left={r['sample_id']:r for r in a['lines']};right={r['sample_id']:r for r in b['lines']}
    ka,kb=(f'errors{f}' for f in frames)
    rows=[dict(sample_id=i,**{ka:left[i]['mu']['errors'],kb:right[i]['mu']['errors']}) for i in ids]
    return dict(improved=sum(r[kb]<r[ka] for r in rows),tied=sum(r[kb]==r[ka] for r in rows),worsened=sum(r[kb]>r[ka] for r in rows),lines=rows)


def slack_comparison(records,evaluations,ids):
    """Use SAME8-point slack bins in both arms, not differently selected subsets."""
    maps={k:{r['sample_id']:r for r in v['lines']} for k,v in evaluations.items()};bins={k:[] for k in ('tight_le0.25','middle_0.25to0.5','loose_gt0.5')}
    for i in ids:
        rec=records[i];text=rec['text'];chars=len(text);required=chars+sum(a==b for a,b in zip(text,text[1:]))
        margin=((rec['points']+7)//8-required)/chars
        key='tight_le0.25' if margin<=.25 else 'middle_0.25to0.5' if margin<=.5 else 'loose_gt0.5';bins[key].append(i)
    return {k:dict(lines=len(values),groups={name:dict(errors=sum(m[i]['mu']['errors'] for i in values),characters=sum(m[i]['mu']['characters'] for i in values),
        cer=sum(m[i]['mu']['errors'] for i in values)/sum(m[i]['mu']['characters'] for i in values) if values else None) for name,m in maps.items()}) for k,values in bins.items()}


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());configs={k:json.loads((directory/k/'config.json').read_text()) for k in results}
    frames_pair=reader_pair(results);coarse,fine=frames_pair;left,right=(f'points{f}' for f in frames_pair)
    pool,m,vocab=load_pool(root,POOL_SHA)
    if (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise AssertionError('pool drift')
    cfg=configs[left]['cfg'];probe=m['splits']['small_train'][::6];ids=probe+m['splits']['dev']+m['splits']['held_out']
    base,_,_,loaded,alphabet,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if loaded!=cfg or alphabet!=vocab:raise AssertionError('source contract changed')
    base.eval().requires_grad_(False);digest=tensor_digest(base.state_dict());subset=dict(m,records={i:m['records'][i] for i in ids})
    original,texts,audit=cache_corpus(base,pool,subset,vocab,directory,device='cpu',save_geometry=False)
    cpu={};gpu={};checks={};maps={}
    for frames in frames_pair:
        name=f'points{frames}';c=configs[name];r=results[name];folder=directory/name
        if c['points_per_frame']!=frames or c['pool_manifest_sha256']!=POOL_SHA or c['source_sha256']!=SHA:raise AssertionError('framing/source drift')
        if c['train_ids']!=m['splits']['large_train'] or c['dev_ids']!=m['splits']['dev'] or c['held_out_ids']!=m['splits']['held_out'] or c['feature_calibration_ids']!=m['splits']['small_train']:raise AssertionError('split/calibration drift')
        saved=torch.load(folder/'head-best.pt',map_location='cpu',weights_only=True)
        if saved['config']!=c or saved['updates']!=r['best_step'] or file_sha(folder/'head-best.pt')!=r['selected_sha256']:raise AssertionError('selected checkpoint drift')
        cache=frame_cache(original,frames);head=make_head(cfg,len(vocab)+1,'relative_scaled',c['feature_stats'],seed=c['seed'],points_per_frame=frames);head.load_state_dict(saved['ocr_state_dict']);head.eval()
        splits=dict(train=probe,dev=m['splits']['dev'],held_out=m['splits']['held_out'],common_train_probe=probe)
        row=evaluate(head,cache,texts,splits,vocab,folder,'cpu-reload',ids,posterior_sampler=paired_posterior_sampler(original,cache,frames));cpu[name]=row;maps[name]={r['sample_id']:r for r in row['lines']}
        original_eval=json.loads((folder/f'ocr-{r["best_step"]}.json').read_text());gpu[name]=original_eval;gpu_map={r['sample_id']:r for r in original_eval['lines']}
        differences=[dict(sample_id=r['sample_id'],cpu=r['mu']['decoded'],gpu=gpu_map[r['sample_id']]['mu']['decoded']) for r in row['lines'] if r['mu']['decoded']!=gpu_map[r['sample_id']]['mu']['decoded']]
        checks[name]=dict(mean_cpu_gpu_transcript_differences=differences,cpu_scope='192 eval/probe, not all8192TRAIN; CPU train means only common32',cpu_groups=row['groups'],posterior_device_rng_not_paired=True)
    if tensor_digest(base.state_dict())!=digest or file_sha(root/SOURCE)!=SHA:raise AssertionError('CPU codec mutation')
    logs={k:[json.loads(line) for line in (directory/k/'metrics.jsonl').read_text().splitlines()] for k in results}
    paired=dict(same_initialization_dropout_seed=configs[left]['seed']==configs[right]['seed'],same_fresh_head_weights=results[left]['initial_head_tensor_sha256']==results[right]['initial_head_tensor_sha256'],
        same_batches=[r['sample_ids'] for r in logs[left]]==[r['sample_ids'] for r in logs[right]],
        same_lr_schedule=[r['lr'] for r in logs[left]]==[r['lr'] for r in logs[right]],
        same_schedule_digest=results[left]['sample_schedule_sha256']==results[right]['sample_schedule_sha256'],
        same_update_count=results[left]['last_step']==results[right]['last_step'],codec_bitwise_frozen=all(r['entire_codec_bitwise_unchanged'] for r in results.values()))
    if not all(paired.values()):raise AssertionError('unequal controlled study budgets/state')
    preflight=json.loads((directory/'codec-preflight.json').read_text());rows=preflight['lines']
    if not preflight['passed'] or len(rows)!=len(m['records']):raise AssertionError('incomplete GPU codec preflight')
    brief=dict(lines=len(rows),mean_per_line_x_rmse=float(np.mean([r['geometry']['x_rmse'] for r in rows])),mean_per_line_y_rmse=float(np.mean([r['geometry']['y_rmse'] for r in rows])),mean_per_line_turn_p90=float(np.mean([r['geometry']['turn_angle_error_degrees']['p90'] or 0 for r in rows])),max_packed_xy_difference=preflight['maximum_packed_xy_difference'],all_pen_boundaries_and_final_eoc_perfect=True)
    summary=dict(results=results,configs=configs,paired=paired,source_sha256=SHA,pool_manifest_sha256=POOL_SHA,codec_gpu_preflight=brief,codec_cpu_eval192=audit,
        cpu_reload=checks,cpu_line_error_changes={g:compare_rows(cpu[left],cpu[right],m['splits'][g],frames_pair) for g in ('dev','held_out')},
        same_reference8frame_slack={g:slack_comparison(m['records'],gpu,m['splits'][g]) for g in ('dev','held_out')},
        caveats='Fresh paired heads, not parent continuations. Same input384 and parameter weights; different active field counts and chronological frame grouping, TRAIN192 calibration moments and positional indices differ. Dropout unpaired due different shapes; original noise paired before splitting on same device. DEV reused/five writers, codec prior TRAIN192 report prompt overlap; not independent IAM benchmark/paper reproduction.')
    summary['frame_pair']=list(frames_pair)
    summary['head_initialization_dropout_seed']=configs[left]['seed']
    out=directory/'report';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for name,r in results.items():
        for ax,g in zip(axes,('train','dev','held_out')):
            ax.plot([h['step'] for h in r['history']],[100*h['groups'][g]['mu']['cer'] for h in r['history']],label=name,marker='.')
            ax.axvline(r['best_step'],alpha=.2);ax.set_title(g+' CER');ax.set_xlabel('updates');ax.set_ylabel('%');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig);pages=[]
    with h5py.File(pool/'lines.h5') as targets,h5py.File(directory/'geometry-source.h5') as recon:
        for group in ('held_out','dev'):
            for page,page_ids in enumerate(paginate_ids(m['splits'][group],limit=8),1):
                fig,axes=plt.subplots(len(page_ids),2,figsize=(15,2.55*len(page_ids)),squeeze=False)
                for j,sid in enumerate(page_ids):
                    target=targets[sid]['point_seq'][:];target[:,:2]*=.01;fixed=recon[sid]['mean'][:];lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                    caption=f'Shared FROZEN reconstruction\n{coarse}-point OCR: '+maps[left][sid]['mu']['decoded']+f'\n{fine}-point OCR: '+maps[right][sid]['mu']['decoded']
                    for ax,points,title in ((axes[j,0],target,'IAM/RDP target: '+texts[sid]),(axes[j,1],fixed,caption)):
                        draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
                fig.tight_layout();fname=f'{group}-{page}.png';fig.savefig(out/fname,dpi=130);plt.close(fig);pages.append((group,page,fname))
    esc=html.escape;chunks=['<meta charset="utf-8"><title>Frozen codec:paired OCR frames</title><style>body{font-family:sans-serif;max-width:1450px;margin:auto}img{max-width:100%}td{padding:.4em}pre{white-space:pre-wrap}</style><h1>Fresh paired readers</h1>',
        '<h2>What to look for</h2><p>The handwriting is <b>identical</b> across readers: target on left, shared frozen reconstruction on right. Compare OCR captions, not supposed new stroke quality. Re-indexing40 preserved fields into chronological smaller blocks involves no resampling, smoothing or codec modification.</p>',
        '<p>Same8192 TRAIN/128 DEV/32 report, fresh identical head weights/parameter count, same per-update batch IDs and8000-update budget (see exact config), AdamW5e-4→1e-4 after75%, batch16.Finer frames double CTC index resolution and alter input grouping/local-relative anchors/calibration/position indices; this does NOT isolate CTC length alone. Dropout streams differ. Calibration uses same TRAIN192 real points but moments refitted per granularity. Posterior noise drawn in original384×T8 space before splitting,20 draws on192 eval/probe only. No LM.</p>',
        '<p>DEV alone selects. Five DEV writers/repeated evaluations and prior codec192 prompt overlap to report32 limit claims. Initialized polyphase transport research codec, NOT authors’ semantic VAE reproduction. Codec all params/buffers frozen; no joint CTС/geometry/KL/style/InkDiT.</p><p><a href="summary.json">Metrics/provenance</a> · <a href="../pool-manifest.json">Pinned dataset</a></p><table border="1"><tr><th>reader/selected step</th><th>TRAIN CER</th><th>DEV mean / posterior CER</th><th>report32 mean / posterior CER</th></tr>']
    for name,r in results.items():
        v=r['selected'];chunks.append(f'<tr><td>{name}/{r["best_step"]}</td><td>{v["train"]["mu"]["cer"]:.4%}</td><td>{v["dev"]["mu"]["cer"]:.4%} / {v["dev"]["sampled"]["cer"]:.4%}</td><td>{v["held_out"]["mu"]["cer"]:.4%} / {v["held_out"]["sampled"]["cer"]:.4%}</td></tr>')
    chunks.insert(1,f'<p>{coarse} versus {fine} points/frame; OCR initialization/dropout seed {configs[left]["seed"]}; data-order seed43 unchanged.</p>')
    chunks.append('</table><img src="learning.png"><h2>Geometry gate / pairing / CPU reload</h2><pre>'+esc(json.dumps(dict(codec=brief,paired=paired,cpu_reload={k:{a:b for a,b in v.items() if a!='cpu_groups'} for k,v in checks.items()}),indent=2))+'</pre>')
    chunks.append('<h2>Paired line errors and SAME reference8-point CTC slack bins</h2><p>Slack=(ceil(real points/8)−[characters+adjacent repeats])/characters; nonuniform index frames, NOT physical duration. Same bins for both readers. Exploratory/confounded, not causal proof.</p><pre>'+esc(json.dumps(dict(changes={g:{k:v for k,v in values.items() if k!='lines'} for g,values in summary['cpu_line_error_changes'].items()},slack=summary['same_reference8frame_slack']),indent=2))+'</pre>')
    for group,page,fname in pages:chunks.append(f'<h2>{group} page{page}</h2><img loading="lazy" src="{fname}">')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes());(out/'index.html').write_text('\n'.join(chunks));publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),arms={k:dict(best_step=r['best_step'],dev_cer=r['selected']['dev']['mu']['cer'],report_cer=r['selected']['held_out']['mu']['cer']) for k,r in results.items()})
