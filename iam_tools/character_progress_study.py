"""CPU-only TRAIN-form-held transfer probe; no new composition gate opened.

A diagnostic of the frozen reader's weak emission clock, not generated writing.
Positive text-only interval models are fit on four folds, tested on unseen forms
in the fifth. Fold union is precisely the established256 TRAIN lines. One fixed
ridge/fold seed; no sweep, neural updates, or confirmation source access.
"""
import html
import json
import time
from pathlib import Path
import h5py
import numpy as np
from .character_progress import emission_centers,form_folds,fit_clock,predict_clock,monotonic_progress,MODES
from .generation_composition import fit_duration,predict_duration
from .generation_capacity import DATA,DATASET_SHA,SOURCE_H5_SHA
from .generation_weak_alignment_study import PROBE,PROBE_H5_SHA
from .ocr_pool import normalized_text
from .pen_ab import file_sha

PARENT='checkpoints/iam_generation_weak_continuation/20261008-134416'
PRIMARY=('static_ratio','duration_uniform','no_glyph','character','neighbor','same_endpoints_uniform')
ORACLE=('oracle_endpoints_uniform','oracle_endpoints_neighbor')


def distribution(values):
    a=np.asarray(values,dtype=float)
    if not len(a) or not np.isfinite(a).all():raise ValueError('finite nonempty diagnostic distribution required')
    return dict(mean=float(a.mean()),median=float(np.median(a)),p90=float(np.quantile(a,.9)),p99=float(np.quantile(a,.99)))


def compare_progress(pred,reference):
    error=np.abs(np.asarray(pred)-reference)
    return dict(index_error=distribution(error),nearest_token_accuracy=float((np.rint(pred)==reference).mean()))


def fold_partition(records,ids,assignment,k):
    train=[s for s in ids if assignment[s]!=k];test=[s for s in ids if assignment[s]==k]
    if not train or not test or {records[s]['prompt_family'] for s in train}&{records[s]['prompt_family'] for s in test} or {normalized_text(records[s]['text']) for s in train}&{normalized_text(records[s]['text']) for s in test}:
        raise ValueError('nonempty strictly form/transcript-held CPU partition required')
    return train,test


def predictions(models,duration,ratio,record,centers,coordinate):
    text,w=record['text'],record['writer_id'];n=len(text)
    clocks={m:predict_clock(models[m],text,w) for m in MODES}
    outputs={'static_ratio':np.minimum(coordinate/ratio,n-1),
             'duration_uniform':np.minimum(coordinate*n/predict_duration(duration,text,w),n-1)}
    outputs.update({m:monotonic_progress(clocks[m]['centers'],coordinate) for m in MODES})
    # Endpoint-preserving controls isolate the local progression shape from
    # predicted delay/span. Oracle branches deliberately consume source timing;
    # diagnostic ONLY, never export as inference models or generation metrics.
    nc=clocks['neighbor']['centers']
    outputs['same_endpoints_uniform']=monotonic_progress(np.linspace(nc[0],nc[-1],n),coordinate)
    outputs['oracle_endpoints_uniform']=monotonic_progress(np.linspace(centers[0],centers[-1],n),coordinate)
    if n==1:oc=centers.copy()
    else:oc=centers[0]+(nc-nc[0])*(centers[-1]-centers[0])/(nc[-1]-nc[0])
    outputs['oracle_endpoints_neighbor']=monotonic_progress(oc,coordinate)
    return outputs,clocks


def bootstrap_form_delta(rows,mode,baseline,seed=57143,repetitions=2000):
    """Descriptive paired cluster interval on mean linep90, fixed fold predictions."""
    groups={}
    for row in rows:
        groups.setdefault(row['prompt_family'],[]).append(row['metrics'][mode]['index_error']['p90']-row['metrics'][baseline]['index_error']['p90'])
    groups=list(groups.values());rng=np.random.default_rng(seed);means=[]
    for _ in range(repetitions):
        chosen=rng.integers(0,len(groups),len(groups));means.append(float(np.mean([d for j in chosen for d in groups[j]])))
    return dict(metric='mean linep90 token-index absolute error difference; negative favors model',mode=mode,baseline=baseline,
                observed=float(np.mean([d for g in groups for d in g])),form_clusters=len(groups),repetitions=repetitions,seed=seed,
                descriptive_percentile95=np.quantile(means,[.025,.975]).tolist(),
                caveat='Fixed five-fold out-of-form predictions; bootstrap does not refit models or quantify seed/fold/reader uncertainty; not multiplicity-adjusted.')


def render(out,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=summary['lines'];ordered=sorted(rows,key=lambda r:r['metrics']['neighbor']['index_error']['p90'])
    # Fixed display policy after CPU diagnostics, not a held/generation selection.
    display=list(dict.fromkeys([r['sample_id'] for r in ordered[-6:]]+[r['sample_id'] for r in ordered[len(rows)//2-3:len(rows)//2+3]]))
    images=[]
    with h5py.File(out/'predictions.h5') as f:
        for sid in display:
            row=next(r for r in rows if r['sample_id']==sid);g=f[sid];q=g['coordinate'][:];labels=g['labels'][:];valid=labels>=0
            fig,ax=plt.subplots(figsize=(11,3.7));ax.scatter(q[valid],labels[valid],s=9,c='black',label='forced-reader emission labels (not glyph borders)')
            for name,color in [('static_ratio','#b8b8b8'),('duration_uniform','#de8f05'),('neighbor','#0072b2'),('same_endpoints_uniform','#009e73')]:ax.plot(q,g[name][:],label=name,color=color,lw=1.3)
            ax.set_xlabel('(reader frame + 0.5) / 2; packed8 index-block clock, not physical time');ax.set_ylabel('transcript token INDEX')
            ax.set_title(sid+' | '+row['text'],fontsize=10);ax.legend(fontsize=7);fig.tight_layout();fn=sid+'.png';fig.savefig(out/fn,dpi=130);plt.close(fig);images.append(fn)
    table=''.join('<tr><td>'+html.escape(a)+'</td>'+''.join(f'<td>{summary["aggregate"][a][k]:.4f}</td>' for k in ['pooled_median','pooled_p90','pooled_p99','mean_line_p90','nearest_token_accuracy'])+'</tr>' for a in PRIMARY+ORACLE)
    page='''<!doctype html><meta charset="utf-8"><title>Text-only weak monotonic progress: form-held CPU probe</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #bbb}img{width:100%}code{overflow-wrap:anywhere}</style><h1>Text-only monotonic progress: TRAIN-form-held CPU probe</h1><p><b>This is timing, NOT a handwriting-generation improvement.</b> The frozen corpus-familiar bidirectional reader supplies weak emission centers, not exact IAM character borders. No GPU/neural update, blind prompt, KL or style objective. All 256 established TRAIN lines tested out-of-form once across five fixed folds. Every source retained.</p><h2>Out-of-form timing error</h2><p>Absolute token-index error at nonblank four-index emissions. Not physical time, velocity, character geometry, CER or generated text accuracy. Same-endpoint uniform uses the neighbor model's predicted endpoints to isolate within-line speed. Oracle-endpoint arms use source timing and are diagnostic-only.</p><table><tr><th>Clock</th><th>Pooled median</th><th>Pooled p90</th><th>Pooled p99</th><th>Mean line p90</th><th>Nearest token accuracy</th></tr>'''+table+'''</table><h2>Protocol</h2><p>Five folds; complete forms and normalized transcripts disjoint. Folds seed57142; fixed ridge10, equal line weights, positive log-gap regression plus one TRAIN-only arithmetic smearing factor. Character and neighboring-character modes compare to an otherwise matched no-glyph model. Fallback for unknown characters is character class and for unknown writers zero writer intercept. No target duration at primary inference. The eight exposed development lines and all four opened confirmation sets are untouched.</p><p>Interpolation is an explicit proposed weak progress coordinate, not evidence that the actual pen corresponds to that character. Blank gaps, delayed marks, BiGRU lookahead and sparse RDP point spacing remain confounds. No generated trajectory or visual-fidelity claim. Display: six worst neighbor p90 and six around the median, fixed descriptive policy.</p><p><a href="summary.json">All256 metrics, folds and caveats</a> · <a href="config.json">Config/provenance</a> · <a href="predictions.h5">All256 packed predictions</a> · <a href="full-train-model.json">Reusable TRAIN-only clock (not promoted)</a></p>'''+''.join('<img loading="lazy" src="'+fn+'">' for fn in images)
    (out/'index.html').write_text(page)


def run(root='data'):
    start=time.monotonic();root=Path(root);parent=root/PARENT;probe=root/PROBE;source=root/DATA
    if file_sha(source/'dataset.json')!=DATASET_SHA or file_sha(probe/'alignment.h5')!=PROBE_H5_SHA:
        raise ValueError('pinned source manifest/teacher guard')
    config=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());teacher=json.loads((probe/'summary.json').read_text())
    ids=data['training_ids']['control'];records=data['records']
    if ids!=data['training_ids']['weak_alignment'] or len(set(ids))!=256 or {r['sample_id'] for r in teacher['lines']}!=set(ids) or teacher['reader_sha256']!=config['reader_sha256'] or teacher['source_h5_sha256']!=SOURCE_H5_SHA:
        raise ValueError('exact established TRAIN256 teacher and reader required')
    original=json.loads((source/'dataset.json').read_text())
    if any(records[s]!=original['records'][s] for s in ids):raise ValueError('TRAIN metadata drift')
    with h5py.File(probe/'alignment.h5') as f:
        if set(f)!=set(ids):raise ValueError('no held teacher paths allowed')
        labels={s:f[s]['forced_token_indices'][:] for s in ids}
    out=root/'checkpoints/iam_character_progress'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    for module in ['character_progress.py','character_progress_study.py','generation_composition.py']:
        (out/module).write_bytes(Path(__file__).with_name(module).read_bytes())
    assignment=form_folds(records,ids,folds=5,seed=57142)
    cfg=dict(parent=PARENT,teacher=PROBE,teacher_h5_sha256=PROBE_H5_SHA,teacher_summary_sha256=file_sha(probe/'summary.json'),source_manifest=DATA,source_manifest_sha256=DATASET_SHA,
             reader_sha256=config['reader_sha256'],scope='precisely256 established TRAIN, no exposed development or blind source opening',folds=5,fold_seed=57142,ridge=10.,modes=list(MODES),
             primary=list(PRIMARY),oracle_diagnostic_only=list(ORACLE),assignment=assignment,train_ids=ids,
             definitions='Teacher centers mean((emission frame+.5)/2); intervals [0, centers, ceil(realpoints/8)]; positive log-ridge regression. Frame axis counts processed indices, not uniform physical time.',
             exploratory_pilot='/tmp/pilot-character-progress.py ran on same TRAIN256 before this fixed diagnostic; this is not preregistered independent evidence',
             no_neural_updates=True,no_confirmation_opening=True,not_promoted=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    rows=[];fold_summaries=[];pooled={a:[] for a in PRIMARY+ORACLE}
    with h5py.File(out/'predictions.h5','w') as dest:
        for k in range(5):
            train,test=fold_partition(records,ids,assignment,k)
            models={m:fit_clock(records,{s:labels[s] for s in train},train,mode=m,ridge=10.) for m in MODES}
            (out/f'fold-{k}-models.json').write_text(json.dumps(models,indent=2)+'\n')
            writers=sorted({records[s]['writer_id'] for s in train});duration=fit_duration(records,train,writers)
            ratio=duration['blocks_per_character'];unknown_chars=set();unknown_writers=set()
            for sid in test:
                r=records[sid];lab=labels[sid];centers,_=emission_centers(lab,len(r['text']),r['points']);q=(np.arange(len(lab))+.5)/2;valid=lab>=0
                preds,clocks=predictions(models,duration,ratio,r,centers,q);g=dest.create_group(sid);g.create_dataset('coordinate',data=q);g.create_dataset('labels',data=lab)
                g.create_dataset('forced_emission_centers',data=centers)
                metrics={};total=(r['points']+7)//8
                for name,p in preds.items():
                    g.create_dataset(name,data=p);metrics[name]=compare_progress(p[valid],lab[valid]);pooled[name].extend(np.abs(p[valid]-lab[valid]).tolist())
                nc=clocks['neighbor'];unknown_chars.update(nc['unknown_characters'])
                if not nc['known_writer']:unknown_writers.add(r['writer_id'])
                row=dict(sample_id=sid,text=r['text'],writer_id=r['writer_id'],prompt_family=r['prompt_family'],fold=k,characters=len(r['text']),nonblank_frames=int(valid.sum()),
                         metrics=metrics,neighbor_center_error_blocks=distribution(np.abs(nc['centers']-centers)),neighbor_total_blocks=nc['total_blocks'],actual_total_blocks=total,
                         neighbor_total_absolute_error=abs(nc['total_blocks']-total),duration_uniform_total_absolute_error=abs(predict_duration(duration,r['text'],r['writer_id'])-total),
                         unknown_characters=nc['unknown_characters'],known_writer=nc['known_writer'])
                rows.append(row);g.attrs['row']=json.dumps(row)
            fold_summaries.append(dict(fold=k,train_ids=train,test_ids=test,train_forms=len({records[s]['prompt_family'] for s in train}),test_forms=len({records[s]['prompt_family'] for s in test}),unknown_characters=sorted(unknown_chars),unknown_writers=sorted(unknown_writers)))
            print('fold',k,'train',len(train),'test',len(test),flush=True)
    if {r['sample_id'] for r in rows}!=set(ids) or len(rows)!=256:raise ValueError('one out-of-fold result per TRAIN line required')
    aggregate={}
    for name,e in pooled.items():
        d=distribution(e);aggregate[name]=dict(pooled_median=d['median'],pooled_p90=d['p90'],pooled_p99=d['p99'],mean_line_p90=float(np.mean([r['metrics'][name]['index_error']['p90'] for r in rows])),
                                             nearest_token_accuracy=float(np.average([r['metrics'][name]['nearest_token_accuracy'] for r in rows],weights=[r['nonblank_frames'] for r in rows])))
    full=fit_clock(records,labels,ids,mode='neighbor',ridge=10.)
    (out/'full-train-model.json').write_text(json.dumps(full,indent=2)+'\n')
    insample=[]
    for sid in ids:
        c=predict_clock(full,records[sid]['text'],records[sid]['writer_id']);lab=labels[sid];q=(np.arange(len(lab))+.5)/2;v=lab>=0
        insample.extend(np.abs(monotonic_progress(c['centers'],q)[v]-lab[v]).tolist())
    # Source metadata confirms no cross-form identical texts, allowing ordinary
    # form-cluster bootstrap here; generic fold helper connects such forms.
    text_forms={}
    for sid in ids:text_forms.setdefault(normalized_text(records[sid]['text']),set()).add(records[sid]['prompt_family'])
    if any(len(fs)>1 for fs in text_forms.values()):raise ValueError('bootstrap must cluster connected text/form groups')
    summary=dict(config=cfg,lines=sorted(rows,key=lambda r:ids.index(r['sample_id'])),aggregate=aggregate,folds=fold_summaries,
                 neighbor_in_sample_index_error=distribution(insample),neighbor_duration_mae=float(np.mean([r['neighbor_total_absolute_error'] for r in rows])),duration_uniform_mae=float(np.mean([r['duration_uniform_total_absolute_error'] for r in rows])),
                 descriptive_form_bootstrap=[bootstrap_form_delta(rows,'neighbor',a) for a in ['static_ratio','duration_uniform','no_glyph','same_endpoints_uniform']],
                 source_reader_exact=sum(r['reader_errors']==0 for r in teacher['lines']),source_reader_lines=256,nonblank_frames=sum(r['nonblank_frames'] for r in rows),
                 definitions=cfg['definitions'],caveats='Not glyph segmentation, unseen generation, reader-independent validation or architecture promotion. TRAIN-source reader familiar; forced paths may conceal reader errors; every256 source retained. No new confirmation, new source trajectory, decoder change, KL/style or GPU job. Historical curve exemplars not retested.')
    render(out,summary);summary['wall_seconds']=time.monotonic()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),aggregate=aggregate,bootstrap=summary['descriptive_form_bootstrap'],duration_mae=summary['neighbor_duration_mae'],wall_seconds=summary['wall_seconds']),indent=2),flush=True)
    return str(out)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='data',help='Local mirror containing pinned packed TRAIN artifacts; CPU only')
    args=parser.parse_args()
    run(args.root)
