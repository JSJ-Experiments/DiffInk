"""Marker-free one-shot new-text confirmation; no fabricated synthetic targets."""
import html,json
from pathlib import Path
import h5py
from .pen_ab import file_sha


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    p=Path(directory);s=json.loads((p/'summary.json').read_text());arms=['control','weak_alignment'];reservation=s['reservation']
    if file_sha(p/'source.h5')!=s['source_h5_sha256'] or set(s['arms'])!=set(arms):raise ValueError('paired confirmation source/arms guard')
    for a in arms:
        if any(file_sha(p/name)!=sha for name,sha in s['arms'][a]['files'].items()):raise ValueError('immutable one-shot output guard')
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());pages=[]
    for kind in ['paired','synthetic']:
        records=reservation[kind]['records'];ids=reservation[kind]['ids'] if kind=='paired' else list(records)
        for start in range(0,len(ids),4):
            group=ids[start:start+4];columns=7 if kind=='paired' else 4
            fig,axs=plt.subplots(len(group),columns,figsize=(7*columns,3*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                if kind=='paired':
                    with h5py.File(p/'source.h5') as f:truth=f[sid]['target'][:]
                    draw(axs[j,0],split_xy(truth[:,:2],truth[:,2:].argmax(1)));axs[j,0].set_title(sid+' / reference\n'+records[sid]['text'],fontsize=8)
                    columns_spec=[(a,policy) for a in arms for policy in ['oracle','estimated','generous']];axes=axs[j,1:]
                else:
                    columns_spec=[(a,policy) for a in arms for policy in ['synthetic','generous']];axes=axs[j]
                for ax,(a,policy) in zip(axes,columns_spec):
                    step=s['arms'][a]['selected_step'];folder=p/a/('synthetic' if kind=='synthetic' else '')
                    file=folder/(f'evaluation-{step}.h5' if policy=='oracle' else f'generous-evaluation-{step}.h5' if policy=='generous' else f'duration-evaluation-{step}.h5');mode='correct' if policy=='oracle' else 'generous_correct' if policy=='generous' else 'estimated_correct'
                    with h5py.File(file) as f:q=f[mode+'/'+sid];row=json.loads(q.attrs['row']);points=q['points'][:row['generated_points_at_stop']]
                    draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(records[sid]['text']+'\n'+a+'/'+policy+' / reader: '+row['free_decoded'],fontsize=8)
            fig.tight_layout();name=f'{kind}-{start//4+1}.png';fig.savefig(out/name,dpi=120);plt.close(fig);pages.append(name)
    preflight=s['preflight'];reader_cer=sum(r['reader_errors'] for r in preflight)/sum(r['characters'] for r in preflight)
    body='<!doctype html><meta charset="utf-8"><title>New-text composition confirmation</title><style>body{font:17px system-ui;max-width:2400px;margin:30px auto;padding:20px}img{width:100%}th,td{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>New reserved text, frozen TRAIN-selected models</h1><p>New16paired/16synthetic prompts reserved before final training finished. All historical1032generator transcripts and ALL THREE earlier opened16paired/16synthetic sets excluded; paired forms also exclude current TRAIN/dev and previous opened forms. Known writers/corpus-familiar reader, NOT an independent-writer benchmark. Synthetic has no reference trajectory or oracle length. No smoothing, markers, forced EOC or sample exclusions.</p><p><a href="../summary.json">Every source/output hash, reservation, controls and provenance</a></p><p>Source reader CER '+f'{100*reader_cer:.3f}%'+', exact '+str(sum(r['reader_errors']==0 for r in preflight))+'/16. No failed source/reader line excluded.</p><table><tr><th>Arm / evaluation</th><th>CER</th><th>Exact</th><th>Missing EOC</th></tr>'
    for a in arms:
        arm=s['arms'][a]
        for label,group in [('paired/oracle',arm['oracle']['confirmation']),('paired/estimated',arm['estimated']),('synthetic/estimated',arm['synthetic']),('paired/generous256',arm['generous']),('synthetic/generous256',arm['synthetic_generous'])]:
            for policy,r in group.items():body+=f'<tr><td>{a}/{label}/{policy}</td><td>{100*r["free_cer"]:.3f}%</td><td>{r["free_exact"]}/{r["evaluations"]}</td><td>{r["missing_eoc"]}</td></tr>'
    body+='</table><p>Oracle native timing, target-free timing and genuine composition remain separate gates. Paired: source then oracle/estimated/generous256 for each arm. Synthetic: estimated/generous256 for each arm, no invented reference. Generous budget uses neither source length nor predictor; first learnedEOC stops writing. All32prompts shown, not cherry-picked. After this opening these prompts are exposed; never tune/reselect these candidates on them.</p>'
    for n in pages:body+='<img loading="lazy" src="'+html.escape(n)+'">'
    (out/'index.html').write_text(body);return str(out/'index.html')
