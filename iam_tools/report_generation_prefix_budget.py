"""Marker-free generous-budget gallery; familiar writing != text composition."""
import html,json
from pathlib import Path
import h5py
from .pen_ab import file_sha


def report(directory,root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .generation_capacity import DATA
    from .curve_audit import draw,split_xy
    p=Path(directory);s=json.loads((p/'summary.json').read_text());data=json.loads((p.parent/'dataset.json').read_text())
    for a in ['noncausal','causal']:
        if file_sha(p/(a+'.h5'))!=s['arms'][a]['packed_h5_sha256']:raise ValueError('frozen budget output guard')
        result=json.loads((p.parent/a/'result.json').read_text())
        if result['selected_sha256']!=s['arms'][a]['checkpoint_sha256'] or result['best_step']!=s['arms'][a]['selected_step']:
            raise ValueError('TRAIN-selected frozen weight provenance guard')
    if any(r['blocks']!=256 for r in s['lines'] if r['policy']=='generous256'):raise ValueError('same target-free generous budget for all prompts required')
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());pages=[]
    with h5py.File(Path(root)/DATA/'source.h5') as source,h5py.File(p/'noncausal.h5') as control,h5py.File(p/'causal.h5') as causal:
        for split in ['retained_train','added_same_writer','new_train128','expansion256','unseen_prompt']:
            for start in range(0,len(data['splits'][split]),4):
                ids=data['splits'][split][start:start+4];fig,axs=plt.subplots(len(ids),5,figsize=(32,3*len(ids)),squeeze=False)
                for j,sid in enumerate(ids):
                    points=source[sid]['target'][:];draw(axs[j,0],split_xy(points[:,:2],points[:,2:].argmax(1)));axs[j,0].set_title(sid+' / processed source\n'+data['records'][sid]['text'],fontsize=8)
                    for ax,(file,arm,policy) in zip(axs[j,1:],[(control,'noncausal','native'),(control,'noncausal','generous256'),(causal,'causal','native'),(causal,'causal','generous256')]):
                        q=file[policy+'/'+sid];r=json.loads(q.attrs['row']);points=q['points'][:r['generated_points_at_stop']]
                        draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(arm+'/'+policy+'\nreader: '+r['free_decoded'],fontsize=8)
                fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=110);plt.close(fig);pages.append(name)
    body='<!doctype html><meta charset="utf-8"><title>Does a generous output budget deform handwriting?</title><style>body{font:17px system-ui;max-width:2400px;margin:30px auto;padding:20px}img{width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Frozen output-budget test</h1><p>Both arms use absolute query positions. Only hidden-query self-attention becomes causal. All256TRAIN plus8already-exposed development lines shown. Native requested windows are an oracle-timing diagnostic. Generous256 requests2048points for EVERY text, with learned firstEOC stopping, no source length, predictor or forced EOC in inference. All text remains visible; this is NOT teacher-forced stroke autoregression or original InkDiT.</p><p>Familiar-text timing, native fit, and new-text composition are separate questions. A good generous-budget TRAIN output proves timing robustness, NOT composition. Prefix drift is measured separately in latent and decoded space. Shorter-one/estimated outputs can truncate even when common prefixes are unchanged. No fictitious unequal-index reconstruction RMSE.</p><p><a href="../summary.json">All five budget policies/264lines, hashes and prefix drifts</a> · <a href="../../report/index.html">Matched-final native/estimated report</a> · <a href="../../reserved-confirmation/seal.json">New reserved gate (do not assume opened)</a></p><table><tr><th>Arm/selected step</th><th>TRAIN budget policy</th><th>CER</th><th>Exact</th><th>MissingEOC</th><th>Max latent/XY prefix drift</th><th>Pen changes</th></tr>'
    for arm in ['noncausal','causal']:
        a=s['arms'][arm]
        for policy,r in a['groups']['train256'].items():
            body+=f'<tr><td>{arm}/{a["selected_step"]}</td><td>{policy}</td><td>{100*r["cer"]:.3f}%</td><td>{r["exact"]}/256</td><td>{r["missing_eoc"]}</td><td>{r["max_latent_prefix_difference"]:.6g}/{r["max_xy_prefix_difference"]:.6g}</td><td>{r["pen_changes"]}</td></tr>'
    body+='</table><p>Source / noncausal native / noncausal generous / causal native / causal generous. Marker-free renders use PREDICTED pens, actual learned firstEOC, no smoothing or filtering. Every line saved, not handpicked.</p>'
    for name in pages:body+='<img loading="lazy" src="'+html.escape(name)+'">'
    (out/'index.html').write_text(body);return str(out/'index.html')
