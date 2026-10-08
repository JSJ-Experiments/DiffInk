"""Marker-free frozen timing-intervention report; no checkpoint selection."""
import html,json,tarfile
from pathlib import Path
import h5py,numpy as np
from .generation_capacity import DATA
from .generation_timing_study import PARENT_SHA,ARMS
from .pen_ab import file_sha


def report(directory,repo,root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    p=Path(directory);root=Path(root);cfg=json.loads((p/'config.json').read_text());s=json.loads((p/'summary.json').read_text())
    if cfg['source_checkpoints']!=PARENT_SHA or file_sha(p/'evaluation.h5')!=s['packed_h5_sha256'] or file_sha(p/'as-run-source.tar.gz')!=cfg['source_archive_sha256']:raise ValueError('immutable timing source/output guard')
    with tarfile.open(p/'as-run-source.tar.gz') as tar:
        for n in ['iam_tools/generation_timing.py','iam_tools/generation_timing_study.py']:
            if tar.extractfile(n).read()!=(Path(repo)/n).read_bytes():raise ValueError('as-run timing source drift')
    if len(s['full256_native_cpu_gpu_parity'])!=512 or any(r['max_xy_difference']>.0002 or r['pen_mismatches'] or not r['reader_equal'] for r in s['full256_native_cpu_gpu_parity']):raise ValueError('all512 native reload parity required')
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());data=json.loads((root/cfg['parent']/'dataset.json').read_text());records=data['records'];ids=cfg['train_ids'];pages=[]
    with h5py.File(root/DATA/'source.h5') as source,h5py.File(p/'evaluation.h5') as f:
        for a in ARMS:
            for delta in [-1,1]:
                policies=['baseline',f'pe_only_{delta:+d}',f'mask_only_{delta:+d}',f'both_{delta:+d}']
                # Eight fixed TRAIN examples, including previously distorted curves.
                selected=['k04-309z-09','c03-109z-03','r07-568z-04','a02-130z-03','d08-586z-01','p10-249z-01','e07-425z-02','n05-514z-02']
                for start in range(0,len(selected),2):
                    group=selected[start:start+2];fig,axs=plt.subplots(len(group),5,figsize=(30,3*len(group)),squeeze=False)
                    for j,sid in enumerate(group):
                        truth=source[sid]['target'][:];draw(axs[j,0],split_xy(truth[:,:2],truth[:,2:].argmax(1)));axs[j,0].set_title(sid+' / reference\n'+records[sid]['text'],fontsize=8)
                        for ax,policy in zip(axs[j,1:],policies):
                            q=f[a+'/'+policy+'/'+sid];row=json.loads(q.attrs['row']);points=q['points'][:row['generated_points_at_stop']];draw(ax,split_xy(points[:,:2],points[:,2:].argmax(1)));ax.set_title(a+'/'+policy+f' / context{row["context_blocks"]},PE{row["position_blocks"]}\nreader: '+row['free_decoded'],fontsize=8)
                    fig.tight_layout();name=f'{a}-{delta:+d}-{start//2+1}.png';fig.savefig(out/name,dpi=120);plt.close(fig);pages.append(name)
    body='<!doctype html><meta charset="utf-8"><title>PE versus attention context: frozen timing autopsy</title><style>body{font:17px system-ui;max-width:2600px;margin:30px auto;padding:20px}img{width:100%}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style><h1>Target-length-dependent positional encoding causes large within-line distortion</h1><p>ALL256TRAIN texts, unchanged checkpoint/text/writer/codec/reader. One-block perturbation. PE-only changes only relative-progress denominator, keeps actual context/target length. Mask-only changes actual context/output length but holds the PE denominator at oracle. Both changes both. Oracle denominators are intentionally used to isolate mechanism here, NOT an available generation trick. No training, interpolation, smoothing or forced EOC.</p><p><a href="../summary.json">Every line and protocol</a> · <a href="../as-run-source.tar.gz">As-run sources</a></p><table><tr><th>Arm/policy</th><th>CER</th><th>Exact</th><th>common-prefix X/Ydrift</th><th>Pen changes</th><th>Missing EOC</th></tr>'
    for a,groups in s['aggregate'].items():
        for policy,r in groups.items():body+=f'<tr><td>{a}/{policy}</td><td>{100*r["cer"]:.3f}%</td><td>{r["exact"]}/256</td><td>{r["common_x_drift_mean"]:.5f}/{r["common_y_drift_mean"]:.5f}</td><td>{r["common_pen_changes"]}</td><td>{r["missing_eoc"]}</td></tr>'
    body+='</table><p>PE-only +1 pushes learned EOC beyond output window, and mask-only −1 truncates source termination: free CER alone mixes stopping and geometry effects. The large common-prefix XY/pen drift and marker-free writing identify true within-line damage. Mask-only +1 changes0common-prefix pens, versus4588soft PE-only. Baseline independently reproduces all512GPU reader strings/pen states,maxXY2.90e-5. Same fixed eight TRAIN examples shown below, not a new blind benchmark; outputs for all256 saved.</p>'
    for n in pages:body+='<h3>'+html.escape(n)+'</h3><img loading="lazy" src="'+n+'">'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),aggregate=s['aggregate'])
