"""Marker-free matched KL trade-off report; final arms, not selected step0."""
import json,html
from pathlib import Path
import numpy as np


def report(directory,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .codec_kl_study import verify,SOURCE,SHA
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);results=json.loads((directory/'result.json').read_text());arms=list(results)
    prov=json.loads((directory/arms[0]/'provenance.json').read_text());splits=prov['splits'];targets={}
    for kind,ids in [('train',splits['train']),('val',splits['held_out'])]:
        with h5py.File(root/f'diffink/iam_overfit/tiny_{kind}.h5') as hf:
            for sid in ids:
                a=hf[sid]['point_seq'][:].copy();a[:,:2]*=.01;targets[sid]=a
    checks={a:verify(directory/a,'/app',root) for a in arms}
    stages={a:json.loads((directory/a/f'eval-{results[a]["last_step"]}.json').read_text()) for a in arms}
    initial=json.loads((directory/arms[0]/'eval-0.json').read_text())
    for arm in arms[1:]:
        other=json.loads((directory/arm/'eval-0.json').read_text())
        if other!=initial:raise AssertionError('initial paired evaluation mismatch')
    posterior={a:json.loads((directory/a/f'posterior-{results[a]["last_step"]}.json').read_text()) for a in arms}
    before=json.loads((directory/arms[0]/'posterior-0.json').read_text())
    protected=any(a.startswith('pen_bias_') for a in arms)
    comparison=root/'checkpoints/iam_codec_kl_study/20261007-010118'
    controls={}
    if protected:
        oldprov=json.loads((comparison/'kl1e-6/provenance.json').read_text())
        if oldprov['samples']!=prov['samples'] or oldprov['splits']!=prov['splits']:raise AssertionError('control sample/split drift')
        controls={a:json.loads((comparison/a/'eval-200.json').read_text()) for a in ('kl0','kl1e-6')}
    columns=2+len(arms)+len(controls)
    out=directory/'report';out.mkdir(exist_ok=True)
    def arrays(sid):
        return [('IAM/RDP target',targets[sid]),('protected source200',np.load(directory/arms[0]/f'step-0/{sid}/mu.npy'))]+[(a+' full-affine final200',np.load(comparison/a/f'step-200/{sid}/mu.npy')) for a in controls]+[(a+' final',np.load(directory/a/f'step-{results[a]["last_step"]}/{sid}/mu.npy')) for a in arms]
    pages=[]
    for group in ('old','new','held_out'):
        for page,ids in enumerate(paginate_ids(splits[group]),1):
            fig,axs=plt.subplots(len(ids),columns,figsize=(5*columns,2*len(ids)),squeeze=False)
            for j,sid in enumerate(ids):
                lo=targets[sid][:,:2].min(0);hi=targets[sid][:,:2].max(0)
                for ax,(label,a) in zip(axs[j],arrays(sid)):
                    draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' — '+label,fontsize=8)
            fig.tight_layout();name=f'{group}-{page}.png';fig.savefig(out/name,dpi=140);plt.close(fig);pages.append((group,page,name))
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axs=plt.subplots(columns,2,figsize=(8,3*columns),squeeze=False)
    for col,(sid,xlim,ylim,label) in enumerate(regions):
        for ax,(title,a) in zip(axs[:,col],arrays(sid)):
            draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(label+' — '+title,fontsize=9)
    fig.tight_layout();fig.savefig(out/'user-regions.png',dpi=170);plt.close(fig)
    galleries=[]
    for sid in splits['old']:
        fig,axs=plt.subplots(4,len(arms),figsize=(7*len(arms),7),squeeze=False);lo=targets[sid][:,:2].min(0);hi=targets[sid][:,:2].max(0)
        for col,arm in enumerate(arms):
            line=next(r for r in stages[arm]['lines'] if r['sample_id']==sid)
            ranked=sorted(line['sampled'],key=lambda v:v['geometry']['x_rmse']+v['geometry']['y_rmse'])
            rows=[('target',targets[sid]),('mean',np.load(directory/arm/f'step-{results[arm]["last_step"]}/{sid}/mu.npy'))]
            for label,v in [('median z',ranked[len(ranked)//2]),('worst z',ranked[-1])]:rows.append((label,np.load(directory/arm/f'step-{results[arm]["last_step"]}/{sid}/{v["kind"]}.npy')))
            for ax,(label,a) in zip(axs[:,col],rows):
                draw(ax,split_xy(a[:,:2],a[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(arm+' — '+label,fontsize=8)
        fig.tight_layout();name=sid+'.png';fig.savefig(out/name,dpi=140);plt.close(fig);galleries.append((sid,name))
    paired=[]
    if protected:
        previous=controls['kl1e-6']
        for line in previous['lines']:
            sid=line['sample_id']
            for v in line['sampled']:
                if v['pen']['pen_up_f1']==1 and v['pen']['final_eoc_correct'] and not v['pen']['non_final_false_eoc_count']:continue
                a=np.load(comparison/f'kl1e-6/step-200/{sid}/{v["kind"]}.npy')
                b=np.load(directory/arms[0]/f'step-{results[arms[0]]["last_step"]}/{sid}/{v["kind"]}.npy')
                indices=np.flatnonzero(a[:,2:].argmax(1)!=targets[sid][:,2:].argmax(1));x,y=targets[sid][indices[0],:2]
                fig,axs=plt.subplots(2,3,figsize=(15,5));lo=targets[sid][:,:2].min(0);hi=targets[sid][:,:2].max(0)
                for col,(title,array) in enumerate([('target',targets[sid]),('full-affine KL1e-6: failed draw',a),('pen-feature-weights frozen: SAME noise',b)]):
                    for row in range(2):
                        ax=axs[row,col];draw(ax,split_xy(array[:,:2],array[:,2:].argmax(1)));ax.set_title(title,fontsize=9)
                        if row==0:ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08)
                        else:ax.set_xlim(x-.12,x+.12);ax.set_ylim(y-.12,y+.12)
                fig.tight_layout();name=f'paired-{sid}-{v["kind"]}.png';fig.savefig(out/name,dpi=150);plt.close(fig);paired.append((sid,v['kind'],name))
    summary=dict(source_rel=SOURCE,source_sha256=SHA,results=results,initial=initial['groups'],matched_full_affine_controls={a:r['groups'] for a,r in controls.items()},final={a:r['groups'] for a,r in stages.items()},posterior_initial=before['groups'],posterior_final={a:p['groups'] for a,p in posterior.items()},cpu_checks=checks,provenance=prov)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    page=['<meta charset="utf-8"><title>Initialized codec: KL trade-off</title><h1>Matched KL continuations: preserve real curves, test posterior prior</h1>',
        '<p><strong>Engineering initialized transport, NOT a paper reproduction or generative-ready VAE.</strong> Every arm resumes the SAME protected update200 model, Adam moments/groups, CPU/CUDA RNG and shuffled training schedule. No geometry reset. Original arms change only KL: 0, 1e-6, 1e-5. A pen_bias follow-up instead freezes the24 routed pen logvar feature-weight rows while leaving their biases trainable; only those weight-row Adam moments are zeroed to prevent momentum drift. Its paired full-affine KL1e-6 reference uses the same source, update count, data and noise; KL0 is a second control. Body LR1e-7, posterior LR1e-3, readout/OCR/style fixed. Physical1/accum8, scale.01, dropout/rotation0, mean geometry + .204718 target index-difference matching + .1 sampled geometry + .0209905 bounded pen. GMM/CTC/style OFF. KL averages valid channel×time elements. Held-out32 seen-writer line-disjoint/forms overlap, evaluation only.</p>',
        '<p>FINAL endpoints shown even if step0 selected. Small positional/index errors are not physical velocity/curvature; tangent/turn are spatial angles. No generic smoothing, target corners remain. Posterior draws are fixed paired noise, 20 per line. Tiny KL compatibility is NOT evidence of an N(0,I) aggregate prior or useful semantic latents. DiT can learn a nonstandard latent distribution; N(0,I) is not imposed here as a generation-readiness requirement. Generation itself remains untested. Frozen OCR transfer is diagnostic, not a trained head or selection objective.</p>',
        '<a href="summary.json">Metrics, source hashes, CPU reload checks and provenance</a><table border="1"><tr><th>Stage/group</th><th>Mean X/Y RMSE</th><th>Δ/Δ² relative</th><th>turn p90</th><th>pen F1</th><th>sampled X/Y</th><th>sampled turn p90</th><th>internal EOCs</th></tr>']
    for label,row in [('source200',initial)]+[(a+' full-affine',r) for a,r in controls.items()]+list(stages.items()):
        for group in ('train','held_out','old'):
            s=row['groups'][group];m,z=s['mu'],s['sampled']
            page.append(f'<tr><td>{label}/{group}</td><td>{m["mean_per_line_x_rmse"]:.8f}/{m["mean_per_line_y_rmse"]:.8f}</td><td>{m["mean_per_line_first_difference_relative"]:.6f}/{m["mean_per_line_second_difference_relative"]:.6f}</td><td>{m["turn_angle_error_degrees"]["p90"]:.4f}°</td><td>{s["mu_macro_pen_f1"]:.6f}</td><td>{z["mean_per_line_x_rmse"]:.7f}/{z["mean_per_line_y_rmse"]:.7f}</td><td>{z["turn_angle_error_degrees"]["p90"]:.3f}°</td><td>{s["false_internal_eoc"]}</td></tr>')
    page.append('</table><h2>Posterior statistics (means of per-line summaries)</h2><table border="1"><tr><th>Stage/group</th><th>KL/element</th><th>XY std median</th><th>pen std median</th><th>unused std median</th><th>XY mean RMS</th></tr>')
    for label,p in [('source200',before)]+list(posterior.items()):
        for group in ('train','held_out'):
            s=p['groups'][group];page.append(f'<tr><td>{label}/{group}</td><td>{s["kl_per_element"]:.6f}</td><td>{s["xy"]["std_median"]:.7f}</td><td>{s["pen"]["std_median"]:.6f}</td><td>{s["unused"]["std_median"]:.6f}</td><td>{s["xy"]["mu_rms"]:.4f}</td></tr>')
    page.append('</table><h2>User-identified curves, marker-free</h2><img style="max-width:100%" src="user-regions.png">')
    for sid,kind,name in paired:page.append(f'<h2>Former false EOC — {sid}/{kind}, paired noise</h2><img style="max-width:100%" src="{name}">')
    for sid,name in galleries:page.append(f'<h2>{sid}: {html.escape(prov["samples"][sid]["text"])}</h2><img style="max-width:100%" src="{name}">')
    for group,num,name in pages:page.append(f'<h2>{group}, page{num}</h2><img style="max-width:100%" src="{name}">')
    (out/'index.html').write_text('\n'.join(page))
    from .report_pointer import publish_pointer
    publish_pointer(directory.parent,out)
    return dict(output=str(out/'index.html'),final={a:dict(step=r['last_step'],sha256=r['final_sha256']) for a,r in results.items()})


def tail_report(directory,root,audit):
    """Append all failed posterior draws, with identical-XY true-pen control."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import h5py
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);out=directory/'report';results=json.loads((directory/'result.json').read_text())
    notes=['<section id="posterior-tail-audit"><h2>Posterior tails and every final pen failure</h2>',
        '<p><a href="../posterior-tail-audit.json">Per-line tail statistics and every failing point</a>. Mean pen quality alone hides these failures. Routed std describes initializer channels, not an exact decoder-output variance; learned body cross-coupling remains possible.</p><ul>']
    for arm,a in audit['arms'].items():
        notes.append(f'<li>{arm}: {a["failing_draws"]}/4480 posterior draws have a pen/EOC error; maximum routed pen std on training lines {a["train_max_routed_pen_std"]:.5f}.</li>')
    notes.append('</ul>')
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5') as train,h5py.File(root/'diffink/iam_overfit/tiny_val.h5') as val:
        for arm,auditarm in audit['arms'].items():
            for case in auditarm['failures']:
                sid=case['sample_id'];kind=case['kind'];hf=train if sid in train else val
                target=hf[sid]['point_seq'][:].copy();target[:,:2]*=.01
                a=np.load(directory/arm/f'step-{results[arm]["last_step"]}/{sid}/{kind}.npy');same=a.copy();same[:,2:]=target[:,2:]
                mean=np.load(directory/arm/f'step-{results[arm]["last_step"]}/{sid}/mu.npy')
                columns=[('target',target),('mean + predicted pens',mean),('sampled XY + TRUE pens',same),('same sampled XY + predicted pens',a)]
                fig,axs=plt.subplots(2,4,figsize=(16,6));lo=target[:,:2].min(0);hi=target[:,:2].max(0)
                first=case['points'][0];x,y=first['target_xy']
                for col,(label,array) in enumerate(columns):
                    for row in range(2):
                        ax=axs[row,col];draw(ax,split_xy(array[:,:2],array[:,2:].argmax(1)));ax.set_title(label,fontsize=9)
                        if row==0:ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08)
                        else:ax.set_xlim(x-.12,x+.12);ax.set_ylim(y-.12,y+.12)
                fig.tight_layout();name=f'failure-{arm}-{sid}-{kind}.png';fig.savefig(out/name,dpi=150);plt.close(fig)
                notes.append(f'<h3>{arm} — {sid}, {kind}</h3><p>Point{first["point_index"]}: state{first["target_state"]}→{first["predicted_state"]}; routed continue/up/EOC std {first["routed_pen_std"]}. Crop location selected by the actual failed point, no markers or smoothing.</p><img style="max-width:100%" src="{name}">')
    notes.append('</section>')
    path=out/'index.html';page=path.read_text();tag='<section id="posterior-tail-audit">'
    if tag in page:page=page[:page.index(tag)]
    tmp=out/'index.tail.tmp';tmp.write_text(page+'\n'+'\n'.join(notes));tmp.replace(path)
