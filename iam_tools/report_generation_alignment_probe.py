"""Weak forced-reader alignment gallery, not exact IAM character segmentation."""
import html,json
from pathlib import Path
import h5py
import numpy as np
from .generation_alignment_probe import quantiles
from .pen_ab import file_sha


def report(directory, root='data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .generation_capacity import DATA
    from .curve_audit import draw,split_xy
    p=Path(directory);s=json.loads((p/'summary.json').read_text());f=p/'alignment.h5'
    if file_sha(f)!=s['alignment_h5_sha256']:raise ValueError('immutable packed weak-alignment guard')
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());pages=[]
    fixed=['k04-309z-09','c03-109z-03','r07-568z-04','a02-130z-03','d08-586z-01','p10-249z-01','e07-425z-02','n05-514z-02']
    with h5py.File(f) as alignment,h5py.File(Path(root)/DATA/'source.h5') as source:
        for sid in fixed:
            row=next(r for r in s['lines'] if r['sample_id']==sid);q=alignment[sid];points=source[sid]['target'][:]
            token=q['forced_token_indices'][:];expected=q['last_attention_expected_char_index'][:];valid=token>=0;frames=np.arange(len(token));blocks=frames//2
            fig,ax=plt.subplots(3,1,figsize=(17,9),gridspec_kw={'height_ratios':[1,1,1.4]})
            draw(ax[0],split_xy(points[:,:2],points[:,2:].argmax(1)));ax[0].set_title(sid+' / actual processed source (no generation)\n'+row['text'])
            ax[1].scatter(4*frames[valid]+1.5,token[valid],s=9,label='forced CTC nonblank token index')
            ax[1].plot(4*frames+1.5,q['prior_center'][:],label='static Gaussian center',color='black')
            ax[1].plot(4*frames+1.5,expected[:3,blocks].mean(0),label='final3local attention conditional-character mean',color='red',alpha=.8)
            ax[1].plot(4*frames+1.5,expected[3,blocks],label='final global head mean',alpha=.6)
            ax[1].set_xlabel('processed point INDEX (not physical time)');ax[1].set_ylabel('character token INDEX');ax[1].legend(fontsize=8)
            im=ax[2].imshow(q['last_attention'][:3].mean(0).T,aspect='auto',origin='lower',interpolation='nearest',cmap='magma')
            ax[2].set_ylabel('text token INDEX (BOS excluded)');ax[2].set_xlabel('packed8 query INDEX');fig.colorbar(im,ax=ax[2],label='mean local head weight')
            fig.tight_layout();name=sid+'.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append(name)
    per_writer={}
    for writer in sorted({r['writer_id'] for r in s['lines']}):
        rows=[r for r in s['lines'] if r['writer_id']==writer]
        per_writer[writer]=dict(lines=len(rows),mean_prior_p90=float(np.mean([r['prior_token_index_error']['p90'] for r in rows])),mean_prior_outside_sigma=float(np.mean([r['prior_outside_sigma_fraction'] for r in rows])))
    (out/'per-writer.json').write_text(json.dumps(per_writer,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Weak character alignment: static prior versus reader timing</title><style>body{font:17px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}</style><h1>TRAIN-only weak alignment probe</h1><p>Frozen absolute-query48000 model, all256TRAIN source trajectories, frozen corpus-familiar BiGRU reader. No model training/selection/held-source opening. All256source transcripts read exactly. Forced CTC timing is WEAK: reader lookahead, delayed marks, blank emissions and nonuniform trajectory spacing prevent treating it as actual IAM character boundaries.</p><p>Across forced nonblank frames, compare text token indices, not XY reconstruction or physical time. '+f'{100*s["aggregate"]["prior_outside_sigma_fraction"]:.2f}%'+ ' differ from the static prior center by more than its2.5-character sigma. The final local-head conditional-character mean does not demonstrate a cleaner alignment. This supports investigating learned/weakly supervised alignment, but DOES NOT establish attention is causally wrong: multihead means can hide useful routing.</p><p><a href="../summary.json">All256line statistics/provenance</a> · <a href="per-writer.json">All32writer summary</a> · <a href="../alignment.h5">Packed paths/attention arrays</a></p><pre>'+html.escape(json.dumps(s['aggregate'],indent=2))+'</pre><p>Eight fixed previously declared TRAIN lines shown, not selected by output quality. Marker-free source render above diagnostic timing plots. Orange/red attention features are explanations, not predicted handwriting or smoothing.</p>'
    for name in pages:body+='<img loading="lazy" src="'+name+'">'
    (out/'index.html').write_text(body);return str(out/'index.html')
