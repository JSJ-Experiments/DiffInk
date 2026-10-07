"""All-line marker-free continuation comparisons and independentCPUsampler check."""
import json,html
from pathlib import Path
import h5py,numpy as np,torch
from .generation_study import collate,decode_sample
from .generation_refinement import ARMS,PARENT,PARENT_SHA
from .latent_diffusion import TextLatentDenoiser,transform,cosine_schedule,ddim_sample
from .writer_expansion import load
from .ocr_joint_adapter import load_reader
from .kl_tradeoff import SOURCE,SHA
from .pen_ab import file_sha
from .report_resources import report as resource_report


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);parent=root/PARENT;cfg=json.loads((directory/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());results=json.loads((directory/'result.json').read_text())
    if file_sha(parent/'text/checkpoint-last.pt')!=PARENT_SHA or file_sha(parent/'source.h5')!=cfg['source_h5_sha256'] or file_sha(parent/'whitening.pt')!=cfg['whitening_sha256']:raise ValueError('immutable parent/cache drift')
    out=directory/'report';out.mkdir(exist_ok=True);resource_report(directory)
    (out/'index.html').replace(out/'resources.html');(out/'report-source.py').replace(out/'resource-report-source.py')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    logs={a:[json.loads(l) for l in (directory/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS}
    if any(len(logs[a])!=results[a]['last_step'] for a in ARMS):raise ValueError('incomplete arm')
    if any(x['sample_ids']!=y['sample_ids'] or x.get('noise_rng_sha256')!=y.get('noise_rng_sha256') for a in ARMS[1:] for x,y in zip(logs[ARMS[0]],logs[a])):raise AssertionError('schedule/RNG pairing failure')
    if len({r['final_noise_rng_sha256'] for r in results.values()})!=1 or not all(r['codec_reader_unchanged'] for r in results.values()):raise AssertionError('frozen/RNG guard failure')
    evaluations={};source={};latents={};stats=torch.load(parent/'whitening.pt',weights_only=True)
    with h5py.File(parent/'source.h5') as f:
        for sid in data['records']:source[sid]=f[sid]['target'][:];latents[sid]=torch.tensor(f[sid]['latent_mean'][:])
    labels=['parent8000']+list(ARMS);files={'parent8000':parent/'text/evaluation-8000.h5'}
    evaluations['parent8000']=json.loads((parent/'text/eval-8000.json').read_text())
    for a,r in results.items():
        if file_sha(directory/a/'checkpoint-last.pt')!=r['last_sha256']:raise ValueError('checkpoint drift')
        files[a]=directory/a/f'evaluation-{r["last_step"]}.h5';evaluations[a]=json.loads((directory/a/f'eval-{r["last_step"]}.json').read_text())
        if file_sha(files[a])!=evaluations[a]['packed_h5_sha256']:raise ValueError('evaluation drift')
    aggregate={}
    for label,ev in evaluations.items():
        aggregate[label]={}
        for split,ids in data['splits'].items():
            rows=[r for r in ev['lines'] if r['sample_id'] in ids and r['policy']=='correct']
            def avg(key,sub=None):return float(np.mean([r['geometry'][key][sub] if sub else r['geometry'][key] for r in rows]))
            aggregate[label][split]=dict(**ev['groups'][split]['correct'],x_rmse=avg('x_rmse'),y_rmse=avg('y_rmse'),segment_vector_rmse=avg('first_difference','vector_rmse'),second_difference_vector_rmse=avg('second_difference','vector_rmse'),tangent_p90_mean=avg('tangent_angle_error_degrees','p90'),turn_p90_mean=avg('turn_angle_error_degrees','p90'),corner_turn_p90_mean=avg('target_corner_turn_error_degrees','p90'),pen_f1_min=min(r['pen_aligned_reference']['pen_up_f1'] for r in rows),nonfinal_false_eoc=sum(r['pen_aligned_reference']['non_final_false_eoc_count'] for r in rows))
    # IndependentCPUfull-noise resampling with actualGPUepsilon,2TRAIN/2unseen.
    torch.set_num_threads(4);codec,_,_,cc,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None);codec.eval().requires_grad_(False);reader,_=load_reader(root,cc)
    ids=data['splits']['train'][:2]+data['splits']['unseen_prompt'][:2];_,mask,text=collate(latents,data['records'],cfg['vocab'],ids,stats,'cpu');checks={}
    for a,r in results.items():
        model=TextLatentDenoiser(**cfg['model']);saved=torch.load(directory/a/'checkpoint-last.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model_state_dict']);model.eval()
        epsilon=torch.zeros(mask.shape[0],mask.shape[1],384)
        with h5py.File(files[a]) as f:
            for j,sid in enumerate(ids):epsilon[j,:len(latents[sid])]=torch.tensor(f[f'correct/9142/{sid}/initial_noise'][:])
            sample=transform(ddim_sample(model,epsilon,text,mask,cosine_schedule(),steps=50),stats,True);rows=[]
            for j,sid in enumerate(ids):
                points,metrics=decode_sample(codec,reader,sample[j,:len(latents[sid])],data['records'][sid],cfg['vocab']);q=f[f'correct/9142/{sid}'];gpu=q['points'][:];row=json.loads(q.attrs['row'])
                rows.append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-gpu[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=gpu[:,2:].argmax(1)).sum()),free_transcript_equal=metrics['free_decoded']==row['free_decoded']))
        checks[a]=rows
    pages=[]
    for split,ids in data['splits'].items():
        for seed in ([9142] if split=='train' else [9142,9143]):
            for start in range(0,len(ids),4):
                group=ids[start:start+4];fig,axs=plt.subplots(len(group),5,figsize=(25,3*len(group)),squeeze=False)
                for j,sid in enumerate(group):
                    draw(axs[j,0],split_xy(source[sid][:,:2],source[sid][:,2:].argmax(1)));axs[j,0].set_title(sid+' | IAM/RDP\n'+data['records'][sid]['text'],fontsize=8)
                    for ax,label in zip(axs[j,1:],labels):
                        with h5py.File(files[label]) as f:
                            q=f[f'correct/{seed}/{sid}'];p=q['points'][:];row=json.loads(q.attrs['row']);p=p[:row['generated_points_at_stop']]
                        draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_title(label+' | predicted stop\nreader: '+row['free_decoded'],fontsize=8)
                fig.tight_layout();name=f'{split}-seed{seed}-{start//4+1}.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append(name)
    summary=dict(config=cfg,results=results,aggregate=aggregate,independent_cpu_sampler=checks,paired_rng_schedule=True,codec_reader_frozen=True,
                 metric_note='TRAIN paired point/segment/angle errors measure memorization. Held-out reference trajectory is not the unique valid writing. Difference metrics use nonuniform index spacing, not physical velocity or curvature. Means of per-line p90 angles, not pooled p90.',
                 limitations='32TRAIN,onewriter,oracle duration,held8generator prompts but reader-familiar corpus. No production promotion. Final steps shown even if bestTRAINdenoising selector differs; no unseen selection.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    body='<!doctype html><meta charset="utf-8"><title>Frozen codec generation geometry refinement</title><style>body{font:16px system-ui;max-width:1800px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style><h1>Controlled generation continuation: objective alignment</h1><p>Same step8000source,restoredAdam/RNG,minibatches,frozenfaithfulcodec/reader,LR1e-4,2000steps. A:originalwhitenedMSE; B:+physicalXY; C:+sameXY+targetsegmentmatching. Coefficients calibrated to25%/10% base full-model gradient. This is NOT generic smoothing. Oracle duration remains; no generation success is inferred from denoising loss.</p><p><a href="summary.json">All metrics/config/CPU reload</a> · <a href="../config.json">As-run config</a> · <a href="../source-code/">As-run source directory</a> · <a href="resources.html">Resource measurements/alerts</a></p><table><tr><th>Model</th><th>Split</th><th>FreeCER</th><th>X/YRMSE</th><th>segment / second diff</th><th>mean tangent/turnp90°</th><th>minpenF1</th></tr>'
    for label,splits in aggregate.items():
        for split,r in splits.items():
            body+=f'<tr><td>{label}</td><td>{split}</td><td>{100*r["free_cer"]:.2f}%</td><td>{r["x_rmse"]:.5f} / {r["y_rmse"]:.5f}</td><td>{r["segment_vector_rmse"]:.5f} / {r["second_difference_vector_rmse"]:.5f}</td><td>{r["tangent_p90_mean"]:.1f} / {r["turn_p90_mean"]:.1f}</td><td>{r["pen_f1_min"]:.3f}</td></tr>'
    body+='</table><p>All TRAIN32 and held-form8 shown, not handpicked. Real predicted pen states and first-EOC stop, no true boundaries. Both held-out noise seeds shown. Held-out point matching is not a generation-quality criterion; text readability matters.</p>'
    for name in pages:body+='<h2>'+name+'</h2><img loading="lazy" src="'+name+'">'
    body+='<h2>CPU sampler checks</h2><pre>'+html.escape(json.dumps(checks,indent=2))+'</pre>'
    (out/'index.html').write_text(body);return dict(report=str(out/'index.html'),aggregate=aggregate,cpu=checks)
