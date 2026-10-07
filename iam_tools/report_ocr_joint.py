"""Independent selected-codec reload, geometry/OCR audit and marker-free gallery."""
import hashlib,html,inspect,json,os
from pathlib import Path
import h5py,numpy as np,torch
from torch.nn import functional as F
from .ocr_joint_study import ARMS,evaluate,gate,SOURCE,SHA
from .ocr_joint_adapter import load_reader,READER_REL,READER_SHA
from .ocr_pool_study import load_pool,single_batch
from .ocr_convergence import POOL_SHA
from .writer_expansion import load
from .pen_ab import file_sha
from .ocr_context_study import tensor_digest
from .report_pointer import publish_pointer
from .latent_integration import encoded
from .inkvae import greedy_ctc,edit_distance


def cached_cpu_evaluation(model,batches,records,vocab,splits,folder,step,draws,save):
    """Checkpoint/source/data-bound completed stage; safe after CPU preemption.

    No training cache. Fingerprint exact weights, point bytes, texts, masks,
    version, reader stats, relevant source bytes and draw/seed policy.
    """
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    ids=sorted(set(i for values in splits.values() for i in values))
    source_files=[Path(__file__),Path(__file__).with_name('ocr_joint_study.py'),Path(__file__).with_name('ocr_joint_adapter.py'),Path(__file__).with_name('ocr_recurrent.py'),Path(__file__).with_name('ocr_context_features.py')]
    model_file=Path(inspect.getfile(type(model)))
    source_files.extend(p for p in (model_file,model_file.with_name('blocks.py'),model_file.with_name('losses.py')) if p.exists())
    sources={p.name:file_sha(p) for p in source_files}
    fp=dict(model=tensor_digest(model.state_dict()),stats=model.ocr_model.stats,sources=sources,
      points={i:hashlib.sha256(batches[i][0].cpu().numpy().tobytes()).hexdigest() for i in ids},
      records={i:records[i] for i in ids},splits=splits,step=step,draws=draws,save_trajectories=save,
      vocab=list(vocab),masks={i:batches[i][1].tolist() for i in ids},labels={i:batches[i][2].tolist() for i in ids},
      torch_version=str(torch.__version__),threads=torch.get_num_threads(),device='cpu',seed_policy='8042+sortedIDindex*100')
    key=hashlib.sha256(json.dumps(fp,sort_keys=True).encode()).hexdigest();cache=folder/'completed-cpu-evaluation.json'
    if cache.exists():
        saved=json.loads(cache.read_text())
        if saved['fingerprint']==key:
            row=saved['result']
            if {r['sample_id'] for r in row['lines']}!=set(ids):raise ValueError('incomplete cached CPU stage')
            if save:
                for sid in ids:
                    for name in ['mu']+[f'z-{j}' for j in range(draws)]:
                        if not (folder/f'step-{step}'/sid/(name+'.npy')).is_file():raise ValueError('cached trajectory missing')
            print('Reused completed bound CPU stage',str(folder),flush=True);return row
    row=evaluate(model,batches,records,vocab,splits,folder,step,draws=draws,save=save)
    temp=folder/'completed-cpu-evaluation.tmp'
    temp.write_text(json.dumps(dict(fingerprint=key,contract=fp,result=row),indent=2)+'\n');os.replace(temp,cache)
    if str(folder).startswith('/data/'):
        import modal
        modal.Volume.from_name('diffink-data').commit()
    return row


def pack_rendered_points(xy,states,channels):
    """Chronological B,N,2 and hard predicted pens→original B,C,T8 fields."""
    if xy.ndim!=3 or xy.shape[-1]!=2 or xy.shape[1]%8 or states.shape!=xy.shape[:2] or channels<40:
        raise ValueError('matching multiple-of-eight XY/pen and>=40 channels required')
    points=torch.cat((xy,F.one_hot(states,3).to(xy.dtype)),dim=2)
    fields=points.reshape(xy.shape[0],-1,8,5).permute(0,2,3,1).reshape(xy.shape[0],40,-1)
    return torch.cat((fields,xy.new_zeros(xy.shape[0],channels-40,fields.shape[-1])),dim=1)


def rebuild(folder,repo,root,configs,which='checkpoint-best.pt'):
    model,_,_,cfg,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
    if folder is not None:
        saved=torch.load(folder/which,map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        model.apply_checkpoint_contract(saved,allow_research_ocr=True)
    if cfg!=configs:raise ValueError('codec configuration drift')
    model.ocr_model,_=load_reader(root,cfg);model.eval().requires_grad_(False)
    return model


def legacy_eight(directory,repo,root,cfg,vocab,results,out):
    """Supplement ALL original curve examples, including four outside reader pool.

    No new training, stop or selection decision. Same CPU seeds for all versions.
    The eight were previously used for codec development; not a fresh test set.
    """
    from .writer_expansion import MANIFEST_SHA
    from .ocr_joint_study import summarize
    from .report_writer_expansion import paginate_ids
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    ids=['c08-434z-05','e08-429z-04','p08-936z-05','a07-421z-03','k07-640z-02','l10-072z-02','h05-195z-03','a07-421z-02']
    if file_sha(root/'diffink/iam_overfit/manifest.json')!=MANIFEST_SHA:raise ValueError('legacy eight source manifest drift')
    batches={};records={};targets={}
    with h5py.File(root/'diffink/iam_overfit/tiny_train.h5') as hf:
        for sid in ids:
            g=hf[sid];points=g['point_seq'][:];text=g['line_text'][()].decode()
            records[sid]=dict(text=text,writer_id=g['writer_id'][()].decode())
            batches[sid]=single_batch(points,text,vocab);targets[sid]=points.copy();targets[sid][:,:2]*=.01
    d=directory/'diagnostics/legacy-eight';d.mkdir(parents=True,exist_ok=True);rows={};gates={}
    versions={'source':(None,0,'checkpoint-best.pt')}
    versions.update({a:(directory/a,results[a]['best_step'],'checkpoint-best.pt') for a in ARMS})
    versions['geometry_ocr_last']=(directory/ARMS[1],results[ARMS[1]]['last_step'],'checkpoint-last.pt')
    for name,(folder,step,which) in versions.items():
        model=rebuild(folder,repo,root,cfg,which);rows[name]=cached_cpu_evaluation(model,batches,records,vocab,dict(legacy_eight=ids),d/name,step,draws=20,save=True)
        if name!='source':gates[name]=gate(rows[name],rows['source'],ids)
        del model
    result=dict(ids=ids,records=records,manifest_sha256=MANIFEST_SHA,draws=20,
      versions={k:v['groups'] for k,v in rows.items()},gates=gates,
      caveat='Four original eight absent from8192 reader pool; supplemental legacy-h5 CPU audit only, never GPUselection/stop/training. Previously codec-trained/development examples, not independent test. Same20CPU seeds across these versions, not paired to GPU noise.')
    (d/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    pages=[]
    for j,page in enumerate(paginate_ids(ids,limit=4),1):
        fig,axes=plt.subplots(len(page),4,figsize=(22,3*len(page)),squeeze=False)
        for r,sid in enumerate(page):
            variants=[targets[sid]];titles=['IAM/RDP target']
            for a in ('source',*ARMS):
                step=versions[a][1];variants.append(np.load(d/a/f'step-{step}'/sid/'mu.npy'));titles.append(a)
            lo,hi=targets[sid][:,:2].min(0),targets[sid][:,:2].max(0)
            for ax,p,title in zip(axes[r],variants,titles):
                draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=9)
        fig.tight_layout();name=f'legacy-eight-{j}.png';fig.savefig(out/name,dpi=140);plt.close(fig);pages.append(name)
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')]
    fig,axes=plt.subplots(2,4,figsize=(18,9))
    for r,(sid,xlim,ylim,label) in enumerate(regions):
        points=[targets[sid]]+[np.load(d/a/f'step-{versions[a][1]}'/sid/'mu.npy') for a in ('source',*ARMS)]
        for ax,p,title in zip(axes[r],points,['target','source',*ARMS]):
            draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(label+' | '+title)
    fig.tight_layout();fig.savefig(out/'named-curve-closeups.png',dpi=170);plt.close(fig)
    return result,pages


def latent_vs_drawing(model,batches,records,vocab,ids):
    """Does lower latent-reader CTC actually change reading of rendered XY/pen?"""
    from model.losses import mixture_expectation
    from .ocr_joint_study import local_geometry
    rows=[]
    for sid in ids:
        raw,pm,label=batches[sid]
        with torch.no_grad():
            _,mu,_,lm=encoded(model,raw,pm);output=model.decode(mu,padding_mask=~pm)
            xy=mixture_expectation(output);states=output[:,:3].argmax(1)
            rendered=pack_rendered_points(xy,states,mu.shape[1])
            losses={k:float(model.get_ocr_loss(x,label,lm,point_mask=pm)) for k,x in [('latent',mu),('rendered',rendered)]}
            texts={k:greedy_ctc(model.ocr_model(x,padding_mask=~lm,point_mask=pm)[:int((pm.sum()+3)//4),0].argmax(-1).tolist(),vocab) for k,x in [('latent',mu),('rendered',rendered)]}
            pens=(mu[:,:40]-rendered[:,:40]).reshape(1,8,5,-1).permute(0,3,1,2).reshape(1,-1,5)
            discrepancy=float(pens[:,:,2:][pm].square().mean())
        z=mu.detach().requires_grad_();loss=model.get_ocr_loss(z,label,lm,point_mask=pm)
        gradient=torch.autograd.grad(loss,z)[0]
        c=torch.arange(z.shape[1])[None,:,None];energy=gradient.square()
        total=float(energy.sum());fractions={k:float(energy.masked_select(mask.expand_as(energy)).sum())/max(total,1e-30) for k,mask in [('xy',(c<40)&(c%5<2)),('pen',(c<40)&(c%5>=2)),('unused',c>=40)]}
        rows.append(dict(sample_id=sid,text=records[sid]['text'],ctc=losses,decoded=texts,
          errors={k:edit_distance(records[sid]['text'],text) for k,text in texts.items()},characters=len(records[sid]['text']),
          latent_pen_vs_hard_decoded_pen_mse=discrepancy,input_gradient_energy_fractions=fractions))
    return dict(lines=rows,groups={k:dict(cer=sum(r['errors'][k] for r in rows)/sum(r['characters'] for r in rows),mean_ctc_loss=float(np.mean([r['ctc'][k] for r in rows]))) for k in ('latent','rendered')},
      explanation='Rendered reader gets actual mixture-expected XY and predicted hard pen states repacked through the SAME frozen input adapter; no targetpen oracle. Mean-only CPU diagnostic, no selection. Pen amplitudes can encode cues invisible to pen argmax/geometry; gradient energy fractions diagnose this, not causal attribution.')


def pairing(configs,results,logs):
    a,b=ARMS;c,d=configs[a],configs[b]
    for k in ('source_sha256','reader_sha256','pool_manifest_sha256','cfg','splits','train_ids','schedule_sha256','calibration','optimizer_groups','max_updates','physical_batch','gradient_accumulation','mean_geometry_weight','sampled_geometry_weight','pen_weight','kl_weight','style_weight','gmm_weight'):
        if c[k]!=d[k]:raise ValueError('uncontrolled joint difference: '+k)
    checks=dict(same_batches=[r['sample_ids'] for r in logs[a]]==[r['sample_ids'] for r in logs[b]],
      same_latent_noise=[r['noise_sha256'] for r in logs[a]]==[r['noise_sha256'] for r in logs[b]],
      equal_budget=all(r['last_step']==c['max_updates'] for r in results.values()),
      frozen_reader=all(r['reader_weights_buffers_bitwise_unchanged'] for r in results.values()),
      frozen_sources=all(r['source_unchanged'] and r['reader_source_unchanged'] for r in results.values()),
      protected_readout_style_pen_variance=all(r['readout_style_pen_variance_protected'] for r in results.values()))
    # Early TRAIN geometry stopping is valid negative evidence, not pairing proof.
    checks['matched_prefix_batches']=all(x['sample_ids']==y['sample_ids'] and x['noise_sha256']==y['noise_sha256'] for x,y in zip(logs[a],logs[b]))
    if not all(checks[k] for k in ('matched_prefix_batches','frozen_reader','frozen_sources','protected_readout_style_pen_variance')):raise ValueError('joint immutable/matched-prefix guard failed')
    if c['ctc_weight']!=0 or d['ctc_weight']!=d['calibration']['ctc_weight'] or d['ctc_weight']<=0:raise ValueError('explicit calibrated A/B objective required')
    return checks


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    from .report_writer_expansion import paginate_ids
    directory=Path(directory);root=Path(root);torch.set_num_threads(4)
    results=json.loads((directory/'result.json').read_text());configs={a:json.loads((directory/a/'config.json').read_text()) for a in ARMS}
    logs={a:[json.loads(l) for l in (directory/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS};paired=pairing(configs,results,logs)
    pool,m,vocab=load_pool(root,POOL_SHA);cfg=configs[ARMS[0]]['cfg'];splits=configs[ARMS[0]]['splits'];ids=set(i for values in splits.values() for i in values)
    if (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise ValueError('pool bytes drift')
    batches={};targets={}
    with h5py.File(pool/'lines.h5') as hf:
        for sid in sorted(ids):
            p=hf[sid]['point_seq'][:];targets[sid]=p.copy();targets[sid][:,:2]*=.01
            batches[sid]=single_batch(p,m['records'][sid]['text'],vocab)
    out=directory/'report';out.mkdir(exist_ok=True);(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    checks={};gpu={};cpu={};gates={}
    for a in ARMS:
        folder=directory/a;r=results[a];saved=torch.load(folder/'checkpoint-best.pt',map_location='cpu',weights_only=True)
        if file_sha(folder/'checkpoint-best.pt')!=r['selected_sha256'] or saved['continuation_updates']!=r['best_step'] or saved['config']!=configs[a]:raise ValueError('selected codec drift')
        model,_,_,actual_cfg,_,_=load(Path(repo)/'configs/engineering_english.yaml',repo,root,SOURCE,SHA,writer_id=None)
        if actual_cfg!=cfg:raise ValueError('model config drift')
        model.load_state_dict(saved['model_state_dict'],strict=True);model.apply_checkpoint_contract(saved,allow_research_ocr=True)
        model.ocr_model,_=load_reader(root,cfg);model.eval().requires_grad_(False);before=tensor_digest(model.ocr_model.state_dict())
        cpu[a]=cached_cpu_evaluation(model,batches,m['records'],vocab,splits,out/('cpu-'+a),r['best_step'],draws=0,save=False)
        gpu[a]=json.loads((folder/f'eval-{r["best_step"]}.json').read_text());reference=json.loads((folder/'eval-0.json').read_text())
        diffs=[]
        with torch.no_grad():
            from model.losses import mixture_expectation
            for sid in sorted(ids):
                raw,pm,_=batches[sid];_,mu,_,_=encoded(model,raw,pm)
                output=model.decode(mu,padding_mask=~pm);n=int(pm.sum());xy=mixture_expectation(output)[0,:n].numpy();p=output[0,:3,:n].argmax(0).numpy()
                saved_points=np.load(folder/f'step-{r["best_step"]}'/sid/'mu.npy')
                diffs.append(dict(sample_id=sid,max_xy_difference=float(np.abs(xy-saved_points[:,:2]).max()),pen_mismatches=int((p!=saved_points[:,2:].argmax(1)).sum())))
        original={x['sample_id']:x for x in gpu[a]['lines']}
        transcripts=[x['sample_id'] for x in cpu[a]['lines'] if x['mu']['decoded']!=original[x['sample_id']]['mu']['decoded']]
        checks[a]=dict(max_cpu_gpu_xy_difference=max(x['max_xy_difference'] for x in diffs),pen_mismatches=sum(x['pen_mismatches'] for x in diffs),
          mean_transcript_differences=transcripts,reader_bitwise_unchanged=tensor_digest(model.ocr_model.state_dict())==before,
          cpu_scope='independent selected196MEANS; GPU20draws/all196 alreadygated. Supplemental legacy8 uses20paired CPUdraws; CPU/CUDA random streams not paired.',lines=diffs)
        if checks[a]['max_cpu_gpu_xy_difference']>1e-4 or checks[a]['pen_mismatches'] or not checks[a]['reader_bitwise_unchanged']:raise AssertionError('CPU codec reload mismatch')
        gates[a]=gate(gpu[a],reference,list(ids));del model
    legacy,legacy_pages=legacy_eight(directory,repo,root,cfg,vocab,results,out)
    diagnostics={}
    for a in ('source',*ARMS):
        model=rebuild(None if a=='source' else directory/a,repo,root,cfg)
        diagnostics[a]=latent_vs_drawing(model,batches,m['records'],vocab,splits['train_probe']+splits['dev'])
        del model
    (directory/'diagnostics/latent-vs-rendered-reader.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    summary=dict(results=results,configs=configs,paired=paired,cpu_reload=checks,cpu_groups={a:r['groups'] for a,r in cpu.items()},
      selected_gpu_groups={a:r['groups'] for a,r in gpu.items()},selected_gpu_all_reported_gates=gates,
      selected_scope='TRAIN32probe+DEV128+report32+4 pool curve cases =196GPU/CPUselected lines; supplemental8legacy CPU/source/selected/final audit separately. Notfull8352post-update geometrygate.',
      source_sha256=SHA,reader_sha256=READER_SHA,legacy_eight=legacy,
      latent_vs_rendered_reader={a:r['groups'] for a,r in diagnostics.items()},
      caveats='Initialized transport compatibility, not normal semanticVAE/generation readiness. Frozen reader trained8192; TRAIN32probe selection only, DEV/report neverchoose. Sourcecodec priorreport promptoverlap. No KL/style/GMM. CTC can alter encoder without improving visibledecodedgeometry; strict gates required.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(16,4))
    for a,r in results.items():
        for ax,key in zip(axes,('cer','mean_per_line_turn_p90','x_rmse')):
            ax.plot([h['step'] for h in r['history']],[h['groups']['dev']['mu'][key] for h in r['history']],label=a,marker='.')
            ax.set_title('DEV '+key);ax.set_xlabel('updates');ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    pages=[]
    for group in ('named','dev','held_out'):
        for j,page_ids in enumerate(paginate_ids(splits[group],limit=6),1):
            fig,axes=plt.subplots(len(page_ids),4,figsize=(22,2.8*len(page_ids)),squeeze=False)
            maps={a:{r['sample_id']:r for r in gpu[a]['lines']} for a in ARMS}
            for row,sid in enumerate(page_ids):
                target=targets[sid];reference=np.load(directory/ARMS[0]/'step-0'/sid/'mu.npy')
                variants=[target,reference]+[np.load(directory/a/f'step-{results[a]["best_step"]}'/sid/'mu.npy') for a in ARMS]
                titles=['IAM/RDP target','source reconstruction']+[a+'; OCR: '+maps[a][sid]['mu']['decoded'] for a in ARMS]
                lo,hi=target[:,:2].min(0),target[:,:2].max(0)
                for ax,p,title in zip(axes[row],variants,titles):
                    draw(ax,split_xy(p[:,:2],p[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=8)
            fig.tight_layout();name=f'{group}-{j}.png';fig.savefig(out/name,dpi=130);plt.close(fig);pages.append((group,name))
    chunks=['<meta charset="utf-8"><title>Geometry plus frozen English GRU</title><style>body{font:17px system-ui;max-width:1500px;margin:2rem auto}img{max-width:100%}pre{white-space:pre-wrap}td,th{padding:.5em}</style>',
      '<h1>Geometry/pen versus geometry/pen + frozen OCR</h1><p>Left to right: IAM/RDP target, original source reconstruction, selected geometry-only, selected geometry+OCR. No rendering smoothing. Actual input lengths control the reader—not predicted EOC. Compare curves and captions. Source geometry starts near-lossless; unchanged appearance alone is compatibility, not a new drawing improvement.</p>',
      '<p>Same immutable initialized-polyphase40 codec, frozen seed42 GRU, pool, source optimizer moments, physical encoder batch1/accumulation8, schedule4042 and fixed latent noise. G=point MSE+0.20471838744633777 target first-difference MSE within true strokes; objective G(mu)+0.1G(z)+bounded pen. B adds sampled CTC at a TRAIN-only gradient-calibrated weight. KL/style/GMM/dropout off. Frozen readout/style and protected pen variance. Not ordinary semantic VAE or paper reproduction.</p>',
      '<h2>Results and gates</h2><pre>'+html.escape(json.dumps(dict(pairing=paired,selected_gates=gates,selected_gpu_groups=summary['selected_gpu_groups']),indent=2))+'</pre>',
      '<p>Checkpoint selection uses TRAIN32 probe CER/CTC only among passing geometry/pen evaluations. DEV/report never select or stop. Early TRAIN geometry failures stop safely and remain visible in history. Generated galleries cover all reported lines; no newfull8352post-update gate or generic latent contract claimed.</p>',
      '<img src="learning.png"><p><a href="summary.json">Exact configs, metrics, SHA hashes, guards and CPU reload</a></p>']
    chunks.append('<h2>Original eight curve examples: supplemental CPU audit</h2><p>Four are outside the8192 pool, including a07-421z-02. All eight are reloaded from their original immutable overfit dataset, never used to choose/stop this GPU experiment. Same20CPU draws compare source/selected/final codecs. Previously used for codec development, not fresh test examples.</p><pre>'+html.escape(json.dumps(legacy,indent=2))+'</pre><img src="named-curve-closeups.png">')
    for name in legacy_pages:chunks.append('<img loading="lazy" src="'+name+'">')
    chunks.append('<h2>Latent reading versus actual reconstructed trajectory</h2><pre>'+html.escape(json.dumps(summary['latent_vs_rendered_reader'],indent=2))+'</pre><p>This diagnostic detects improvement confined to latent pen amplitudes rather than actual rendered handwriting. No target-pen oracle or checkpoint selection.</p>')
    for group,name in pages:chunks.append('<h3>'+html.escape(group)+' '+name+'</h3><img loading="lazy" src="'+name+'">')
    (out/'index.html').write_text('\n'.join(chunks));publish_pointer(directory.parent,out)
    return dict(output=str(out),reload=checks,paired=paired)
