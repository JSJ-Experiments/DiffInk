"""Independent reload, paired-source-sigma curve audit and final/selected report."""
import hashlib,html,json,time
from pathlib import Path
import h5py,numpy as np,torch
from .posterior_isolation import ARMS,SOURCE,SHA,evaluate,load_batches,eligibility,assert_output_isolation
from .ocr_pool_study import load_pool
from .ocr_convergence import POOL_SHA
from .ocr_joint_adapter import load_reader,READER_SHA
from .ocr_joint_study import LEGACY_EIGHT
from .ocr_context_study import tensor_digest
from .report_ocr_joint import cached_cpu_evaluation
from .inkvae import setup
from .pen_ab import file_sha


def rebuild(folder,which,repo,root,configs):
    model,train,val,_,_=setup(Path(repo)/'configs/engineering_english.yaml',repo,Path(root)/'diffink/iam_overfit')
    train.hf.close();val.hf.close();saved=torch.load(Path(folder)/which,map_location='cpu',weights_only=True)
    if saved['config']['cfg']!=configs['cfg'] or saved['source_sha256']!=SHA or saved['frozen_reader_sha256']!=READER_SHA:
        raise ValueError('checkpoint data/reader contract drift')
    model.load_state_dict(saved['model_state_dict'],strict=True);model.apply_checkpoint_contract(saved,allow_research_ocr=True)
    model.ocr_model,_=load_reader(root,configs['cfg']);return model.eval().requires_grad_(False),saved


def pairing(configs,results,logs):
    base=next(iter(configs.values()));checks={}
    for key in ('source_sha256','reader_sha256','pool_manifest_sha256','source_std_sha256','cfg','splits','train_ids','schedule_seed','schedule_sha256','noise_seed','optimizer_groups','max_updates','physical_batch','gradient_accumulation','mean_geometry_weight','sampled_geometry_weight','pen_weight','ctc_weight','style_weight','gmm_weight','dropout','target_delta_weight'):
        if any(c[key]!=base[key] for c in configs.values()):raise ValueError('unmatched '+key)
    for a,w in ARMS.items():
        if configs[a]['kl_weight']!=w or results[a]['kl_weight']!=w:raise ValueError('KL arm coefficient drift')
        if len(logs[a])!=results[a]['last_step']:raise ValueError('incomplete per-update log')
    checks['freeze_contract']=configs['joint']['freeze_policy']=='source body+posterior groups' and configs['posterior_only']['freeze_policy']=='entire codec except conv_logvar'
    checks['frozen_optimizer_and_weights']=all(all(v for k,v in r['frozen_state_invariance'].items() if k!='frozen_parameters') for r in results.values())
    checks['posterior_output_isolation']=results['posterior_only']['posterior_output_isolation']
    if not all(checks.values()):raise ValueError('freeze/isolation guards failed')
    reference=logs['joint']
    checks['matched_prefix_batches_noise']=all(all(x['sample_ids']==y['sample_ids'] and x['noise_sha256']==y['noise_sha256'] for x,y in zip(reference,rows)) for rows in logs.values())
    checks['full_matched_budget']=all(r['last_step']==base['max_updates'] for r in results.values())
    checks['same_final_rng_if_full_budget']=not checks['full_matched_budget'] or len({(r['final_rng_cpu_sha256'],r['final_rng_cuda_sha256']) for r in results.values()})==1
    checks['immutable_protected']=all(r['reader_weights_unchanged'] and r['readout_style_pen_variance_protected'] and r['source_unchanged'] for r in results.values())
    if not checks['matched_prefix_batches_noise'] or not checks['immutable_protected'] or not checks['same_final_rng_if_full_budget']:raise ValueError('paired/immutable guards failed')
    return checks


def aggregate_row_table(results):
    rows=[]
    for arm,result in results.items():
        for entry in result['history']:
            for policy in ('own','fixed_source_sigma'):
                for split in ('train_probe','dev','held_out','named'):
                    g=entry['groups'][policy][split]
                    for mode in ('mu','sampled'):
                        rows.append(dict(arm=arm,step=entry['step'],policy=policy,split=split,mode=mode,kl=entry['posterior'][split]['kl_per_element'],xy_std_mean=entry['posterior'][split]['xy']['std_mean'],**g[mode]))
    return rows


def checkpoint_isolation(initial,last):
    """Independently check serialized geometry weights and body Adam state."""
    a,b=initial['model_state_dict'],last['model_state_dict'];count=0
    if a.keys()!=b.keys():raise ValueError('codec state keys changed')
    for name in a:
        if name.startswith('conv_logvar.'):continue
        if not torch.equal(a[name],b[name]):raise AssertionError('serialized frozen codec state changed: '+name)
        count+=1
    x,y=initial['optimizer_state_dict'],last['optimizer_state_dict']
    if x['param_groups']!=y['param_groups']:raise AssertionError('serialized optimizer groups changed')
    body=[g for g in x['param_groups'] if g['name']=='body']
    if len(body)!=1:raise ValueError('one restored body optimizer group required')
    for pid in body[0]['params']:
        old,new=x['state'].get(pid,{}),y['state'].get(pid,{})
        if old.keys()!=new.keys():raise AssertionError('serialized body optimizer state keys changed')
        for key in old:
            same=torch.equal(old[key],new[key]) if torch.is_tensor(old[key]) else old[key]==new[key]
            if not same:raise AssertionError('serialized body optimizer moments/steps changed')
    return dict(serialized_geometry_states_equal=True,serialized_body_optimizer_states_equal=True,
        codec_state_tensors_checked=count,body_parameters_checked=len(body[0]['params']))


def trajectory_invariance(folder,steps,ids,draws=20):
    """Check every saved mean/fixed-noise ARRAY, not just scalar metrics."""
    folder=Path(folder);checked=0
    for step in steps:
        if step==0:continue
        for policy,kinds in [('own',['mu']),('fixed_source_sigma',['mu']+[f'z-{j}' for j in range(draws)])]:
            for sid in ids:
                for kind in kinds:
                    a=np.load(folder/policy/'step-0'/sid/(kind+'.npy'))
                    b=np.load(folder/policy/f'step-{step}'/sid/(kind+'.npy'))
                    if not np.array_equal(a,b):raise AssertionError(f'posterior isolation trajectory changed: {policy}/{step}/{sid}/{kind}')
                    checked+=1
    return dict(exact_array_equal=True,compared_trajectory_pairs=checked,lines=len(ids),draws=draws)


def report(directory,repo,root='/data'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .curve_audit import draw,split_xy
    directory=Path(directory);root=Path(root);torch.set_num_threads(2)
    results=json.loads((directory/'result.json').read_text());configs={a:json.loads((directory/a/'config.json').read_text()) for a in ARMS}
    logs={a:[json.loads(x) for x in (directory/a/'metrics.jsonl').read_text().splitlines()] for a in ARMS};paired=pairing(configs,results,logs)
    cfg=configs['joint'];pool,m,vocab=load_pool(root,POOL_SHA)
    if file_sha(root/SOURCE)!=SHA or file_sha(directory/'fixed-source-std.pt')!=cfg['source_std_sha256']:raise ValueError('source weights/std drift')
    splits=cfg['splits'];ids=set(i for group in splits.values() for i in group)
    frozen=checkpoint_isolation(torch.load(directory/'posterior_only/checkpoint-initial.pt',map_location='cpu',weights_only=True),torch.load(directory/'posterior_only/checkpoint-last.pt',map_location='cpu',weights_only=True))
    isolation=trajectory_invariance(directory/'posterior_only',[v['step'] for v in results['posterior_only']['history']],ids)
    batches,records,origins=load_batches(root,pool,m,vocab,ids,'cpu')
    if json.loads((directory/'evaluation-records.json').read_text())!={i:dict(records[i],origin=origins[i]) for i in sorted(ids)}:raise ValueError('evaluation identities/origins drift')
    if (directory/'pool-manifest.json').read_bytes()!=(pool/'manifest.json').read_bytes():raise ValueError('pool manifest byte drift')
    for q in (directory/'source-code').rglob('*.py'):
        relative=q.relative_to(directory/'source-code');current=Path(__file__).parent/relative.name if relative.parts[0]=='iam_tools' else Path(repo)/relative
        if q.read_bytes()!=current.read_bytes():raise ValueError('as-run source differs: '+str(relative))
    source_std=torch.load(directory/'fixed-source-std.pt',map_location='cpu',weights_only=True)
    out=directory/'report';out.mkdir(parents=True,exist_ok=True);(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    cpu={};reloads={};versions={'source':('joint','checkpoint-initial.pt',0)}
    for arm,r in results.items():
        for which,key in [('checkpoint-best.pt','selected_sha256'),('checkpoint-last.pt','last_sha256')]:
            if file_sha(directory/arm/which)!=r[key]:raise ValueError('checkpoint SHA changed')
        versions[arm+'-final']=(arm,'checkpoint-last.pt',r['last_step'])
        if r['best_step'] not in (0,r['last_step']):versions[arm+'-selected']=(arm,'checkpoint-best.pt',r['best_step'])
    for name,(arm,which,step) in versions.items():
        model,saved=rebuild(directory/arm,which,repo,root,cfg);before=tensor_digest(model.ocr_model.state_dict())
        if saved['continuation_updates']!=step:raise ValueError('step metadata drift')
        cpu[name]=cached_cpu_evaluation(model,batches,records,vocab,splits,out/'cpu-means'/name,step,draws=0,save=True)
        gpu=json.loads((directory/arm/'own'/f'eval-{step}.json').read_text());original={v['sample_id']:v for v in gpu['lines']};differences=[]
        for row in cpu[name]['lines']:
            sid=row['sample_id'];points=np.load(out/'cpu-means'/name/f'step-{step}'/sid/'mu.npy');reference=np.load(directory/arm/'own'/f'step-{step}'/sid/'mu.npy')
            differences.append(dict(sample_id=sid,max_xy_difference=float(np.abs(points[:,:2]-reference[:,:2]).max()),pen_mismatches=int((points[:,2:].argmax(1)!=reference[:,2:].argmax(1)).sum()),transcript_difference=row['mu']['decoded']!=original[sid]['mu']['decoded']))
        check=dict(max_xy_difference=max(v['max_xy_difference'] for v in differences),pen_mismatches=sum(v['pen_mismatches'] for v in differences),transcript_differences=sum(v['transcript_difference'] for v in differences),reader_unchanged=tensor_digest(model.ocr_model.state_dict())==before,lines=differences)
        if check['max_xy_difference']>1e-4 or check['pen_mismatches'] or check['transcript_differences'] or not check['reader_unchanged']:raise AssertionError(check)
        reloads[name]=check;del model
    # Source/each FINAL all8 own/fixed20 CPUdraws, not GPU-paired epsilon.
    legacy={};legacy_splits={'named':list(LEGACY_EIGHT),'train_probe':list(LEGACY_EIGHT),'dev':list(LEGACY_EIGHT),'held_out':list(LEGACY_EIGHT)}
    old_batches={i:batches[i] for i in LEGACY_EIGHT};source_legacy=None
    source_files=[Path(__file__),Path(__file__).with_name('posterior_isolation.py'),Path(__file__).with_name('ocr_joint_study.py'),Path(__file__).with_name('ocr_joint_adapter.py'),Path(repo)/'model/vae.py',Path(repo)/'model/blocks.py',Path(repo)/'model/losses.py',Path(__file__).with_name('ocr_context_features.py'),Path(__file__).with_name('ocr_recurrent.py'),Path(__file__).with_name('latent_integration.py'),Path(__file__).with_name('trajectory_geometry.py'),Path(__file__).with_name('curve_audit.py')]
    for name,(arm,which,step) in versions.items():
        if name.endswith('-selected'):continue
        folder=out/'cpu-legacy-eight'/name;folder.mkdir(parents=True,exist_ok=True)
        fingerprint=dict(checkpoint=file_sha(directory/arm/which),source_std=file_sha(directory/'fixed-source-std.pt'),source={p.name:file_sha(p) for p in source_files},data={i:records[i] for i in LEGACY_EIGHT},draws=20,torch=str(torch.__version__),threads=torch.get_num_threads())
        key=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest();cache=folder/'completed.json'
        if cache.exists() and json.loads(cache.read_text())['fingerprint']==key:
            value=json.loads(cache.read_text())['result']
            for policy in ('own','fixed_source_sigma'):
                for sid in LEGACY_EIGHT:
                    for k in ['mu']+[f'z-{j}' for j in range(20)]:
                        if not (folder/policy/f'step-{step}'/sid/(k+'.npy')).exists():raise ValueError('incomplete cached legacy trajectories')
        else:
            model,_=rebuild(directory/arm,which,repo,root,cfg)
            value=evaluate(model,old_batches,records,vocab,legacy_splits,source_std,folder,step,save_trajectories=True)
            temp=folder/'completed.tmp';temp.write_text(json.dumps(dict(fingerprint=key,contract=fingerprint,result=value),indent=2)+'\n');temp.replace(cache);del model
            if str(folder).startswith('/data/'):
                import modal
                modal.Volume.from_name('diffink-data').commit()
        legacy[name]=value
        if name=='source':source_legacy=value
    assert_output_isolation(legacy['posterior_only-final'],source_legacy)
    gates={n:eligibility(v,source_legacy,list(LEGACY_EIGHT)) for n,v in legacy.items() if n!='source'}
    summary=dict(independent_checkpoint_isolation=frozen,gpu_exact_trajectory_isolation=isolation,results=results,configs=configs,paired=paired,cpu_reload=reloads,cpu_groups={k:r['groups'] for k,r in cpu.items()},legacy_groups={k:{p:r['groups']['named'] for p,r in v.items()} for k,v in legacy.items()},legacy_gates=gates,
       scope='GPU200means+20draws under each own/fixed-source-sigma; independent CPU200source/final/selected(ifnot0orfinal)means and all8source/final20pairedCPUdraws. CPU draws not paired to GPU. DEV reused,report32prior codec overlap, initializedpolyphase40 NOT semantic/generative readiness.',source_sha256=SHA,reader_sha256=READER_SHA,source_std_sha256=cfg['source_std_sha256'])
    summary['kl_decomposition']={a:{str(e['step']):{k:e['posterior']['dev'][k]['kl_contribution_per_total_element'] for k in ('xy','pen','unused')} for e in r['history']} for a,r in results.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');table=aggregate_row_table(results);(out/'metrics-table.json').write_text(json.dumps(table,indent=2)+'\n')
    # All200 marker-free means, final arms. Gallery is not a claim all manually reviewed.
    targets={sid:np.column_stack((batches[sid][0][0,:2,:int(batches[sid][1].sum())].T.numpy()*.01,batches[sid][0][0,2:,:int(batches[sid][1].sum())].T.numpy())) for sid in ids}
    pages=[];columns=['source']+[a+'-final' for a in ARMS]
    for split,values in splits.items():
        for start in range(0,len(values),4):
            group=values[start:start+4];fig,axes=plt.subplots(len(group),1+len(columns),figsize=(6*(1+len(columns)),2.8*len(group)),squeeze=False)
            for j,sid in enumerate(group):
                qs=[targets[sid]]+[np.load(out/'cpu-means'/v/f'step-{versions[v][2]}'/sid/'mu.npy') for v in columns]
                lo,hi=targets[sid][:,:2].min(0),targets[sid][:,:2].max(0)
                for ax,q,title in zip(axes[j],qs,['target']+columns):
                    draw(ax,split_xy(q[:,:2],q[:,2:].argmax(1)));ax.set_xlim(lo[0]-.05,hi[0]+.05);ax.set_ylim(lo[1]-.08,hi[1]+.08);ax.set_title(sid+' | '+title,fontsize=9)
            fig.tight_layout();name=f'{split}-{start//4+1}.png';fig.savefig(out/name,dpi=135);plt.close(fig);pages.append(name)
    # Closeups with each final arm, not selection hiding final tradeoffs.
    source_rows={r['sample_id']:r for r in source_legacy['own']['lines']}
    worst={i:max(range(20),key=lambda j:source_rows[i]['sampled'][j]['geometry']['turn_angle_error_degrees']['p90'] or 0) for i in LEGACY_EIGHT}
    regions=[('p08-936z-05',(4.94,5.34),(.15,.57),'c in chocolate'),('a07-421z-02',(8.28,8.75),(.40,.97),'h in hope')];closeups=[]
    for policy,kind in [('own','mu'),('own','source-worst-z'),('fixed_source_sigma','source-worst-z')]:
        fig,axes=plt.subplots(2,1+len(columns),figsize=(6*(1+len(columns)),9))
        for j,(sid,xlim,ylim,label) in enumerate(regions):
            k='mu' if kind=='mu' else 'z-'+str(worst[sid]);qs=[targets[sid]]+[np.load(out/'cpu-legacy-eight'/v/policy/f'step-{versions[v][2]}'/sid/(k+'.npy')) for v in columns]
            for ax,q,title in zip(axes[j],qs,['target']+columns):
                draw(ax,split_xy(q[:,:2],q[:,2:].argmax(1)));ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_title(f'{label} | {title} | {k}',fontsize=9)
        fig.tight_layout();name=f'{policy}-{kind}-closeups.png';fig.savefig(out/name,dpi=175);plt.close(fig);closeups.append(name)
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for arm,r in results.items():
        entries=r['history'];x=[v['step'] for v in entries]
        plots=[([v['posterior']['dev']['kl_per_element'] for v in entries],'DEV KL / valid element'),([v['posterior']['dev']['xy']['std_mean'] for v in entries],'DEV mean XYσ'),([v['groups']['own']['dev']['sampled']['x_rmse']+v['groups']['own']['dev']['sampled']['y_rmse'] for v in entries],'DEV own sampled X+Y RMSE'),([v['groups']['fixed_source_sigma']['dev']['sampled']['x_rmse']+v['groups']['fixed_source_sigma']['dev']['sampled']['y_rmse'] for v in entries],'DEV fixedσ sampled X+Y RMSE'),([v['groups']['own']['dev']['sampled']['mean_per_line_turn_p90'] for v in entries],'DEV own sampled mean turnp90°'),([100*v['groups']['own']['dev']['mu']['cer'] for v in entries],'DEV mean CER%')]
        for ax,(y,title) in zip(axes.ravel(),plots):ax.plot(x,y,'o-',label=arm);ax.set_title(title);ax.legend();ax.grid(alpha=.3)
    fig.tight_layout();fig.savefig(out/'learning.png',dpi=150);plt.close(fig)
    rows=''
    for t in table:
        if t['split']=='dev' and t['mode']=='sampled' and (t['step']==0 and t['arm']=='joint' or t['step']==results[t['arm']]['last_step']):
            rows+=f'<tr><td>{t["arm"]}/{t["step"]}/{t["policy"]}</td><td>{t["kl"]:.7f}</td><td>{t["xy_std_mean"]:.7f}</td><td>{t["x_rmse"]:.7f}</td><td>{t["y_rmse"]:.7f}</td><td>{t["mean_per_line_turn_p90"]:.4f}°</td><td>{100*t["cer"]:.5f}%</td><td>{t["min_pen_f1"]}</td></tr>'
    body=f'''<!doctype html><meta charset="utf-8"><title>Posterior isolation with frozen geometry</title><style>body{{font:16px system-ui;max-width:1800px;margin:25px auto;padding:15px}}img{{width:100%}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:7px}}</style><h1>Posterior isolation: body+posterior versus uncertainty-only</h1><p>Posterior-only freezes encoder, latent mean, convolutional decoder AND Transformer. Exact saved-array invariance: {html.escape(str(isolation))}.</p><p>Initialized polyphase40 codec, fixed readout. No OCR/style/GMM training. Two200update matched arms: body+posterior versus posterior-only from original source, not contracted joint checkpoint. No production promotion or semantic/generative readiness. All final arms shown, including failures.</p><p><a href="summary.json">All configs/selection/gates/CPU reload</a> · <a href="metrics-table.json">Every split/step/policy/metric</a> · <a href="../joint/gradient-diagnostics.json">Initial gradient norms</a></p><p>Selection: {html.escape(str({a:dict(selected=r['best_step'],final=r['last_step'],stop=r['stop_reason']) for a,r in results.items()}))}. TRAIN-only corrected KL then sampled XY, eligible under both noise-policy curve/pen gates. DEV/report never select/stop.</p><h2>DEV sampled results</h2><p>Angles are average per-line p90 target turn errors; index differences are not physical velocity or geometric curvature. fixed_source_sigma is deliberately NOT the learned posterior.</p><table><tr><th>Arm/step/noise</th><th>KL</th><th>Mean XYσ</th><th>X RMSE</th><th>Y RMSE</th><th>Turn</th><th>CER</th><th>Pen F1</th></tr>{rows}</table><img src="learning.png"><h2>Named mean and same-draw closeups</h2><p>Source-worst draw chosen among20 by SOURCE full-line turnp90, same index across all versions/policies; not checkpoint selection. All8 CPU draws use their own stable ID-order seeds, not GPU-paired draws. Target polygonality/corners remain deliberately unsmoothed.</p>'''
    for name in closeups:body+=f'<h3>{name}</h3><img src="{name}">'
    body+='<h2>Where KL changes</h2><p>Contributions use the SAME full valid-channel denominator; unused344channels near prior can hide active16XY/24pen mismatches. A lower total KL through pen uncertainty alone is not an improved coordinate prior.</p><pre>'+html.escape(json.dumps(summary['kl_decomposition'],indent=2))+'</pre>'
    body+='<h2>All reported mean lines</h2><p>200unique GPU/CPU lines, all8 original references included; four absent from reader pool remain evaluation-only. Gallery generation is not equivalent to manual visual inspection of every line. ReusedDEV5writers/report32prior codec overlap limit independent benchmark claims.</p>'
    for name in pages:body+=f'<h3>{name}</h3><img loading="lazy" src="{name}">'
    (out/'index.html').write_text(body)
    return dict(report=str(out/'index.html'),paired=paired,cpu_reload={k:{a:b for a,b in v.items() if a!='lines'} for k,v in reloads.items()},legacy_gates=gates)
