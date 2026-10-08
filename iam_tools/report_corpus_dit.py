"""Fail-closed corpus DiT report: unseen text, genuine noise starts, all probes.

Generated trajectories are NOT pointwise reconstructions of one IAM exemplar.
No RMSE-to-reference ranking, cherry-picking, smoothing or writer-style claims.
"""
import hashlib,html,json,math
from pathlib import Path
import h5py,numpy as np
from .corpus_dit import learning_rate
from .corpus_dit_contract import duration
from .corpus_dit_study import verify_sources,schedule
from .inkvae import edit_distance
from .pen_ab import file_sha
from .report_autoregressive_study import svg


def aggregate(rows,splits,guidances):
    out={}
    for split in splits:
        out[split]={}
        for g in guidances:
            r=[x for x in rows if x['split']==split and x['guidance']==g and x['policy']=='correct']
            if not r or sum(x['characters'] for x in r)==0:raise ValueError('complete nonempty evaluated text required')
            out[split][str(g)]=dict(evaluations=len(r),cer=sum(x['errors'] for x in r)/sum(x['characters'] for x in r),exact=sum(x['errors']==0 for x in r),missing_eoc=sum(not x['found_eoc'] for x in r),median_points=float(np.median([x['generated_points'] for x in r])))
    return out


def verify_evaluation(cfg,data,ev,split_ids,controls):
    rows=ev['rows'];keys=set()
    for r in rows:
        key=(r['split'],r['seed'],r['guidance'],r['policy'],r['sample_id'])
        if key in keys:raise ValueError('duplicate evaluated row')
        keys.add(key)
        if r['split'] not in split_ids or r['sample_id'] not in split_ids[r['split']] or r['text']!=data['records'][r['sample_id']]['text']:raise ValueError('declared complete split/text required')
        if r['characters']!=len(r['text']) or r['errors']!=edit_distance(r['text'],r['decoded']):raise ValueError('CER must reproduce from requested text and reader result')
        if any(not r[k] for k in ('no_source_trajectory','no_source_length','no_writer_id')):raise ValueError('genuine target-free generation required')
        if r['estimated_blocks']!=duration(data['duration'],r['text']) or not 1<=r['generated_points']<=8*r['estimated_blocks'] or (not r['found_eoc'] and r['generated_points']!=8*r['estimated_blocks']):raise ValueError('TRAIN-only requested duration and actual learned EOC/cap required')
        if r['policy']=='correct' and r['conditioning_text']!=r['text']:raise ValueError('correct request must actually be supplied')
        if r['policy']=='swapped':
            ids=split_ids[r['split']];other=ids[(ids.index(r['sample_id'])+1)%len(ids)]
            if r['conditioning_text']!=data['records'][other]['text']:raise ValueError('declared cyclic swapped conditioning must reproduce')
        if r['policy']=='null' and (r['conditioning_text'] or r['errors_against_supplied_text'] is not None):raise ValueError('NULL control must drop text')
        if r['policy']!='null' and r['errors_against_supplied_text']!=edit_distance(r['conditioning_text'],r['decoded']):raise ValueError('control CER against supplied text must reproduce')
    expected={(s,seed,g,'correct',sid) for s,ids in split_ids.items() for sid in ids for seed in cfg['eval_noise_seeds'] for g in cfg['eval_guidance']}
    if controls:
        expected|={(s,cfg['eval_noise_seeds'][0],1.,policy,sid) for s,ids in split_ids.items() for sid in ids for policy in ('swapped','null')}
    if keys!=expected:raise ValueError('all declared samples/seeds/guidances/controls required, no extras')
    if aggregate(rows,split_ids,cfg['eval_guidance'])!=ev['aggregate']:raise ValueError('aggregate must independently reproduce')


def verify_log(cfg,data,result,rows):
    if not rows or [r['step'] for r in rows]!=list(range(1,result['last_step']+1)):raise ValueError('every sequential update required')
    ids=data['scope']['splits']['train'];lengths={i:(data['records'][i]['points']+7)//8 for i in ids}
    batches=list(schedule(ids,lengths,cfg['max_updates'],cfg['batch'],cfg['schedule_seed']))
    if hashlib.sha256(json.dumps(batches).encode()).hexdigest()!=result['training_order_sha256'] or [r['sample_ids'] for r in rows]!=batches[:len(rows)]:raise ValueError('TRAIN-only declared full-epoch length-bucket order required')
    for r in rows:
        if not np.isfinite([r[k] for k in ('x0_loss','active40_loss','unused344_loss','raw_grad_norm','train_seconds')]).all():raise ValueError('finite actual update required')
        if r['clipped']!=(r['raw_grad_norm']>cfg['clip']) or r['lr']!=learning_rate(r['step'],cfg) or (r['text_dropped'] and r['prefix_retained']):raise ValueError('actual optimizer/CFG contract must reproduce')
        if not math.isclose(r['x0_loss'],(40*r['active40_loss']+344*r['unused344_loss'])/384,rel_tol=2e-6,abs_tol=2e-7):raise ValueError('all384-channel objective must reproduce from active/unused fields')
    if result['clip_fraction']!=sum(r['clipped'] for r in rows)/len(rows) or result['last_step']>cfg['max_updates'] or result['stop'] not in ('budget_completed','wall_limit'):raise ValueError('actual bounded budget/clipping required')
    if result['stop']=='budget_completed' and result['last_step']!=cfg['max_updates']:raise ValueError('completed budget must actually finish')
    if result['stop']=='wall_limit' and result['train_seconds']<cfg['max_train_wall_seconds']:raise ValueError('wall stop requires measured training time')
    selected=min(result['history'],key=lambda h:h['aggregate']['dev_unseen_text']['1.0']['cer'])
    if result['best_step']!=selected['step'] or result['best_dev_cer']!=selected['aggregate']['dev_unseen_text']['1.0']['cer']:raise ValueError('DEV-only both-seed guidance1 selection must reproduce')
    if not all(result[k] for k in ('codec_reader_unchanged','confirmation_not_encoded_during_training','not_promoted')):raise ValueError('frozen isolated unpromoted study required')


def generate(relative,root='data'):
    root=Path(root);p=root/relative;cfg=json.loads((p/'config.json').read_text());data=json.loads((p/'dataset.json').read_text());verify_sources(root,p,cfg,data);arm=p/'baseline';result=json.loads((arm/'result.json').read_text());rows=[json.loads(x) for x in (arm/'metrics.jsonl').read_text().splitlines()];verify_log(cfg,data,result,rows)
    for name,sha in [('checkpoint-best.pt',result['selected_sha256']),('checkpoint-last.pt',result['last_sha256'])]:
        if file_sha(arm/name)!=sha:raise ValueError('selected/final checkpoint drift')
    audit=json.loads((arm/'codec-cache-audit.json').read_text());cache_ids=data['scope']['splits']['train']+data['scope']['splits']['dev']
    if not audit['passed'] or not audit['confirmation_not_encoded'] or audit['cached_ids']!=cache_ids or file_sha(arm/'posterior-cache.h5')!=audit['cache_sha256'] or file_sha(arm/'whitening.pt')!=audit['whitening_sha256']:raise ValueError('TRAIN/DEV-only frozen source cache required')
    source_baselines={}
    for label,key in [('train','eval_train_ids'),('dev_unseen_text','eval_dev_ids')]:
        rs=[r for r in audit['source_codec_and_reader_probe'] if r['sample_id'] in cfg[key]]
        if len(rs)!=len(cfg[key]) or {r['sample_id'] for r in rs}!=set(cfg[key]):raise ValueError('every declared source decoder/reader probe required')
        if any(max(r['x_rmse'],r['y_rmse'])>.0025 or r['pen']['pen_up_f1']!=1 or not r['pen']['final_eoc_correct'] or r['pen']['non_final_false_eoc_count'] for r in rs):raise ValueError('source codec geometry/boundaries gate must reproduce')
        source_baselines[label]=dict(lines=len(rs),cer=sum(r['errors'] for r in rs)/sum(r['characters'] for r in rs),exact=sum(r['errors']==0 for r in rs))
    splits=dict(train=cfg['eval_train_ids'],dev_unseen_text=cfg['eval_dev_ids']);confirm=dict(exposed_held=data['scope']['splits']['exposed_held'],fresh_confirmation=data['scope']['splits']['fresh_confirmation']);evaluations={}
    for h in result['history']:
        step=h['step'];ev=json.loads((arm/f'eval-{step}.json').read_text());verify_evaluation(cfg,data,ev,splits,step in (0,cfg['max_updates']))
        if h['aggregate']!=ev['aggregate']:raise ValueError('history evaluation drift')
        evaluations[str(step)]=ev
    for name in ('final-confirmation','selected-confirmation'):
        ev=json.loads((arm/f'eval-{name}.json').read_text());verify_evaluation(cfg,data,ev,confirm,True);evaluations[name]=ev
    out=p/'report';out.mkdir(exist_ok=False);(out/'report-source.py').write_bytes(Path(__file__).read_bytes());pages=[]
    for stage,ev in evaluations.items():
        path=arm/f'evaluation-{stage}.h5'
        if file_sha(path)!=ev['packed_h5_sha256']:raise ValueError('saved evaluated trajectory drift')
        sections=[]
        with h5py.File(path) as hf:
            for j,r in enumerate(ev['rows']):
                key=f'{r["split"]}/{r["seed"]}/{r["guidance"]}/{r["policy"]}/{r["sample_id"]}';g=hf[key];q=g['points'][:]
                if json.loads(g.attrs['row'])!=r or len(q)!=r['generated_points']:raise ValueError('saved row/trajectory mismatch')
                states=q[:,2:].argmax(-1);hits=np.flatnonzero(states==2)
                if r['found_eoc']!=(len(hits)>0) or (len(hits) and (len(hits)!=1 or hits[0]!=len(q)-1)):raise ValueError('actual first decoded EOC must be the reported stop')
                sections.append(trajectory_section(r,q))
        page=f'gallery-{stage}.html';(out/page).write_text(_head('All outputs '+stage)+'<a href="index.html">Summary</a><p>Every declared row, no selection by appearance. Geometry is rendered at fixed100px/model-unit; scroll strips. No smoothing or width normalization.</p>'+''.join(sections));pages.append((stage,page,ev['aggregate']))
    table=[]
    for stage,page,a in pages:
        for split,values in a.items():
            for guidance,v in values.items():table.append('<tr><td><a href="'+page+'">'+html.escape(stage)+'</a></td>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in (split,guidance,f'{100*v["cer"]:.2f}%',v['exact'],v['evaluations'],v['missing_eoc'],v['median_points']))+'</tr>')
    control=[]
    for stage in ('final-confirmation','selected-confirmation'):
        ev=evaluations[stage]
        for split in confirm:
            for policy in ('correct','swapped','null'):
                rs=[r for r in ev['rows'] if r['split']==split and r['policy']==policy and r['seed']==cfg['eval_noise_seeds'][0] and r['guidance']==1.]
                control.append(dict(stage=stage,split=split,policy=policy,cer=sum(r['errors'] for r in rs)/sum(r['characters'] for r in rs),exact=sum(r['errors']==0 for r in rs),rows=len(rs)))
    summary=dict(result=result,config=cfg,cache_audit=dict(lines=audit['lines'],training_lines=audit['training_lines'],probe_source_cer=audit['probe_source_cer'],cache_seconds=audit['cache_seconds']),evaluations={k:v['aggregate'] for k,v in evaluations.items()},matched_noise_controls=control,source_reader_baselines=source_baselines,scope=cfg['caveats'],visual_inspection_pending=True,no_paper_reproduction=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'index.html').write_text(_head('Corpus actual-DiT pilot')+'<h1>Corpus actual-DiT: novel text, not familiar-line polishing</h1><p>'+html.escape(cfg['profile'])+'</p><p>8144 TRAIN lines,186 writers. Genuine sampled-x0 diffusion, not a zero-input deterministic mapper. Fixed codec is initialized40-field transport, NOT learned semantic InkVAE. Generic writer distribution, NOT writer-specific style control. Frozen reader is corpus-familiar; fresh16 is generator-novel, not an independent reader benchmark.</p><p>Requested text + TRAIN-only predicted duration + randomnoise only. No oracle target length, reference trajectory, smoothing, forced endpoint/EOC, KL/CTC/style updates. DEV-only selection guidance1 over both noise seeds. Fresh/held are used only after the budget; their results do not select checkpoints. Controls keep original requested duration/noise; swapped control CER below is against original requested text, not evidence it should write that text.</p><p>Selected step'+str(result['best_step'])+'; final'+str(result['last_step'])+'; stop '+html.escape(result['stop'])+'. <a href="summary.json">Metrics/config/scope</a> | <a href="../config.json">Immutable config</a> | <a href="../as-run-source.tar.gz">As-run training source</a></p><table><tr><th>Stage/gallery</th><th>Split</th><th>Guidance</th><th>CER</th><th>Exact</th><th>Evaluations</th><th>Missing EOC</th><th>Median points</th></tr>'+''.join(table)+'</table><h2>Matched first-noise guidance1 controls</h2><pre>'+html.escape(json.dumps(control,indent=2))+'</pre><h2>Source validation</h2><p>Source codec probe CER context: '+str(audit['probe_source_cer'])+'. Source baseline by split: '+html.escape(json.dumps(source_baselines))+'. This evaluator is imperfect on genuine handwriting; zero CER is not its universal source baseline. All cached TRAIN/DEV packed fields passed <=.001 XY and exact pens;48 source decoder probes passed<=.0025 axisRMSE and perfect boundaries. Free output is stochastic and is NOT scored using pointwise RMSE to one exemplar. Visual inspection of all fixed/confirmation outputs is still required; report generation alone is not a visual quality verdict.</p>')
    return str(out)


def _head(title):
    return '<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;border:1px solid #ddd;max-height:650px}.strip svg{display:block;max-width:none}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}section{margin:35px 0}pre{white-space:pre-wrap}</style>'


def trajectory_section(row,points):
    """Inline exact marker-free SVG: no per-output inode or hidden rescaling.

    Corpus galleries can contain thousands of outputs. Original volume is near
    its inode limit; keep complete rows in a few HTML files, never prune rows.
    """
    r=row;label=f'{r["split"]} | {r["sample_id"]} | {r["policy"]} | seed{r["seed"]} | guidance{r["guidance"]}'
    return '<section><h3>'+html.escape(label)+'</h3><p>Requested: '+html.escape(r['text'])+'</p><p>Supplied: '+html.escape(repr(r['conditioning_text']))+'</p><p>Reader: '+html.escape(repr(r['decoded']))+f' | errors{r["errors"]}/{r["characters"]} | {len(points)}points | EOC={r["found_eoc"]}</p><div class="strip">'+svg(points)+'</div></section>'
