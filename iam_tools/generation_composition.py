"""TRAIN-only duration estimator and sealed generator-confirmation scope."""
import hashlib,math
from collections import defaultdict
import numpy as np
from .ocr_pool import normalized_text


def text_features(text):
    if not isinstance(text,str) or not text:raise ValueError('nonempty requested text required')
    n=len(text)
    return [math.log(n),sum(c==' ' for c in text)/n,sum(not c.isalnum() and c!=' ' for c in text)/n,sum(c.isupper() for c in text)/n,sum(c.isdigit() for c in text)/n]


def fit_duration(records,train_ids,writers,ridge=1.):
    if not train_ids or len(set(train_ids))!=len(train_ids) or ridge<=0:raise ValueError('unique nonempty TRAIN IDs and positive ridge required')
    x=np.array([text_features(records[i]['text']) for i in train_ids]);mean=x.mean(0);std=x.std(0).clip(.05)
    w=np.eye(len(writers))[[writers.index(records[i]['writer_id']) for i in train_ids]]
    design=np.c_[np.ones(len(x)),(x-mean)/std,w];target=np.log([(records[i]['points']+7)//8 for i in train_ids])
    penalty=np.eye(design.shape[1])*ridge;penalty[0,0]=0
    coef=np.linalg.solve(design.T@design+penalty,design.T@target)
    return dict(train_ids=list(train_ids),writers=list(writers),feature_mean=mean.tolist(),feature_std=std.tolist(),coefficients=coef.tolist(),ridge=ridge,
        blocks_per_character=sum((records[i]['points']+7)//8 for i in train_ids)/sum(len(records[i]['text']) for i in train_ids),
        minimum_blocks=1,maximum_blocks=256,definition='TRAIN-only log-duration ridge on requested text features + shrunk writer intercept; round(exp), bounded1..256; NO sample ID/target points at inference')


def predict_duration(model,text,writer_id):
    x=(np.array(text_features(text))-model['feature_mean'])/model['feature_std']
    writer=np.zeros(len(model['writers']))
    if writer_id in model['writers']:writer[model['writers'].index(writer_id)]=1
    log=float(np.r_[1.,x,writer]@np.array(model['coefficients']))
    return max(model['minimum_blocks'],min(model['maximum_blocks'],int(np.rint(np.exp(np.clip(log,-10,10))))))


def seal_confirmation(manifest,history_records,train_ids,writers,count=16,seed=26142,development_ids=()):
    if count!=16 or len(set(train_ids))!=len(train_ids):raise ValueError('fixed16 sealed confirmation and unique TRAIN required')
    records=manifest['records'];known=set(history_records);forms={records[i]['prompt_family'] for i in list(train_ids)+list(development_ids)};texts={normalized_text(r['text']) for r in history_records.values()}
    chars={c for i in train_ids for c in records[i]['text']};blocked=set(manifest['dev_writers'])|set(manifest['test_writers'])
    groups=defaultdict(list)
    for i in manifest['splits']['large_train']:
        r=records[i]
        if i not in known and r['writer_id'] in writers and r['writer_id'] not in blocked and r['prompt_family'] not in forms and normalized_text(r['text']) not in texts and set(r['text'])<=chars:
            groups[(r['writer_id'],r['prompt_family'])].append(i)
    ordered=sorted(groups,key=lambda key:hashlib.sha256(f'{seed}:{key}'.encode()).hexdigest());used=set();picked_texts=set();picked_forms=set();ids=[]
    for writer,family in ordered:
        options=sorted(groups[(writer,family)],key=lambda i:hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest())
        if writer not in used and family not in picked_forms and len(options)>=2:
            selected=[];seen=set()
            for i in options:
                t=normalized_text(records[i]['text'])
                if t not in seen and t not in picked_texts:selected.append(i);seen.add(t)
                if len(selected)==2:break
            if len(selected)==2:ids.extend(selected);used.add(writer);picked_forms.add(family);picked_texts.update(seen)
        if len(ids)==count:break
    if len(ids)!=count:raise ValueError('insufficient sealed8writers×2 novel-form prompts')
    return dict(ids=ids,records={i:records[i] for i in ids},seed=seed,selection='metadata-only fixed16/8writers/8forms outside every packed1032 prior generation example ID/normalized transcript; forms excluded from CURRENT fresh TRAIN/development only (other lines from these forms occurred in older1024 runs); globally TRAIN-character-covered; known TRAIN writers, no reserved writers',
        not_independent_reader_benchmark=True,policy='never loaded/encoded/scored during GPU training; expose only selected TRAIN checkpoints once both arms finish; no confirmation tuning')


def learning_rate(step,steps):
    if type(step)!=int or type(steps)!=int or not 1<=step<=steps or steps<16000:raise ValueError('bounded valid fresh-training schedule required')
    if step<=1000:return 5e-5*(.2+.8*step/1000)
    if step<=2*steps//3:return 5e-5
    progress=(step-2*steps//3)/(steps-2*steps//3)
    return 1e-5+4e-5*.5*(1+math.cos(math.pi*progress))


def seal_synthetic(history_records,train_ids,writers,seed=26143):
    subjects=['I','We','She','They'];verbs=['left','found','kept','moved']
    objects=['the blue notebook','a small letter','the old book','a cup of tea'];places=['on the desk','by the window','near the door','in the office']
    prompts=[f'{subject} {verb} {objects[(j+k)%4]} {places[(2*j+k)%4]}.' for j,subject in enumerate(subjects) for k,verb in enumerate(verbs)]
    known={normalized_text(r['text']) for r in history_records.values()};chars={c for i in train_ids for c in history_records[i]['text']}
    if len(set(prompts))!=16 or any(normalized_text(t) in known or not set(t)<=chars for t in prompts):raise ValueError('synthetic prompts must be unseen and TRAIN character covered')
    selected=sorted(writers,key=lambda w:hashlib.sha256(f'{seed}:{w}'.encode()).hexdigest())[:8]
    return dict(records={f'synthetic-{j+1:02d}':dict(text=t,writer_id=selected[j//2]) for j,t in enumerate(prompts)},
        selection='fixed16 new grammatical recombinations/8known writers, metadata duplicate and TRAIN character coverage checks only; no paired trajectory exists',
        policy='only TRAIN-selected frozen models after both budgets finish; estimated text/writer duration ONLY; never fabricated oracle duration or reference trajectory')
