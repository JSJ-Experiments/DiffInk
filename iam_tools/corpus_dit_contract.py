"""Corpus-scale released-DiT baseline scope. Metadata-only novel-text reservation.

No raw trajectory of a fresh confirmation is encoded or read during training.
Faithful initialized transport is NOT the authors' semantic VAE. No eight-line
capacity ranking is a proxy for new-text composition. Original pools immutable.
"""
from collections import Counter,defaultdict
import hashlib,json,time,tarfile
from pathlib import Path
from .ocr_pool import normalized_text
from .ocr_pool_study import load_pool
from .ocr_convergence import POOL_SHA
from .frozen_ocr_study import SOURCE as CODEC,SHA as CODEC_SHA
from .generation_capacity import DATA,DATASET_SHA
from .ocr_joint_adapter import READER_REL,READER_SHA
from .pen_ab import file_sha
from .generation_composition import fit_duration,predict_duration
SEED=72142


def scope(manifest,history_records,seed=SEED):
    records=manifest['records'];available=manifest['splits']['large_train']
    oldforms={r['prompt_family'] for r in history_records.values()};oldtexts={normalized_text(r['text']) for r in history_records.values()}
    families=defaultdict(list)
    for sid in available:
        r=records[sid]
        if r['prompt_family'] not in oldforms and normalized_text(r['text']) not in oldtexts:
            families[r['prompt_family']].append(sid)
    key=lambda v:hashlib.sha256(f'{seed}:{v}'.encode()).hexdigest()
    selected=[];reserved=[]
    for family in sorted(families,key=key):
        bywriter=defaultdict(list)
        for sid in sorted(families[family],key=key):bywriter[records[sid]['writer_id']].append(sid)
        picked=[]
        for writer in sorted(bywriter,key=key):
            texts=set();pair=[]
            for sid in bywriter[writer]:
                text=normalized_text(records[sid]['text'])
                if text not in texts:pair.append(sid);texts.add(text)
                if len(pair)==2:break
            if len(pair)==2:picked.extend(pair)
            if len(picked)==4:break
        if len(picked)==4:selected.extend(picked);reserved.append(family)
        if len(reserved)==4:break
    if len(selected)!=16:raise ValueError('need four novel families with two writers/two distinct texts each')
    selected_text={normalized_text(records[i]['text']) for i in selected}
    train=[i for i in available if records[i]['prompt_family'] not in reserved and normalized_text(records[i]['text']) not in selected_text]
    result=dict(train=train,dev=list(manifest['splits']['dev']),exposed_held=list(manifest['splits']['held_out']),fresh_confirmation=selected)
    validate_scope(records,result,history_records,manifest['test_writers'],manifest['dev_writers'])
    return dict(splits=result,fresh_reserved_families=reserved,excluded_from_training=sorted(set(available)-set(train)),fresh_metadata_only=True,
        fresh_selection='four deterministic new form families outside prior1032 generator examples; two writers x two texts each; exclude whole families AND matching normalized transcripts globally from generator TRAIN',
        caveats='fresh for generator trajectory/text families, not an independent reader/pretrained-codec benchmark; source codec previously saw original192 geometry; reader corpus-familiar; exposed_held labels existed in older reports')


def validate_scope(records,splits,history,test_writers,dev_writers):
    names=('train','dev','exposed_held','fresh_confirmation')
    if set(splits)!=set(names) or any(not splits[k] or len(set(splits[k]))!=len(splits[k]) for k in names):raise ValueError('unique nonempty declared corpus splits required')
    populations=[set(splits[k]) for k in names]
    if any(populations[j]&populations[k] for j in range(4) for k in range(j)):raise ValueError('line leakage')
    train=splits['train'];families={records[i]['prompt_family'] for i in train};texts={normalized_text(records[i]['text']) for i in train}
    for split in names[1:]:
        if families&{records[i]['prompt_family'] for i in splits[split]} or texts&{normalized_text(records[i]['text']) for i in splits[split]}:raise ValueError('global prompt/transcript TRAIN leakage')
    tw={records[i]['writer_id'] for i in train}
    if tw&(set(test_writers)|set(dev_writers)):raise ValueError('reserved writer leakage')
    if {records[i]['writer_id'] for i in splits['dev']}-set(dev_writers):raise ValueError('independent dev writer scope changed')
    fresh=splits['fresh_confirmation']
    if len(fresh)!=16 or len({records[i]['prompt_family'] for i in fresh})!=4 or {records[i]['writer_id'] for i in fresh}-tw:raise ValueError('fixed fresh16/4families known TRAIN writers required')
    if set(fresh)&set(history) or {records[i]['prompt_family'] for i in fresh}&{r['prompt_family'] for r in history.values()} or {normalized_text(records[i]['text']) for i in fresh}&{normalized_text(r['text']) for r in history.values()}:raise ValueError('prior-generator confirmation exposure')
    return dict(lines_disjoint=True,global_train_eval_prompt_and_transcript_disjoint=True,reserved_writers_excluded=True,fresh_family_outside_prior_generator_scope=True)


def duration_model(records,train):
    # This first generator has NO writer-ID conditioning. A pooled duration is
    # therefore fitted from TRAIN text only; writer identity cannot encode a line.
    pooled={i:dict(records[i],writer_id='corpus') for i in train}
    m=fit_duration(pooled,train,['corpus']);m['minimum_policy']='at least number of requested characters; prevents released TextEmbedding truncation';return m


def duration(m,text):
    if not text or len(text)>256:raise ValueError('nonempty <=256-character line required')
    return max(len(text),predict_duration(m,text,'corpus'))


def prepare(repo,root='data',use_cross_attention=False):
    root=Path(root);pool,m,vocab=load_pool(root,POOL_SHA)
    for path,sha in [(root/CODEC,CODEC_SHA),(root/READER_REL,READER_SHA),(root/DATA/'dataset.json',DATASET_SHA)]:
        if file_sha(path)!=sha:raise ValueError('pinned corpus/codec/reader/history source drift')
    historical=json.loads((root/DATA/'dataset.json').read_text())['records'];s=scope(m,historical);records={i:m['records'][i] for ids in s['splits'].values() for i in ids};train=s['splits']['train'];counts=Counter(records[i]['writer_id'] for i in train)
    if len(train)<7000 or len(counts)<150 or min(counts.values())<2:raise ValueError('real broad corpus with repeated writer examples required')
    profile=('actual released DiT backbone with joint cross-attention (MMDiT) + frozen initialized transport codec; NOT paper reproduction'
             if use_cross_attention else
             'actual released DiT backbone with engineering correctness fixes + frozen initialized transport codec; NOT paper reproduction')
    cfg=dict(profile=profile,pool_relative=str(pool.relative_to(root)),pool_manifest_sha256=POOL_SHA,source_h5_sha256=m['lines_h5_sha256'],codec_relative=CODEC,codec_sha256=CODEC_SHA,reader_relative=READER_REL,reader_sha256=READER_SHA,history_relative=DATA+'/dataset.json',history_sha256=DATASET_SHA,
        vocab=vocab,model=dict(dim=384,depth=8,heads=6,dim_head=64,latent_dim=384,text_dim=192,num_text_embedding=len(vocab)+1,text_mask_padding=True,conv_layers=3,ff_mult=4,dropout=.05,long_skip_connection=False,use_cross_attention=use_cross_attention),
        diffusion_steps=1000,timestep_input_divisor=1000.,prediction='x0',schedule='cosine',batch=32,lr=5e-5,min_lr=1e-6,warmup_updates=600,betas=[.9,.99],weight_decay=.0001,clip=1.,max_updates=12000,max_train_wall_seconds=1800,seed=SEED+1,schedule_seed=SEED+2,
        prefix_keep_probability=.7,text_drop_probability=.1,prefix_target_ratio=.3,prefix_loss='valid suffix only when retained; all valid points for pure text/NULL branches',posterior_target='fresh sampled z from cached model means/logvars; all384 channels retained',whitening='TRAIN-only mean and total posterior std=sqrt(variance(mu)+mean(exp(logvar))); floor.1; no held/fresh statistics',
        eval_steps=[0,1000,3000,6000,12000],eval_train_ids=sorted(train,key=lambda i:hashlib.sha256(('train-probe:'+i).encode()).hexdigest())[:16],eval_dev_ids=list(s['splits']['dev'])[:32],
        sampling_steps=50,eval_noise_seeds=[73142,73143],eval_guidance=[1.,2.],free_input='requested text + TRAIN-only pooled duration + random noise; NO target lengths/reference/writerID/trajectory at generation',
        selection='lowest DEV estimated-duration freeCER at guidance1, both declared noise seeds; held/fresh only final selected and last, never selection',stop='first decoded EOC, otherwise estimated-duration cap; no forced EOC/token-clock gate',
        source_volume_read_only=True,output_volume='diffink-experiments-v2',not_promoted=True,no_codec_training=True,no_kl=True,no_ctc_training=True,no_style_training=True,
        caveats=s['caveats']+'; text-only output has generic learned writer distribution, NOT requested writer style. Reference-preserving suffix denoising is TRAIN support only, not claimed arbitrary-text style transfer')
    data=dict(records=records,scope=s,training_writer_counts=dict(counts),history_records_sha256=DATASET_SHA,duration=duration_model(records,train),codec_contract='hand-initialized polyphase40 carried by actual384-channel VAE; not learned semantic bottleneck; untouched frozen checkpoint')
    out=root/'checkpoints/iam_corpus_dit'/time.strftime('%Y%m%d-%H%M%S',time.gmtime());out.mkdir(parents=True,exist_ok=False)
    with tarfile.open(out/'as-run-source.tar.gz','w:gz') as archive:
        for directory in ['iam_tools','model','utils']:
            base=Path(__file__).parent if directory=='iam_tools' else Path(repo)/directory
            for f in sorted(base.rglob('*.py')):
                if '__pycache__' not in f.parts:archive.add(f,arcname=directory+'/'+str(f.relative_to(base)))
    cfg['source_archive_sha256']=file_sha(out/'as-run-source.tar.gz')
    cfg['dataset_sha256']=hashlib.sha256((json.dumps(data,indent=2)+'\n').encode()).hexdigest()
    for name,value in [('config.json',cfg),('dataset.json',data)]: (out/name).write_text(json.dumps(value,indent=2)+'\n')
    return str(out.relative_to(root))
