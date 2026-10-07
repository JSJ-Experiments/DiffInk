"""CPU-only, prompt/text/writer guarded larger IAM OCR supervision pool.

Keeps the original codec/vocabulary/overfit files untouched. Form families remove
IAM writer-version suffixes (a01-000u/w/z). Existing 32 lines remain report-only;
five previously reserved validation writers supply independent OCR selection.
"""
from collections import defaultdict
import hashlib,json,os,random,re,tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from .build import inventory
from .preview import parse_line
from .preprocess import convert,NORMALIZATION
from .pen_ab import file_sha
from .writer_expansion import MANIFEST_SHA


def prompt_family(sid):
    match=re.fullmatch(r'([a-z]\d{2}-\d{3})[a-z]?-(\d{2})',sid)
    if not match:raise ValueError('unknown IAM line/form syntax: '+sid)
    return match[1]


def normalized_text(text):return ' '.join(text.split()).casefold()


def round_robin(entries,seed=42):
    groups=defaultdict(list)
    for e in sorted(entries,key=lambda e:e['id']):groups[e['writer_id']].append(e)
    rng=random.Random(seed);writers=sorted(groups);rng.shuffle(writers)
    for w in writers:rng.shuffle(groups[w])
    index=0
    while any(index<len(groups[w]) for w in writers):
        for w in writers:
            if index<len(groups[w]):yield groups[w][index]
        index+=1


def blocked(entries,writers,families,texts):
    return [e for e in entries if e['writer_id'] not in writers and prompt_family(e['id']) not in families
            and normalized_text(e['text']) not in texts]


def assert_splits(records,splits,test_writers,dev_writers):
    train=[records[i] for i in splits['large_train']];dev=[records[i] for i in splits['dev']];held=[records[i] for i in splits['held_out']]
    if any(not v or len(v)!=len(set(v)) for v in splits.values()):raise ValueError('nonempty unique splits required')
    if not set(splits['small_train'])<=set(splits['large_train']):raise ValueError('small must be nested in large')
    sets=[set(splits[k]) for k in ('large_train','dev','held_out')]
    if any(sets[i]&sets[j] for i in range(3) for j in range(i)):raise ValueError('line overlap')
    train_writers={r['writer_id'] for r in train};small_writers={records[i]['writer_id'] for i in splits['small_train']}
    if train_writers!=small_writers:raise ValueError('small/large writer populations differ')
    if train_writers&(set(test_writers)|set(dev_writers)):raise ValueError('reserved writer leakage')
    if {r['writer_id'] for r in dev}-set(dev_writers):raise ValueError('dev must use reserved writers only')
    if {r['writer_id'] for r in held}&(set(test_writers)|set(dev_writers)):raise ValueError('report writers must remain outside test/dev reservations')
    for other in (dev,held):
        if {r['prompt_family'] for r in train}&{r['prompt_family'] for r in other}:raise ValueError('prompt-family leakage')
        if {normalized_text(r['text']) for r in train}&{normalized_text(r['text']) for r in other}:raise ValueError('exact normalized transcript leakage')
    if {r['prompt_family'] for r in dev}&{r['prompt_family'] for r in held}:raise ValueError('dev/held prompt overlap')
    if {normalized_text(r['text']) for r in dev}&{normalized_text(r['text']) for r in held}:raise ValueError('dev/held exact transcript overlap')
    return dict(lines_disjoint=True,train_dev_writers_disjoint=True,train_eval_prompt_families_disjoint=True,
        train_eval_normalized_transcripts_disjoint=True,small_large_same_writer_population=True,test_writers_excluded=True)


def _build(raw,original,out,train_size=2048,small_size=192,dev_size=128):
    raw=Path(raw);original=Path(original);out=Path(out)
    if file_sha(original/'manifest.json')!=MANIFEST_SHA:raise ValueError('pinned original manifest required')
    old=json.loads((original/'manifest.json').read_text());vocab=json.loads((original/'chars.json').read_text())
    entries,rejected_forms=inventory(raw);by_id={e['id']:e for e in entries}
    test=set(old['heldout_writers']);dev_writers=set(old['excluded_validation_writers'])
    held_ids=old['sample_ids']['val'];held=[by_id[i] for i in held_ids]
    families={prompt_family(e['id']) for e in held};texts={normalized_text(e['text']) for e in held}
    rejected=[];data={};records={}
    def prepare(entry):
        sid=entry['id'];sample=parse_line(entry['xml'],entry['text'],entry['writer_id']);seq,ends,*_=convert(sample,old['height'],old['rdp_epsilon'])
        required=len(entry['text'])+sum(a==b for a,b in zip(entry['text'],entry['text'][1:]))
        if not entry['text']:raise ValueError('empty transcript')
        if set(entry['text'])-set(vocab):raise ValueError('fixed training-vocabulary OOV')
        if not 200<=len(seq)<=2000:raise ValueError('unchanged released loader length filter')
        if (len(seq)+7)//8<required:raise ValueError('exact CTC infeasible')
        data[sid]=(seq,ends)
        records[sid]=dict(writer_id=entry['writer_id'],text=entry['text'],points=len(seq),prompt_family=prompt_family(sid),
            points_sha256=hashlib.sha256(seq.tobytes()).hexdigest(),raw_xml=str(Path(entry['xml']).relative_to(raw)),
            xml_sha256=file_sha(entry['xml']),transcript_sha256=file_sha(entry['transcript']))
    def select(pool,count):
        selected=[]
        for entry in round_robin(pool):
            try:prepare(entry)
            except (ValueError,KeyError,ET.ParseError) as exc:rejected.append(dict(id=entry['id'],reason=str(exc)));continue
            selected.append(entry['id'])
            if len(selected)==count:return selected
        raise ValueError(f'insufficient safe accepted lines: {len(selected)}/{count}')
    dev_candidates=blocked([e for e in entries if e['writer_id'] in dev_writers],test,families,texts)
    dev=select(dev_candidates,dev_size)
    families|={records[i]['prompt_family'] for i in dev};texts|={normalized_text(records[i]['text']) for i in dev}
    train_candidates=blocked(entries,test|dev_writers,families,texts);large=select(train_candidates,train_size)
    small=[e['id'] for e in round_robin([dict(id=i,writer_id=records[i]['writer_id']) for i in large])][:small_size]
    # Existing report lines are copied BITWISE from the original HDF5, never
    # reprocessed or relabeled. Verify pairing with the raw inventory.
    with h5py.File(original/'tiny_val.h5') as hf:
        for sid in held_ids:
            g=hf[sid];e=by_id[sid];seq=g['point_seq'][:]
            if g['writer_id'][()].decode()!=e['writer_id'] or g['line_text'][()].decode()!=e['text']:raise ValueError('held-out pairing changed')
            records[sid]=dict(writer_id=e['writer_id'],text=e['text'],points=len(seq),prompt_family=prompt_family(sid),
                points_sha256=hashlib.sha256(seq.tobytes()).hexdigest(),copied_from_original=True)
            data[sid]=(seq,g['stroke_points_idx'][:])
    splits=dict(small_train=small,large_train=large,dev=dev,held_out=held_ids)
    checks=assert_splits(records,splits,test,dev_writers)
    if len(small)!=small_size:raise ValueError('small quota unmet')
    with h5py.File(out/'lines.h5','w') as hf:
        for sid in sorted(data):
            seq,ends=data[sid];g=hf.create_group(sid);g.create_dataset('point_seq',data=seq,compression='gzip')
            g.create_dataset('stroke_points_idx',data=ends);g.create_dataset('char_points_idx',data=np.array([],dtype=np.int64))
            for key in ('writer_id','text'):g.create_dataset('line_text' if key=='text' else key,data=records[sid][key],dtype=h5py.string_dtype('utf-8'))
        hf.attrs.update(normalization=NORMALIZATION,paper_equivalent=False,training_performed=False)
    (out/'chars.json').write_text(json.dumps(vocab,ensure_ascii=False,indent=2)+'\n')
    manifest=dict(profile='english-frozen-ocr-prompt-guarded-pool',training=False,gpu=False,paper_equivalent=False,
        seed=42,original_manifest_sha256=MANIFEST_SHA,vocab_sha256=file_sha(original/'chars.json'),lines_h5_sha256=file_sha(out/'lines.h5'),
        splits=splits,records=records,test_writers=sorted(test),dev_writers=sorted(dev_writers),checks=checks,
        train_writers=sorted({records[i]['writer_id'] for i in large}),normalization=NORMALIZATION,height=old['height'],rdp_epsilon=old['rdp_epsilon'],
        model_input_scale=.01,min_points=200,max_points=2000,rejected_candidates=rejected,rejected_forms=rejected_forms,
        paired_inventory=len(entries),eligible_train_candidates=len(train_candidates),
        vocabulary_contract='unchanged81-character vocabulary from original broad training transcript inventory, excluding25 test and5 dev writers; not refit on evaluation',
        held_out_policy='original32 seen-writer report lines, excluded prompt families and normalized texts from OCR TRAIN; no checkpoint selection on them',
        dev_policy='128 lines from previously reserved5 writers; writer/prompt-family/normalized-text-disjoint from OCR TRAIN; used for checkpoint selection',
        geometry_pretraining_caveat='frozen codec learned reconstruction on original192; that old geometry pool has prompt overlap with original32. This is OCR supervision isolation, NOT a fully independent pretrained-representation IAM benchmark')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


def build(raw='data/raw/iam',original='data/diffink/iam_overfit',out='data/diffink/iam_ocr_pool',**sizes):
    quota={k:sizes.get(k,d) for k,d in [('train_size',2048),('small_size',192),('dev_size',128)]}
    if any(not isinstance(n,int) or n<1 for n in quota.values()) or quota['small_size']>quota['train_size']:raise ValueError('positive nested quotas required')
    out=Path(out).absolute();raw=Path(raw).resolve();original=Path(original).resolve();resolved=out.resolve()
    if out.is_symlink() or resolved in (Path('/'),Path.home(),Path.cwd()) or any(resolved==p or resolved in p.parents or p in resolved.parents for p in (raw,original)):
        raise ValueError('generated output overlaps protected input/unsafe directory')
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.iam-ocr-pool-',dir=out.parent) as temporary:
        staging=Path(temporary)/'pool';staging.mkdir();manifest=_build(raw,original,staging,**sizes)
        backup=Path(temporary)/'previous'
        if out.exists():os.replace(out,backup)
        try:os.replace(staging,out)
        except BaseException:
            if backup.exists():os.replace(backup,out)
            raise
    return dict(output=str(out),manifest_sha256=file_sha(out/'manifest.json'),splits={k:len(v) for k,v in manifest['splits'].items()},checks=manifest['checks'])


if __name__=='__main__':print(json.dumps(build(),indent=2))
