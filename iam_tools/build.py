"""Build CPU-only, writer-disjoint tiny English HDF5 from persistent raw IAM."""
import argparse
import html
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import h5py
import numpy as np
from PIL import Image
from .preview import writer_map, transcript_lines, parse_line, render
from .preprocess import convert, sequence_strokes, NORMALIZATION


def inventory(raw):
    raw=Path(raw)
    writers=writer_map(raw/'forms.txt')
    known={w.attrib['name'] for w in ET.parse(raw/'writers.xml').getroot().iter('Writer')}
    txt={}
    for p in sorted((raw/'ascii').rglob('*.txt')):
        if p.stem in txt: raise ValueError(f'duplicate transcript {p.stem}')
        txt[p.stem]=p
    groups={}
    for p in sorted((raw/'lineStrokes').rglob('*.xml')):
        form,num=p.stem.rsplit('-',1)
        groups.setdefault(form,[]).append((int(num),p))
    entries=[]; rejected=[]
    for form,files in sorted(groups.items()):
        try:
            text=transcript_lines(txt[form]);writer=writers[form]
            if writer not in known: raise ValueError('writer missing from writers.xml')
            if sorted(n for n,p in files)!=list(range(1,len(text)+1)):
                raise ValueError('1-based line IDs/CSR count mismatch')
            for n,p in sorted(files):
                entries.append({'id':p.stem,'writer_id':writer,'text':text[n-1],
                                'xml':str(p),'transcript':str(txt[form])})
        except (KeyError,ValueError) as exc:
            rejected.append({'form':form,'reason':str(exc)})
    if not entries: raise ValueError('no safe IAM pairs')
    return entries,rejected


def comparison(sample, seq, out):
    """Render encoded N×5, undo display y orientation, compare at same scale."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    render(sample,out/(sample['id']+'-raw'))
    decoded=sequence_strokes(seq)
    for s in decoded: s[:,1]*=-1
    render({**sample,'strokes':[s.tolist() for s in decoded]},out/(sample['id']+'-processed'))
    before=Image.open(out/(sample['id']+'-raw.png'))
    after=Image.open(out/(sample['id']+'-processed.png'))
    combined=Image.new('RGB',(max(before.width,after.width),before.height+after.height),'white')
    combined.paste(before,(0,0));combined.paste(after,(0,before.height))
    combined.save(out/(sample['id']+'-comparison.png'))


def build(raw, canonical, out, train_size=250, val_size=50, holdout_writers=25, seed=42, height=100.0, epsilon=0.5, val_writers=5):
    if min(train_size,val_size,holdout_writers,val_writers)<=0: raise ValueError('sizes must be positive')
    entries,rejected=inventory(raw)
    all_writers=sorted({e['writer_id'] for e in entries})
    if holdout_writers+val_writers>=len(all_writers): raise ValueError('holdouts must leave training writers')
    rng=random.Random(seed)
    held=set(rng.sample(all_writers,holdout_writers))
    val_held=set(rng.sample(sorted(set(all_writers)-held),val_writers))
    train_pool=[e for e in entries if e['writer_id'] not in held|val_held]
    val_pool=[e for e in entries if e['writer_id'] in val_held]
    # Training-only vocabulary: no val labels used to grow the vocabulary.
    chars=sorted({c for e in train_pool for c in e['text']})
    out=Path(out);canonical=Path(canonical)
    out.mkdir(parents=True,exist_ok=True);canonical.mkdir(parents=True,exist_ok=True)
    (out/'chars.json').write_text(json.dumps({c:i for i,c in enumerate(chars)},ensure_ascii=False,indent=2)+'\n')
    selected={}; dropped=[]; preview_ids=[]; tag_inventory=set();point_counts=[]
    for split,pool,target in [('train',train_pool,train_size),('val',val_pool,val_size)]:
        pool=pool.copy();rng.shuffle(pool);selected[split]=[]
        temp=out/(f'tiny_{split}.h5.tmp')
        with h5py.File(temp,'w') as hf:
            hf.attrs.update({'normalization':NORMALIZATION,'paper_equivalent':False,
                             'alignment':'stroke-only','seed':seed,'training_performed':False})
            for entry in pool:
                try:
                    if set(entry['text'])-set(chars): raise ValueError('validation OOV from training vocabulary')
                    sample=parse_line(entry['xml'],entry['text'],entry['writer_id'])
                    sample['source']['transcript']=entry['transcript']
                    seq,ends,normalized,simplified,meta=convert(sample,height,epsilon)
                    if not 200<=len(seq)<=2000: raise ValueError(f'loader length filter: {len(seq)}')
                    ctc_required=len(sample['text'])+sum(a==b for a,b in zip(sample['text'],sample['text'][1:]))
                    if (len(seq)+7)//8 < ctc_required: raise ValueError('CTC target longer than compressed trajectory')
                    # Internal character endpoints are NOT fabricated.
                    group=hf.create_group(sample['id'])
                    group.attrs['preprocessing']=json.dumps(meta)
                    group.attrs['alignment_mode']='stroke-only'
                    dt=h5py.string_dtype('utf-8')
                    group.create_dataset('writer_id',data=sample['writer_id'],dtype=dt)
                    group.create_dataset('line_text',data=sample['text'],dtype=dt)
                    group.create_dataset('point_seq',data=seq,compression='gzip')
                    group.create_dataset('char_points_idx',data=np.array([],dtype=np.int64))
                    group.create_dataset('stroke_points_idx',data=ends)
                    # Preserve original acquisition times for the retained points.
                    group.create_dataset('point_times',data=np.concatenate(simplified)[:,2])
                    (canonical/(sample['id']+'.json')).write_text(json.dumps(sample,ensure_ascii=False)+'\n')
                    tag_inventory.update(sample['source']['xml_tags'])
                    selected[split].append(sample['id']);point_counts.append(len(seq))
                    if len(preview_ids)<10:
                        comparison(sample,seq,out/'previews');preview_ids.append(sample['id'])
                except (ValueError,KeyError,ET.ParseError) as exc:
                    dropped.append({'id':entry['id'],'reason':str(exc)})
                    continue
                if len(selected[split])>=target: break
            if len(selected[split])<target:
                raise ValueError(f'{split}: only {len(selected[split])}/{target} accepted')
        temp.replace(out/f'tiny_{split}.h5')
    sampled_writer_ids={}
    for split in selected:
        sampled_writer_ids[split]=sorted({e['writer_id'] for e in entries if e['id'] in set(selected[split])})
    writers={'train':sampled_writer_ids['train'],'val':sampled_writer_ids['val'],
             'test':sorted(held),'all_training_writers':sorted(set(all_writers)-held-val_held),
             'all_validation_writers':sorted(val_held)}
    (out/'writers.json').write_text(json.dumps(writers,indent=2)+'\n')
    report={'stage':'tiny-english-loader-gate','training':False,'gpu':False,'paper_equivalent':False,
            'normalization':NORMALIZATION,'height':height,'rdp_epsilon':epsilon,'seed':seed,
            'paired_lines':len(entries),'paired_writers':len(all_writers),'vocabulary_size':len(chars),
            'vocabulary_source':'all safely paired training transcripts excluding test and validation writers',
            'heldout_writers':sorted(held),'validation_writers':sorted(val_held),'sample_ids':selected,'sample_writers':sampled_writer_ids,
            'rejected_forms':rejected,'rejected_candidates':dropped,'xml_tags_observed':sorted(tag_inventory),
            'processed_length_range':[min(point_counts),max(point_counts)],'previews':preview_ids,
            'alignment':'empty char_points_idx; explicit exclusive stroke_points_idx; patched mask required'}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    imgs=''.join(f'<h2>{html.escape(i)}</h2><p>Raw above / encoded N×5 below</p><img style="max-width:100%" src="{i}-comparison.png">' for i in preview_ids)
    (out/'previews/index.html').write_text('<!doctype html><meta charset="utf-8"><title>IAM conversion comparison</title><h1>Experimental normalization + RDP 0.5</h1>'+imgs)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/raw/iam');p.add_argument('--canonical',default='data/canonical/iam/tiny')
    p.add_argument('--out',default='data/diffink/iam');p.add_argument('--train-size',type=int,default=250)
    p.add_argument('--val-size',type=int,default=50);p.add_argument('--holdout-writers',type=int,default=25)
    p.add_argument('--val-writers',type=int,default=5);p.add_argument('--seed',type=int,default=42);p.add_argument('--height',type=float,default=100)
    p.add_argument('--epsilon',type=float,default=0.5)
    args=vars(p.parse_args());print(json.dumps(build(**args),indent=2))

if __name__=='__main__':main()
