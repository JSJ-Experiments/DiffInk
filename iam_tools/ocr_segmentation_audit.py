"""CPU-only audit of OCR frame granularity against public IAM character GT.

The external annotations are intentionally NOT vendored. This helper consumes
the Character Queries IAM-OnDB segmented JSON files and maps their raw
point/spike locations through our existing normalization + RDP policy. It does
not train or mutate the codec, OCR head, pool, or source annotations.
"""
import argparse,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .preprocess import rdp_indices


def load_annotations(directory):
    result={}
    for p in sorted(Path(directory).glob('*_segmented.json')):
        for name,value in json.loads(p.read_text()).items():
            sid=Path(name).stem
            if sid in result:raise ValueError('duplicate annotation: '+sid)
            result[sid]=value
    if not result:raise ValueError('no *_segmented.json annotations found')
    return result


def raw_strokes(path):
    root=ET.parse(path).getroot();result=[]
    for node in root.iter():
        if node.tag.rsplit('}',1)[-1]!='Stroke':continue
        points=np.asarray([[float(q.attrib[k]) for k in ('x','y','time')]
            for q in node if q.tag.rsplit('}',1)[-1]=='Point'],dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=3 or not len(points):raise ValueError('invalid stroke XML: '+str(path))
        result.append(points)
    if not result:raise ValueError('no strokes: '+str(path))
    return result


def retained_raw_indices(strokes,height=100.,epsilon=.5):
    """Return flattened raw-point indices retained by the current per-stroke RDP."""
    all_xy=np.concatenate(strokes)[:,:2];lo,hi=all_xy.min(0),all_xy.max(0)
    if hi[1]<=lo[1]:raise ValueError('degenerate line height')
    scale=height/(hi[1]-lo[1]);offsets=np.cumsum([0]+[len(x) for x in strokes[:-1]])
    result=[]
    for si,stroke in enumerate(strokes):
        xy=stroke[:,:2].copy();xy[:,0]=(xy[:,0]-lo[0])*scale+1.;xy[:,1]=(hi[1]-xy[:,1])*scale
        result.extend(int(offsets[si]+j) for j in rdp_indices(xy,epsilon))
    return np.asarray(result,dtype=np.int64)


def map_spikes_to_retained(spikes,retained):
    spikes=np.asarray(spikes,dtype=np.int64);retained=np.asarray(retained,dtype=np.int64)
    if spikes.ndim!=1 or retained.ndim!=1 or not len(retained):raise ValueError('1-D nonempty indices required')
    # Annotation count is tiny relative to points; explicit nearest mapping
    # makes ties deterministic (earlier retained point wins).
    mapped=np.asarray([int(np.argmin(np.abs(retained-x))) for x in spikes],dtype=np.int64)
    displacement=np.abs(retained[mapped]-spikes)
    return mapped,displacement


def frame_collisions(mapped,points_per_frame):
    if points_per_frame<1:raise ValueError('positive points_per_frame required')
    mapped=np.asarray(mapped,dtype=np.int64);frames=mapped//points_per_frame
    _,counts=np.unique(frames,return_counts=True);bad=counts[counts>1]
    return dict(has_collision=bool(len(bad)),excess=int(np.maximum(counts-1,0).sum()),
        chars_in_collided_frames=int(bad.sum()),max_chars_per_frame=int(counts.max()) if len(counts) else 0)


def ink_range_labels(annotation,strokes):
    """Assign annotated character indices to raw points; spaces naturally have none."""
    offsets=np.cumsum([0]+[len(x) for x in strokes[:-1]]);labels=np.full(sum(map(len,strokes)),-1,dtype=np.int64)
    text='';index=0
    for segment in annotation['gt_segmentation']['segments']:
        substring=segment['substring']
        for local,char in enumerate(substring):
            text+=char
            # Existing IAM GT uses character segments. If a future annotation
            # groups chars into one ink range, do not invent an internal split.
            if local==0:
                for r in segment.get('inkRanges',[]):
                    for si in range(int(r['startStroke']),int(r['endStroke'])+1):
                        a=int(r['startPoint']) if si==int(r['startStroke']) else 0
                        b=int(r['endPoint']) if si==int(r['endStroke']) else len(strokes[si])-1
                        labels[offsets[si]+a:offsets[si]+b+1]=index
            index+=1
    return text,labels


def mixed_frame_fraction(labels,points_per_frame):
    labels=np.asarray(labels,dtype=np.int64);mixed=frames=0;maximum=0
    for start in range(0,len(labels),points_per_frame):
        unique=np.unique(labels[start:start+points_per_frame]);unique=unique[unique>=0]
        frames+=1;mixed+=len(unique)>1;maximum=max(maximum,len(unique))
    return dict(fraction=mixed/frames if frames else 0.,mixed_frames=mixed,frames=frames,max_chars_per_frame=maximum)


def audit(pool_root,raw_root,gt_dir,splits=('large_train','dev','held_out'),frame_sizes=(8,4,2,1)):
    pool=Path(pool_root);raw=Path(raw_root);manifest=json.loads((pool/'manifest.json').read_text());annotations=load_annotations(gt_dir)
    result=dict(annotation_lines=len(annotations),frame_sizes=list(frame_sizes),splits={})
    for split in splits:
        ids=manifest['splits'][split];rows=[];collision={str(f):dict(lines=0,excess=0,chars_in_collided_frames=0,max_chars_per_frame=0) for f in frame_sizes}
        displacement=[];exact=0
        for sid in ids:
            if sid not in annotations:continue
            record=manifest['records'][sid];annotation=annotations[sid]
            if annotation.get('ctc_spike_symbols')!=record['text']:continue
            # Copied legacy hold-out records predate raw_xml provenance in the
            # pool manifest; IAM line IDs deterministically encode the path.
            form=sid.rsplit('-',1)[0].removesuffix('z')
            rel=record.get('raw_xml') or f"lineStrokes/{sid[:3]}/{form}/{sid}.xml"
            exact+=1;strokes=raw_strokes(raw/rel);retained=retained_raw_indices(strokes)
            mapped,delta=map_spikes_to_retained(annotation['ctc_spike_positions'],retained);displacement.extend(delta.tolist())
            gt_text,raw_labels=ink_range_labels(annotation,strokes)
            if gt_text!=record['text']:continue
            retained_labels=raw_labels[retained];row=dict(sample_id=sid,points=len(retained),characters=len(record['text']),frames={})
            for f in frame_sizes:
                c=frame_collisions(mapped,f);mix=mixed_frame_fraction(retained_labels,f);row['frames'][str(f)]=dict(collision=c,mixed=mix)
                dst=collision[str(f)];dst['lines']+=int(c['has_collision']);dst['excess']+=c['excess'];dst['chars_in_collided_frames']+=c['chars_in_collided_frames'];dst['max_chars_per_frame']=max(dst['max_chars_per_frame'],c['max_chars_per_frame'])
            rows.append(row)
        q=np.asarray(displacement,dtype=np.float64)
        result['splits'][split]=dict(requested=len(ids),annotation_overlap=sum(i in annotations for i in ids),exact_transcript=exact,analyzed=len(rows),
            spike_to_retained_raw_index_abs=dict(p50=float(np.quantile(q,.5)) if len(q) else None,p90=float(np.quantile(q,.9)) if len(q) else None,p99=float(np.quantile(q,.99)) if len(q) else None,max=float(q.max()) if len(q) else None),
            collision=collision,rows=rows)
    return result


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--pool-root',required=True);p.add_argument('--raw-root',required=True);p.add_argument('--gt-dir',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(argv);r=audit(a.pool_root,a.raw_root,a.gt_dir);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n')
    for split,v in r['splits'].items():
        print(split,'analyzed',v['analyzed'],'/',v['requested'])
        for f,c in v['collision'].items():print(' ',f+'pt','collision-lines',c['lines'],'excess',c['excess'])
    return r


if __name__=='__main__':main()
