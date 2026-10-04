"""Explicit experimental IAM normalization, per-stroke RDP, DiffInk N×5 encoding."""
import numpy as np

NORMALIZATION = 'iam-line-height-v1-experimental'


def rdp_indices(xy, epsilon):
    """Iterative point-to-segment RDP. Retain endpoints and original order."""
    xy = np.asarray(xy, dtype=np.float64)
    if not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('epsilon must be finite and nonnegative')
    if len(xy) < 3:
        return np.arange(len(xy), dtype=np.int64)
    keep = {0, len(xy)-1}
    stack = [(0, len(xy)-1)]
    while stack:
        a,b = stack.pop()
        if b-a < 2:
            continue
        delta = xy[b]-xy[a]
        length2 = float(delta @ delta)
        inner = xy[a+1:b]
        if length2 == 0:
            dist = np.linalg.norm(inner-xy[a], axis=1)
        else:
            t = np.clip((inner-xy[a]) @ delta / length2, 0, 1)
            dist = np.linalg.norm(inner-(xy[a]+t[:,None]*delta), axis=1)
        at = int(np.argmax(dist))
        if dist[at] > epsilon:
            index = a+1+at
            keep.add(index)
            stack.extend(((a,index),(index,b)))
    return np.array(sorted(keep), dtype=np.int64)


def convert(sample, height=100.0, epsilon=0.5):
    if not np.isfinite(height) or height <= 0:
        raise ValueError('height must be finite and positive')
    raw = [np.asarray(s, dtype=np.float64) for s in sample['strokes']]
    if not raw or any(s.ndim != 2 or s.shape[1] != 3 or not len(s) for s in raw):
        raise ValueError('expected nonempty strokes of x,y,time triples')
    if not all(np.isfinite(s).all() for s in raw):
        raise ValueError('nonfinite source')
    all_xy = np.concatenate(raw)[:,:2]
    lo,hi = all_xy.min(axis=0),all_xy.max(axis=0)
    span = float(hi[1]-lo[1])
    if span <= 0:
        raise ValueError('degenerate line height')
    scale = height/span
    # Positive x offset prevents any real end point being the padding sentinel.
    normalized=[]
    for s in raw:
        s=s.copy()
        s[:,0]=(s[:,0]-lo[0])*scale+1
        s[:,1]=(hi[1]-s[:,1])*scale  # IAM down-positive → model up-positive.
        normalized.append(s)
    retained = [rdp_indices(s[:,:2],epsilon) for s in normalized]
    simplified = [s[idx] for s,idx in zip(normalized,retained)]
    seq=np.zeros((sum(map(len,simplified)),5),dtype=np.float32)
    ends=[]
    cursor=0
    for stroke in simplified:
        n=len(stroke)
        seq[cursor:cursor+n,:2]=stroke[:,:2]
        seq[cursor:cursor+n,2]=1
        seq[cursor+n-1,2:]=[0,1,0]  # pen-up follows the final point of this stroke.
        cursor+=n
        ends.append(cursor)  # EXCLUSIVE ends, suitable for prefix slicing.
    # No internal character ends can be inferred from IAM line-stroke XML.
    # Adaptation: use the third class only for the final line endpoint.
    seq[-1,2:]=[0,0,1]
    metadata={'normalization':NORMALIZATION,'paper_equivalent':False,'height':height,
              'rdp_epsilon':epsilon,'rdp_distance':'point-to-segment',
              'scale':scale,'raw_min':lo.tolist(),'raw_max':hi.tolist(),
              'x_offset':1.0,'y_axis':'up','coordinate_mode':'absolute',
              'end_state_policy':'line-final-only; no internal character alignment',
              'raw_points':len(all_xy),'processed_points':len(seq)}
    validate_sequence(seq,ends)
    return seq,np.array(ends,dtype=np.int64),normalized,simplified,metadata


def validate_sequence(seq, ends):
    if seq.ndim != 2 or seq.shape[1] != 5 or not len(seq) or not np.isfinite(seq).all():
        raise ValueError('invalid N×5 sequence')
    if not np.isin(seq[:,2:], [0,1]).all() or not np.all(seq[:,2:].sum(axis=1)==1):
        raise ValueError('pen states must be one-hot')
    if np.any(np.all(seq == [0,0,0,0,1],axis=1)):
        raise ValueError('real point collides with padding sentinel')
    observed=np.flatnonzero(seq[:,2]==0)+1
    if not np.array_equal(observed,ends) or ends[-1]!=len(seq):
        raise ValueError('stroke boundaries inconsistent')


def sequence_strokes(seq):
    """Decode encoded states, rather than relying on original stroke lists."""
    result=[]
    start=0
    for i,p in enumerate(seq):
        if p[2] == 0:
            result.append(np.column_stack((seq[start:i+1,:2],np.zeros(i+1-start))))
            start=i+1
    if start!=len(seq):
        raise ValueError('missing final boundary')
    return result
