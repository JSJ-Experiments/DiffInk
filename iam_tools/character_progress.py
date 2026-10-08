"""Text-only monotonic WEAK reader-emission clock, not glyph segmentation.

Fit positive intervals between forced-reader emission centers with a small ridge
model. Character identities/neighbor identities, writer intercepts and simple
position features are allowed; sample IDs, source lengths and trajectories are
not inference inputs. Coordinates count packed-eight index blocks, NOT physical
seconds. A frozen BiGRU's emission clock can differ from actual glyph borders.
"""
import hashlib
import math
import numpy as np
from .ocr_pool import normalized_text

MODES = ('no_glyph', 'character', 'neighbor')
CLASSES = ('bos', 'eos', 'lower', 'upper', 'digit', 'space', 'punctuation', 'other')
BOS, EOS = '<BOS>', '<EOS>'


def _record(record):
    if not isinstance(record, dict) or not isinstance(record.get('text'), str) or not record['text'] or not isinstance(record.get('writer_id'), str):
        raise ValueError('nonempty transcript and string writer ID required')


def emission_centers(labels, characters, points):
    """Mean (frame+.5)/2 per transcript token; four indices per reader frame.

    Half-frame offset makes first/last intervals strictly positive without
    clipping empirical durations. No interpolation through blanks is used to
    construct the teacher; every token must actually occur in a monotonic path.
    """
    labels = np.asarray(labels)
    if type(characters) is not int or characters < 1 or type(points) is not int or points < 1 or labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('positive transcript/point counts and integer reader path required')
    if len(labels) != (points+3)//4 or (labels < -1).any() or (labels >= characters).any():
        raise ValueError('real four-index reader frames with blank=-1 required')
    emitted = labels[labels >= 0]
    if not len(emitted) or set(emitted) != set(range(characters)) or (np.diff(emitted) < 0).any():
        raise ValueError('monotonic forced path emitting every transcript token required')
    centers = np.array([((np.flatnonzero(labels == j)+.5)/2).mean() for j in range(characters)])
    intervals = np.diff(np.r_[0., centers, (points+7)//8])
    if not np.isfinite(intervals).all() or (intervals <= 0).any():
        raise ValueError('strictly positive emission-clock intervals required')
    return centers, intervals


def form_folds(records, ids, folds=5, seed=57142):
    """Whole-form folds; connect forms sharing normalized transcript text.

    No form or normalized transcript may appear on both sides, even when the
    same prompt has multiple writers or inconsistent corpus form metadata.
    """
    if type(folds) is not int or folds < 2 or not ids or len(set(ids)) != len(ids):
        raise ValueError('unique nonempty sample IDs and at least two folds required')
    parent = {}
    for sid in ids:
        _record(records[sid]); form = records[sid].get('prompt_family')
        if not isinstance(form, str) or not form: raise ValueError('form metadata required')
        parent[form] = form
    def root(form):
        while parent[form] != form:
            parent[form] = parent[parent[form]]; form = parent[form]
        return form
    texts = {}
    for sid in ids:
        form = records[sid]['prompt_family']; text = normalized_text(records[sid]['text'])
        if text in texts:
            a, b = sorted([root(form), root(texts[text])]); parent[b] = a
        else: texts[text] = form
    groups = {form: root(form) for form in parent}
    components = sorted(set(groups.values()), key=lambda s: hashlib.sha256(f'{seed}:{s}'.encode()).hexdigest())
    if len(components) < folds: raise ValueError('insufficient independent form/text components')
    assignment = {c: j % folds for j, c in enumerate(components)}
    return {sid: assignment[groups[records[sid]['prompt_family']]] for sid in ids}


def _category(char):
    if char == BOS: return 'bos'
    if char == EOS: return 'eos'
    if char.islower(): return 'lower'
    if char.isupper(): return 'upper'
    if char.isdigit(): return 'digit'
    if char.isspace(): return 'space'
    if char.isascii(): return 'punctuation'
    return 'other'


def _names(chars, writers, mode):
    names = ['intercept', 'gap_position', 'gap_position_squared', 'log_text_length', 'first_gap', 'last_gap']
    names += ['writer:'+w for w in writers]
    if mode != 'no_glyph':
        for side in (['shared'] if mode == 'character' else ['left','right']):
            names += [side+':class:'+c for c in CLASSES]
            names += [side+':token:'+c for c in [BOS]+chars+[EOS]]
    return names


def features(model, text, writer_id):
    _record(dict(text=text, writer_id=writer_id))
    if model['mode'] not in MODES or model['feature_names'] != _names(model['characters'], model['writers'], model['mode']):
        raise ValueError('explicit character-clock feature contract required')
    names = model['feature_names']; index = {n:j for j,n in enumerate(names)}
    n = len(text); x = np.zeros((n+1, len(names)))
    x[:,0] = 1.; x[:,1] = np.arange(n+1)/n; x[:,2] = x[:,1]**2; x[:,3] = math.log(n)
    x[0,4] = 1.; x[-1,5] = 1.
    if 'writer:'+writer_id in index: x[:,index['writer:'+writer_id]] = 1.
    if model['mode'] != 'no_glyph':
        for j,(left,right) in enumerate(zip([BOS]+list(text),list(text)+[EOS])):
            for side,char in [('left',left),('right',right)]:
                side = 'shared' if model['mode'] == 'character' else side
                scale = .5 if model['mode'] == 'character' else 1.
                x[j,index[side+':class:'+_category(char)]] += scale
                key = side+':token:'+char
                if key in index: x[j,index[key]] += scale
    return x


def fit_clock(records, labels, train_ids, *, mode='neighbor', ridge=10.):
    """Fit exact TRAIN scope; equal line weight, fixed L2=10 by default.

    A global positive smearing factor matches weighted arithmetic gap means.
    No validation labels, target budget, or whole-line text embedding is fitted.
    Rare unseen characters use their category; unknown writers use no intercept.
    """
    if mode not in MODES or not math.isfinite(ridge) or ridge <= 0 or not train_ids or len(set(train_ids)) != len(train_ids) or set(labels) != set(train_ids):
        raise ValueError('positive ridge and exact unique TRAIN-only timing scope required')
    for sid in train_ids: _record(records[sid])
    chars = sorted({c for sid in train_ids for c in records[sid]['text']})
    writers = sorted({records[sid]['writer_id'] for sid in train_ids})
    model = dict(mode=mode, characters=chars, writers=writers, train_ids=list(train_ids), ridge=float(ridge),
                 feature_names=_names(chars,writers,mode), clock='(reader_frame+.5)/2 packed8 index blocks; NOT physical time or glyph borders')
    designs, gaps, weights = [], [], []
    for sid in train_ids:
        record = records[sid]
        _, intervals = emission_centers(labels[sid],len(record['text']),record['points'])
        designs.append(features(model,record['text'],record['writer_id'])); gaps.append(intervals)
        weights.extend([1/len(intervals)]*len(intervals))
    x = np.concatenate(designs); y = np.concatenate(gaps); weight = np.array(weights)
    weight *= len(weight)/weight.sum()
    penalty = np.eye(x.shape[1])*ridge; penalty[0,0] = 0.
    coef = np.linalg.solve(x.T@(x*weight[:,None])+penalty,x.T@(np.log(y)*weight))
    raw = np.exp(x@coef); factor = float((y*weight).sum()/(raw*weight).sum())
    if not np.isfinite(coef).all() or not math.isfinite(factor) or factor <= 0:
        raise FloatingPointError('invalid fitted character clock')
    model.update(coefficients=coef.tolist(), smearing_factor=factor, training_intervals=len(y),
                 definition='positive log-ridge interval clock; equal line weighting rescaled to mean1; global arithmetic smearing; no inference target length')
    return model


def predict_clock(model, text, writer_id):
    """Monotonic centers/positive gaps from ONLY requested text and writer."""
    coef = np.asarray(model['coefficients'])
    x = features(model,text,writer_id)
    if coef.shape != (x.shape[1],) or not np.isfinite(coef).all() or not math.isfinite(model['smearing_factor']) or model['smearing_factor'] <= 0:
        raise ValueError('finite matching fitted clock parameters required')
    log_gap = x@coef
    if not np.isfinite(log_gap).all() or (np.abs(log_gap) > 30).any():
        raise ValueError('requested text outside numerically safe clock range')
    gaps = np.exp(log_gap)*model['smearing_factor']; total = float(gaps.sum())
    return dict(centers=np.cumsum(gaps)[:-1], intervals=gaps, total_blocks=total,
                unknown_characters=sorted(set(text)-set(model['characters'])), known_writer=writer_id in model['writers'])


def monotonic_progress(centers, coordinates):
    """Piecewise-linear token INDEX between weak emission centers, endpoint clamp.

    This is an explicit phase coordinate, NOT proof that the actual pen traces
    character i at that location. Inputs/prefixes do not depend on output budget.
    """
    centers = np.asarray(centers,dtype=float); coordinates = np.asarray(coordinates,dtype=float)
    if centers.ndim != 1 or not len(centers) or not np.isfinite(centers).all() or (centers <= 0).any() or (np.diff(centers) <= 0).any() or not np.isfinite(coordinates).all():
        raise ValueError('finite positive strictly increasing emission centers required')
    return np.interp(coordinates,centers,np.arange(len(centers)),left=0.,right=float(len(centers)-1))
