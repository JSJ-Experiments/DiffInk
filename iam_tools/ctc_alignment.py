"""Forced CTC Viterbi alignment: weak reader timing, NOT IAM character borders.

Frame spacing is four processed trajectory indices, not uniform physical time.
BiGRU lookahead and delayed dots/crosses can displace label emissions. Intended
for diagnosing/weakly supervising TRAIN alignment, never for exact boundaries.
"""
import numpy as np


def forced_ctc(log_probs, targets, blank=0):
    p=np.asarray(log_probs,dtype=np.float64);labels=np.asarray(targets)
    if p.ndim!=2 or not len(p) or p.shape[1]<2 or not np.isfinite(p).all():
        raise ValueError('finite nonempty frames x classes log probabilities required')
    if labels.ndim!=1 or not len(labels) or not np.issubdtype(labels.dtype,np.integer):
        raise ValueError('nonempty integer target labels required')
    if type(blank) is not int or not 0<=blank<p.shape[1] or (labels==blank).any() or (labels<0).any() or (labels>=p.shape[1]).any():
        raise ValueError('valid distinct blank and covered nonblank targets required')
    if len(p)<len(labels)+int((labels[1:]==labels[:-1]).sum()):
        raise ValueError('exact CTC condition not feasible')
    symbols=np.full(2*len(labels)+1,blank,dtype=np.int64);symbols[1::2]=labels
    scores=np.full(len(symbols),-np.inf);scores[0]=p[0,blank];scores[1]=p[0,labels[0]]
    back=np.zeros((len(p),len(symbols)),dtype=np.int64)
    skip=(symbols!=blank)&np.r_[False,False,symbols[2:]!=symbols[:-2]]
    for t in range(1,len(p)):
        choices=np.stack((scores,np.r_[-np.inf,scores[:-1]],np.r_[-np.inf,-np.inf,scores[:-2]]))
        choices[2,~skip]=-np.inf
        transition=choices.argmax(0);back[t]=np.arange(len(symbols))-transition
        scores=choices[transition,np.arange(len(symbols))]+p[t,symbols]
    state=len(symbols)-2+int(scores[-1]>scores[-2]);score=float(scores[state])
    if not np.isfinite(score):raise ValueError('no feasible forced CTC path')
    path=np.empty(len(p),dtype=np.int64);path[-1]=state
    for t in range(len(p)-1,0,-1):path[t-1]=back[t,path[t]]
    token=np.where(path%2==1,(path-1)//2,-1)
    return dict(states=path,emissions=symbols[path],token_indices=token,log_probability=score,
                token_frames=[np.flatnonzero(token==j).tolist() for j in range(len(labels))])
