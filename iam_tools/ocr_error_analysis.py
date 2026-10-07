"""Character-weighted descriptive reader errors, not alignment ground truth."""
from collections import Counter
import string
from .inkvae import edit_distance


def edit_trace(target,predicted):
    """One minimum-cost Levenshtein path; ties prefer substitution, deletion, insertion.

    Counts are descriptive; ambiguous edit attribution is NOT true CTC alignment.
    """
    n,m=len(target),len(predicted);d=[list(range(m+1))]+[[i]+[0]*m for i in range(1,n+1)]
    for i in range(1,n+1):
        for j in range(1,m+1):d[i][j]=min(d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(target[i-1]!=predicted[j-1]))
    i,j=n,m;ops=[]
    while i or j:
        if i and j and d[i][j]==d[i-1][j-1]+(target[i-1]!=predicted[j-1]):
            if target[i-1]!=predicted[j-1]:ops.append(('substitute',target[i-1],predicted[j-1]))
            i-=1;j-=1
        elif i and d[i][j]==d[i-1][j]+1:ops.append(('delete',target[i-1],''));i-=1
        else:ops.append(('insert','',predicted[j-1]));j-=1
    return list(reversed(ops))


def analyze(rows,ids,records):
    if not ids or len(set(ids))!=len(ids):raise ValueError('unique nonempty reporting IDs required')
    by={r['sample_id']:r for r in rows}
    if len(by)!=len(rows) or not set(ids)<=set(by):raise ValueError('unique complete evaluation rows required')
    counts=Counter();pairs=Counter();writers={};errors=0;characters=0;lower=0;stripped=0;stripped_n=0;line_rows=[]
    clean=lambda x:''.join(c for c in x.lower() if c not in string.punctuation)
    for sid in ids:
        r=by[sid];target=r['text'];pred=r['mu']['decoded'];rec=records[sid]
        if target!=rec['text'] or not target:raise ValueError('nonempty pinned transcript required')
        ops=edit_trace(target,pred);e=len(ops)
        if e!=r['mu']['errors'] or e!=edit_distance(target,pred):raise ValueError('recorded edit distance mismatch')
        counts.update(op[0] for op in ops);pairs.update((a,b) for op,a,b in ops if op=='substitute')
        errors+=e;characters+=len(target);lower+=edit_distance(target.lower(),pred.lower());stripped+=edit_distance(clean(target),clean(pred));stripped_n+=len(clean(target))
        w=writers.setdefault(str(rec['writer_id']),dict(lines=0,errors=0,characters=0));w['lines']+=1;w['errors']+=e;w['characters']+=len(target)
        line_rows.append(dict(sample_id=sid,writer_id=str(rec['writer_id']),text=target,decoded=pred,errors=e,characters=len(target)))
    for w in writers.values():w['cer']=w['errors']/w['characters']
    return dict(lines=len(ids),errors=errors,characters=characters,cer=errors/characters,lowercase_cer_same_denominator=lower/characters,
      lowercase_no_ascii_punctuation_cer=stripped/stripped_n if stripped_n else None,edit_counts=dict(counts),top_substitutions=[dict(target=a,predicted=b,count=v) for (a,b),v in pairs.most_common(20)],per_writer=writers,
      worst_lines=sorted(line_rows,key=lambda r:(-r['errors'],r['sample_id']))[:12],
      caveat='One minimum-cost character edit trace, not true character/stroke/CTC alignment. Normalized CER is diagnostic only: labels/evaluation objective remain verbatim.')
