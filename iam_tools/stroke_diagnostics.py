"""Pen-lift / rendered-stroke diagnostics independent of XY reconstruction error.

For this IAM adaptation state1 ends a stroke and state2 ends the line. Count
state1 explicitly; renderer starts a new stroke AFTER any noncontinue point.
A final pen-up does not create an extra empty stroke. Singleton dots are strokes.
Density comparisons are corpus-relative flags, NOT a requirement to reproduce
one writer's exact count for a stochastic generic-writer generation.
"""
import numpy as np


def statistics(points, characters):
    q=np.asarray(points)
    if q.ndim!=2 or q.shape[1]!=5 or not len(q) or not np.isfinite(q).all() or not isinstance(characters,int) or characters<=0:
        raise ValueError('finite nonempty N,5 trajectory and positive character count required')
    p=q[:,2:]
    if not np.isin(p,[0,1]).all() or not np.equal(p.sum(1),1).all():
        raise ValueError('hard onehot pen states required; do not silently argmax soft fields')
    states=p.argmax(1);ends=np.flatnonzero(states!=0);stops=[int(i)+1 for i in ends]
    if not stops or stops[-1]!=len(q):stops.append(len(q))
    lengths=np.diff([0]+stops);lifts=int((states==1).sum());internal=int((states[:-1]==1).sum())
    return dict(points=len(q),characters=characters,pen_lifts=lifts,
                internal_pen_lifts=internal,rendered_strokes=len(lengths),
                singleton_strokes=int((lengths==1).sum()),
                multi_point_strokes=int((lengths>1).sum()),
                connected_segments=int((states[:-1]==0).sum()),
                eoc_count=int((states==2).sum()),final_eoc=bool(states[-1]==2),
                pen_lifts_per_100_points=100*lifts/len(q),
                pen_lifts_per_character=lifts/characters,
                strokes_per_character=len(lengths)/characters)


def compare(generated,source,characters,estimated_points,found_eoc):
    g=statistics(generated,characters);s=statistics(source,characters)
    if not isinstance(estimated_points,int) or estimated_points<len(generated) or bool(found_eoc)!=g['final_eoc']:
        raise ValueError('saved free EOC/estimated duration must agree with points')
    if g['eoc_count']!=int(found_eoc) or (not found_eoc and len(generated)!=estimated_points):
        raise ValueError('first-EOC stop or full estimated cap required, no internal EOC')
    ratio=lambda x,y:x/y if y>0 else None
    count=ratio(g['pen_lifts'],s['pen_lifts']);density=ratio(g['pen_lifts_per_100_points'],s['pen_lifts_per_100_points'])
    severe=count is not None and density is not None and count<.5 and density<.5
    return dict(generated=g,source=s,pen_lift_count_ratio=count,
                point_normalized_lift_ratio=density,
                character_normalized_lift_ratio=ratio(g['pen_lifts_per_character'],s['pen_lifts_per_character']),
                rendered_stroke_ratio=ratio(g['rendered_strokes'],s['rendered_strokes']),
                generated_source_point_ratio=len(generated)/len(source),
                generated_estimated_cap_ratio=len(generated)/estimated_points,
                stop_stratum='EOC_before_80pct_estimated_cap' if found_eoc and len(generated)<.8*estimated_points else ('EOC_near_estimated_cap' if found_eoc else 'estimated_cap_no_EOC'),
                severe_underlifting=severe,
                criterion='less than half matched-source lifts per character AND per point; corpus-relative quality failure flag, not exact writer-count requirement')


def aggregate(rows):
    if not rows:raise ValueError('nonempty diagnostic rows required')
    def med(values):
        a=[v for v in values if v is not None];return float(np.median(a)) if a else None
    out=dict(lines=len(rows),median_generated_pen_lifts=med(r['generated']['pen_lifts'] for r in rows),
             median_source_pen_lifts=med(r['source']['pen_lifts'] for r in rows),
             median_generated_strokes=med(r['generated']['rendered_strokes'] for r in rows),
             median_source_strokes=med(r['source']['rendered_strokes'] for r in rows),
             median_character_normalized_ratio=med(r['character_normalized_lift_ratio'] for r in rows),
             median_point_normalized_ratio=med(r['point_normalized_lift_ratio'] for r in rows),
             severe_underlifting_lines=sum(r['severe_underlifting'] for r in rows))
    out['corpus_severe_underlifting']=out['median_character_normalized_ratio'] is not None and out['median_point_normalized_ratio'] is not None and out['median_character_normalized_ratio']<.5 and out['median_point_normalized_ratio']<.5
    out['by_stop_stratum']={s:dict(lines=sum(r['stop_stratum']==s for r in rows),severe=sum(r['severe_underlifting'] for r in rows if r['stop_stratum']==s)) for s in sorted({r['stop_stratum'] for r in rows})}
    return out
