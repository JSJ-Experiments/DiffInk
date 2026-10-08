"""All-eight causal pen refit/history report; oracle diagnostics clearly separated."""
import html
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .pen_ab import file_sha
from .pen_readout import assert_only_pen_rows_changed
from .report_autoregressive_study import svg
from .history_rollin_study import PARENT,PARENT_SHA


def validate_diagnostic(metrics, ids):
    modes={'true_history','own_xy_true_pen_history','true_xy_own_pen_history','own_both_history'}
    if not metrics['model_reader_unchanged'] or set(metrics['summary'])!=modes:
        raise ValueError('four unchanged-model history interventions required')
    for mode in modes:
        rows=[r for r in metrics['rows'] if r['mode']==mode]
        if len(rows)!=8 or set(r['sample_id'] for r in rows)!=set(ids) or any(not r['source_length'] or not r['not_free_generation'] for r in rows):
            raise ValueError('all8 source-length diagnostic rows, not free generation')
        chars=sum(r['characters'] for r in rows)
        for name in ['true_pen','predicted_pen']:
            if metrics['summary'][mode][name+'_cer']!=sum(r[name+'_errors'] for r in rows)/chars:
                raise ValueError('reader CER must reproduce from all8 rows')
    parity=metrics['actual_free_prefix_parity']
    if len(parity)!=8 or set(r['sample_id'] for r in parity)!=set(ids) or any(r['max_xy_drift']!=0 or r['pen_mismatches']!=0 for r in parity):
        raise ValueError('all-own intervention must match actual free prefix')


def generate(report,root='data'):
    root=Path(root);parent=root/PARENT;base=parent/'history-interventions';out=Path(report)/'pen-history';out.mkdir(exist_ok=False)
    cfg=json.loads((parent/'config.json').read_text());data=json.loads((parent/'dataset.json').read_text());ids=data['training_ids']
    if file_sha(parent/'no_feedback/checkpoint-last.pt')!=PARENT_SHA:raise ValueError('original parent required')
    original=torch.load(parent/'no_feedback/checkpoint-last.pt',map_location='cpu',weights_only=False)['model_state_dict']
    names=['pen-readout-refit','pen-readout-refit-continued-recovered'];results={}
    for name in names:
        folder=base/name;r=json.loads((folder/'result.json').read_text());saved=torch.load(folder/'checkpoint-selected.pt',map_location='cpu',weights_only=False)
        assert_only_pen_rows_changed(original,saved['model_state_dict'])
        if not r['teacher_xy_bit_identical'] or not r['only_pen_rows_changed'] or not r['reader_unchanged'] or len(r['reading'])!=8:raise ValueError('frozen body/XY and reader invariants required')
        if len(r['metadata']['train_ids'])!=8 or set(r['metadata']['train_ids'])!=set(ids):raise ValueError('exact TRAIN8 pen scope required')
        if r['teacher_predicted_pen_cer']!=sum(x['errors'] for x in r['reading'])/sum(x['characters'] for x in r['reading']):raise ValueError('all8 teacher reading CER required')
        results[name]=dict(result=r,checkpoint_sha256=file_sha(folder/'checkpoint-selected.pt'))
    intervention_paths={'original':base,'refitted6000':base/names[-1]/'interventions'};interventions={}
    for name,folder in intervention_paths.items():
        r=json.loads((folder/'metrics.json').read_text());validate_diagnostic(r,ids)
        if file_sha(folder/'interventions.h5')!=r['packed_h5_sha256']:raise ValueError('packed intervention output drift')
        interventions[name]=r
    comparison=[]
    for name,row in results.items():
        r=row['result'];comparison.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['metadata']['head_steps'],r['selected_step'],r['teacher']['min_pen_f1'],f"{100*r['teacher_predicted_pen_cer']:.2f}%",f"{100*r['free']['correct']['cer']:.2f}%",r['teacher_xy_bit_identical']])+'</tr>')
    tables=[]
    for name,r in interventions.items():
        rows=[]
        for mode,summary in r['summary'].items():
            g=summary['teacher_history_metrics'];rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [mode,f"{100*summary['true_pen_cer']:.2f}%",f"{100*summary['predicted_pen_cer']:.2f}%",f"{g['x_rmse']:.5f}",f"{g['y_rmse']:.5f}"])+'</tr>')
        tables.append('<h2>'+name+' selective histories — SOURCE LENGTH, NOT generation</h2><table><tr><th>History</th><th>TRUE rendering pen CER</th><th>PREDICTED rendering pen CER</th><th>X RMSE</th><th>Y RMSE</th></tr>'+''.join(rows)+'</table>')
    gallery=[]
    with h5py.File(root/cfg['source_data']/'source.h5') as src,h5py.File(parent/'no_feedback/evaluation-3000.h5') as before:
        for sid in ids:
            truth=src[sid]['target'][:];entries=[('IAM source',truth),('Original parent genuine FREE',before['free/correct/'+sid]['points'][:])]
            for name in names:
                r=results[name]['result'];folder=base/name
                # Historical cloned filename is 2000 even for actual headstep6000.
                with h5py.File(folder/'evaluation-2000.h5') as hf:
                    entries.extend([(f"Head {r['metadata']['head_steps']} TRUE-HISTORY / PREDICTED-PEN — NOT generation",hf['teacher/'+sid]['points'][:]),(f"Head {r['metadata']['head_steps']} genuine FREE",hf['free/correct/'+sid]['points'][:])])
            with h5py.File(base/names[-1]/'interventions/interventions.h5') as hf:
                for mode in ['true_xy_own_pen_history','own_xy_true_pen_history','own_both_history']:
                    q=hf[mode+'/'+sid]['points'][:];entries.append((mode+' / TRUE rendering pens — SOURCE LENGTH diagnostic, NOT generation',np.c_[q[:,:2],truth[:,2:]]))
            chunks=[]
            for j,(label,q) in enumerate(entries):
                f=sid+f'-{j}.svg';(out/f).write_text(svg(q));chunks.append('<p>'+html.escape(label)+'</p><div class="strip"><img src="'+f+'"></div>')
            gallery.append('<h2>'+html.escape(sid+' | '+data['records'][sid]['text'])+'</h2>'+''.join(chunks))
    summary=dict(results=results,interventions=interventions,parent_sha256=PARENT_SHA,all_nonpen_parameters_independently_verified_unchanged=True,all8_visual_scope=True,no_new_confirmation=True,not_promoted=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Pen-head/history causal audit</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto}.strip{overflow:auto;border:1px solid #ddd}.strip img{max-width:none;display:block}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}</style><h1>Pen errors matter, but continuous own-history instability remains</h1><p>Original step3000 body/XY frozen; train only three pen readout rows. All8 TRAIN lines /7writers. Adam(.9,.999) LR1e-3 gamma2 bounded weights1/3.87668/8, clip5, all2932 realpoints. No OCR training, freeCER selection, smoothing or newtext. The head6000 continuation restores the head2000 optimizer. Previous attempt SIGTERM143 after printed5800/logged5957 had no checkpoint; incomplete outputs preserved, deterministic recovery from saved2000 completed6000. Historical eval-2000 filenames in6000 folder are a cloned-label quirk; metadata records actual6000 steps. Teacher XY is CPU bit-identical; all nonpen parameters independently checked against original.</p><p>Head2000 improves genuine freeTRAIN CER45.90→35.25%; head6000 improves teacher minF1 to.983 and teacher predicted-pen CER to0%, but freeCER36.48% (slightly worse than2000). No exact free prompts. Further familiar-head polishing is NOT the answer. Head6000 oracle-XY/own-pen-history renders with readerCER0%; own-XY/true-pen-history still17.62% TRUE-rendering-pen CER. This causally isolates continuous history errors; it does not prove the text clock is correct or new-text composition works.</p><p>Source-length interventions use selective oracle histories and ignore learned stopping: NOT genuine free generation. Actual free rows use text/writer only, common2048point cap, learned firstEOC; no target point length. Fixed100px/model-unit marker-free SVG, correct up-positiveY, exact aspect, no smoothing/fitting.</p><table><tr><th>Head updates</th><th>Selected</th><th>Teacher minF1</th><th>Teacher predicted-pen CER</th><th>Genuine freeTRAIN CER</th><th>Teacher XY identical</th></tr>'''+''.join(comparison)+'</table><p><a href="summary.json">Full metrics and provenance</a></p>'+''.join(tables)+''.join(gallery))
    return str(out)
