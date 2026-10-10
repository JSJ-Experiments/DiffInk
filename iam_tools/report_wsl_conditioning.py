"""CPU-only, fail-closed snapshots of native corpus conditioning experiments.

No model training, oracle trajectory ranking, confirmation probes or smoothing.
Snapshots explicitly distinguish evaluated milestones from terminal results.
"""
import argparse
import hashlib
import html
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from .corpus_dit import learning_rate
from .corpus_dit_study import schedule
from .ocr_context_study import tensor_digest
from .launch_ledger import study_path
from .pen_ab import file_sha
from .report_corpus_dit import _head, trajectory_section, verify_evaluation
from .stroke_diagnostics import aggregate, compare
from .wsl_conditioning import ARMS, models, verify


def verify_updates(cfg, data, rows, through):
    """Verify an immutable completed prefix, not a partially written live tail."""
    if not isinstance(through, int) or not 0 <= through <= cfg['max_updates']:
        raise ValueError('bounded completed milestone required')
    if [r['step'] for r in rows] != list(range(1, through + 1)):
        raise ValueError('complete sequential update prefix required')
    ids = data['scope']['splits']['train']
    lengths = {i: (data['records'][i]['points'] + 7) // 8 for i in ids}
    batches = list(schedule(ids, lengths, cfg['max_updates'], cfg['batch'], cfg['schedule_seed']))
    if [r['sample_ids'] for r in rows] != batches[:through]:
        raise ValueError('declared TRAIN-only bucket order required')
    previous = 0.
    for r in rows:
        fields = ['objective', 'xy16_x0_mse', 'pen24_x0_mse', 'raw_grad_norm', 'train_seconds', 'lr']
        if not np.isfinite([r[k] for k in fields]).all() or min(r[k] for k in fields) < 0:
            raise ValueError('finite nonnegative measured updates required')
        if not math.isclose(r['objective'], .4*r['xy16_x0_mse'] + .6*r['pen24_x0_mse'], rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError('compact40 x0 decomposition must reproduce')
        if r['prefix_retained'] or r['clipped'] != (r['raw_grad_norm'] > cfg['clip']) or r['lr'] != learning_rate(r['step'], cfg):
            raise ValueError('no prefix / actual LR and clipping required')
        if not isinstance(r['text_dropped'], bool) or r['active_tokens'] != sum(lengths[i] for i in r['sample_ids']):
            raise ValueError('actual text-drop and valid token count required')
        if len(r['timesteps']) != len(r['sample_ids']) or any(not isinstance(t, int) or not 0 <= t < cfg['diffusion_steps'] for t in r['timesteps']):
            raise ValueError('bounded per-example diffusion times required')
        if r['train_seconds'] < previous or r['gpu_peak_allocated_bytes'] <= 0:
            raise ValueError('monotonic actual time and measured allocation required')
        previous = r['train_seconds']
        if r['step'] <= 10 or r['step'] % 250 == 0:
            digest = r.get('data_draw_digest', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('actual sampled posterior/noise digest required')
    return hashlib.sha256(json.dumps(batches).encode()).hexdigest()


def verify_matched_draws(logs):
    """Compare actual shared update identities, including sampled noise digests."""
    if len(logs) < 2:
        return dict(compared_updates=0, comparison_pending=True)
    first = next(iter(logs.values()))
    through = min(map(len, logs.values()))
    for other in list(logs.values())[1:]:
        for a, b in zip(first[:through], other[:through]):
            for key in ['step', 'sample_ids', 'timesteps', 'prefix_retained', 'text_dropped', 'active_tokens', 'lr', 'data_draw_digest']:
                if a.get(key) != b.get(key):
                    raise ValueError('matched architecture DATA draws drift: ' + key)
    return dict(compared_updates=through, comparison_pending=False,
                dropout_draws_not_identical=True)


def verify_denoising(diag, cfg, step):
    expected = {(split, sid, t, policy)
                for split, key in [('train', 'eval_train_ids'), ('dev_unseen_text', 'eval_dev_ids')]
                for sid in cfg[key] for t in [0, 10, 100, 500, 900, 999]
                for policy in ['correct', 'rotated_character_order', 'null']}
    rows = diag['rows']
    keys = [(r['split'], r['sample_id'], r['timestep'], r['policy']) for r in rows]
    if diag['step'] != step or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('complete target-informed source/text/time scope required')
    if not diag['shared_device_noise_across_arms'] or 'NOT free generation' not in diag['scope']:
        raise ValueError('target-informed scope must remain explicit')
    for r in rows:
        v = [r[k] for k in ['xy16_mse', 'pen24_mse', 'active40_mse']]
        if not np.isfinite(v).all() or min(v) < 0 or not math.isclose(v[2], .4*v[0]+.6*v[1], rel_tol=2e-6, abs_tol=2e-7):
            raise ValueError('finite target-informed field decomposition required')
    for split in ['train', 'dev_unseen_text']:
        for t in [0, 10, 100, 500, 900, 999]:
            for policy in ['correct', 'rotated_character_order', 'null']:
                sub = [r for r in rows if (r['split'], r['timestep'], r['policy']) == (split, t, policy)]
                for k in ['xy16_mse', 'pen24_mse', 'active40_mse']:
                    if diag['summary'][split][str(t)][policy][k] != float(np.mean([r[k] for r in sub])):
                        raise ValueError('target-informed aggregate must reproduce')


def checked_points(hf, row):
    key = f"{row['split']}/{row['seed']}/{row['guidance']}/{row['policy']}/{row['sample_id']}"
    g = hf[key]
    q = g['points'][:]
    if json.loads(g.attrs['row']) != row or len(q) != row['generated_points']:
        raise ValueError('packed row/point identity mismatch')
    # Hard state and first-EOC/cap validation occurs again in stroke compare.
    return q


def controls(rows):
    out = []
    for split in ['train', 'dev_unseen_text']:
        for policy in ['correct', 'swapped', 'null']:
            sub = [r for r in rows if r['split'] == split and r['policy'] == policy and r['seed'] == 73142 and r['guidance'] == 1.]
            if not sub:
                raise ValueError('matched first-noise/guidance1 text controls required')
            supplied = sum(len(r['conditioning_text']) for r in sub)
            out.append(dict(split=split, policy=policy, evaluations=len(sub),
                            requested_text_cer=sum(r['errors'] for r in sub)/sum(r['characters'] for r in sub),
                            supplied_text_cer=sum(r['errors_against_supplied_text'] for r in sub)/supplied if supplied else None,
                            exact_requested=sum(r['errors'] == 0 for r in sub)))
    return out


def verify_terminal(cfg, result, rows, histories, order):
    if result['arm'] not in ARMS or not rows or result['last_step'] != rows[-1]['step'] or result['training_order_sha256'] != order:
        raise ValueError('terminal arm/order/actual update count required')
    last = result['last_step']
    expected = sorted(set([s for s in cfg['eval_steps'] if s <= last] + [last]))
    if [h['step'] for h in result['history']] != expected or result['history'] != histories:
        raise ValueError('complete terminal evaluation timeline required')
    best = min(histories, key=lambda h: h['aggregate']['dev_unseen_text']['1.0']['cer'])
    if result['best_step'] != best['step'] or result['best_dev_cer'] != best['aggregate']['dev_unseen_text']['1.0']['cer']:
        raise ValueError('DEV-only both-noise guidance1 selection required')
    if result['stop'] not in ['budget_completed', 'wall_limit'] or (result['stop'] == 'budget_completed' and last != cfg['max_updates']) or (result['stop'] == 'wall_limit' and result['train_seconds'] < cfg['max_train_wall_seconds']):
        raise ValueError('bounded actual terminal budget required')
    if result['train_seconds'] != rows[-1]['train_seconds'] or result['clip_fraction'] != sum(r['clipped'] for r in rows)/len(rows):
        raise ValueError('terminal actual time/clipping must reproduce')
    if result['backend'] != cfg['backend'] or not all(result[k] for k in ['no_confirmations_opened', 'not_promoted', 'codec_reader_unchanged']):
        raise ValueError('frozen native no-confirmation scope required')


def generate(relative, arms, steps=None, root='data'):
    torch.set_num_threads(1)
    root = Path(root)
    p = study_path(root, relative, "checkpoints/iam_wsl_conditioning/")
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError('explicit unique native arms required')
    if steps is not None and (not steps or steps != sorted(set(steps)) or min(steps) < 0):
        raise ValueError('explicit ordered unique completed milestones required')
    cfg = json.loads((p/'config.json').read_text())
    data = verify(root, p, cfg)
    logs, stages, terminal, inputs = {}, {}, {}, {}
    pages = {}
    initial_states = {}
    splits = dict(train=cfg['eval_train_ids'], dev_unseen_text=cfg['eval_dev_ids'])
    with h5py.File(root/cfg['pool_relative']/'lines.h5') as source:
        for arm in arms:
            folder = p/arm
            arm_steps = steps
            if arm_steps is None:
                finished = json.loads((folder/'result.json').read_text())
                arm_steps = [h['step'] for h in finished['history']]
                if not arm_steps or arm_steps != sorted(set(arm_steps)):
                    raise ValueError('actual ordered terminal evaluation timeline required')
            # Read only the completed prefix, never parse a concurrent log tail.
            rows = []
            if max(arm_steps):
                with (folder/'metrics.jsonl').open() as log:
                    for line in log:
                        rows.append(json.loads(line))
                        if len(rows) == max(arm_steps):
                            break
            order = verify_updates(cfg, data, rows, max(arm_steps))
            logs[arm] = rows
            inputs[f'{arm}/verified-update-prefix'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
            initial = torch.load(folder/'checkpoint-initial.pt', map_location='cpu', weights_only=True)
            actual = dict(cfg, model=dict(cfg['model'], use_cross_attention=arm == 'joint'))
            if initial['step'] != 0 or initial['arm'] != arm or initial['config'] != actual or initial['source_archive_sha256'] != cfg['source_archive_sha256'] or initial['study_config_sha256'] != file_sha(p/'config.json') or initial['training_order_sha256'] != order or tensor_digest(initial['model_state_dict']) != initial['initial_model_digest']:
                raise ValueError('actual native initializer/checkpoint provenance required')
            initial_states[arm] = initial['model_state_dict']
            inputs[f'{arm}/checkpoint-initial.pt'] = file_sha(folder/'checkpoint-initial.pt')
            del initial
            histories = []
            for step in arm_steps:
                ev = json.loads((folder/f'eval-{step}.json').read_text())
                if ev['step'] != step:
                    raise ValueError('evaluation step mismatch')
                verify_evaluation(cfg, data, ev, splits, True)
                if any(r.get('conditioning_characters_truncated') != 0 for r in ev['rows']):
                    raise ValueError('untruncated actual supplied text required')
                hfpath = folder/f'evaluation-{step}.h5'
                if file_sha(hfpath) != ev['packed_h5_sha256']:
                    raise ValueError('immutable evaluated H5 hash drift')
                audit = json.loads((folder/f'stroke-audit-{step}.json').read_text())
                diag = json.loads((folder/f'denoising-{step}.json').read_text())
                verify_denoising(diag, cfg, step)
                sections, sr, paired = [], [], {}
                with h5py.File(hfpath) as hf:
                    keys = []
                    hf.visititems(lambda name, obj: keys.append(name) if isinstance(obj, h5py.Group) and 'points' in obj else None)
                    expected = {f"{r['split']}/{r['seed']}/{r['guidance']}/{r['policy']}/{r['sample_id']}" for r in ev['rows']}
                    if set(keys) != expected:
                        raise ValueError('packed scope must match every declared evaluation')
                    for r in ev['rows']:
                        q = checked_points(hf, r)
                        truth = source[r['sample_id']]['point_seq'][:]
                        if hashlib.sha256(truth.tobytes()).hexdigest() != data['records'][r['sample_id']]['points_sha256']:
                            raise ValueError('offline source geometry identity drift')
                        counts = compare(q, truth, r['characters'], 8*r['estimated_blocks'], r['found_eoc'])
                        sr.append(dict(split=r['split'], policy=r['policy'], guidance=r['guidance'], seed=r['seed'], sample_id=r['sample_id'], **counts))
                        if r['seed'] == cfg['eval_noise_seeds'][0] and r['guidance'] == 1.:
                            paired[(r['split'], r['sample_id'], r['policy'])] = trajectory_section(r, q)
                        sections.append('<p>Pen lifts '+str(counts['generated']['pen_lifts'])+' (IAM context '+str(counts['source']['pen_lifts'])+'); severe under-lifting '+str(counts['severe_underlifting'])+'</p>'+trajectory_section(r, q))
                groups = {}
                for r in sr:
                    groups.setdefault(f"{r['split']}/g{r['guidance']}/{r['policy']}", []).append(r)
                groups = {k: aggregate(v) for k, v in groups.items()}
                if audit != dict(step=step, arm=arm, rows=sr, groups=groups, offline_counts_not_generation_inputs=True):
                    raise ValueError('saved stroke audit must independently reproduce')
                stages[f'{arm}/{step}'] = dict(aggregate=ev['aggregate'], controls=controls(ev['rows']), strokes=groups, target_informed_denoising=diag['summary'])
                histories.append(dict(step=step, aggregate=ev['aggregate']))
                pages[f'gallery-{arm}-{step}.html'] = _head(f'{arm} step{step}')+'<a href="index.html">Summary</a><h1>'+html.escape(arm)+f' step{step}</h1><p>All fixed TRAIN16/DEV32, both noises/guidances, matched controls. Fixed100px/unit, marker-free, no smoothing; source counts are offline context, NOT generation inputs. No exemplar-RMSE ranking.</p>'+''.join(sections)
                pairparts = []
                for split, ids in splits.items():
                    for sid in ids:
                        pairparts.append('<h2>'+html.escape(split+' | '+sid)+'</h2><div class=controls>'+''.join('<div>'+paired[(split, sid, policy)]+'</div>' for policy in ['correct', 'swapped', 'null'])+'</div>')
                pages[f'controls-{arm}-{step}.html'] = _head(f'{arm} step{step} matched text controls')+'<style>.controls{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}</style><a href="index.html">Summary</a><h1>Correct / swapped / NULL</h1><p>Matched starting noise and original requested duration, all48 primary-seed guidance1 probes. Horizontal scroll per strip; fixed100px/unit, no smoothing or fit-to-width. Swapped output should follow supplied text, not original request. Neither count nor sensitivity establishes readable generation.</p>'+''.join(pairparts)
                for name in [f'eval-{step}.json', f'evaluation-{step}.h5', f'stroke-audit-{step}.json', f'denoising-{step}.json']:
                    inputs[f'{arm}/{name}'] = file_sha(folder/name)
            result_path = folder/'result.json'
            if result_path.exists():
                result = json.loads(result_path.read_text())
                # A terminal result does not turn a selected-prefix snapshot into a final report.
                if result['last_step'] == max(arm_steps) and [h['step'] for h in result['history']] == arm_steps:
                    verify_terminal(cfg, result, rows, histories, order)
                    if result['arm'] != arm or result['initial_model_digest'] != tensor_digest(initial_states[arm]):
                        raise ValueError('terminal initializer/arm must match actual initial checkpoint')
                    for name, sha in [('checkpoint-best.pt', result['selected_sha256']), ('checkpoint-last.pt', result['last_sha256'])]:
                        if file_sha(folder/name) != sha:
                            raise ValueError('terminal checkpoint hash drift')
                        saved = torch.load(folder/name, map_location='cpu', weights_only=True)
                        wanted_step = result['best_step'] if name == 'checkpoint-best.pt' else result['last_step']
                        if saved['config'] != actual or saved['arm'] != arm or saved['step'] != wanted_step or saved['initial_model_digest'] != result['initial_model_digest'] or saved['source_archive_sha256'] != cfg['source_archive_sha256'] or saved['training_order_sha256'] != order or saved['study_config_sha256'] != file_sha(p/'config.json'):
                            raise ValueError('terminal checkpoint/protocol identity drift')
                        del saved
                    terminal[arm] = result
    # Reproduce both native CPU initializers; they are not T4 regeneration.
    a, b, identity = models(cfg['model'], cfg['init_seed'])
    for arm, m in [('concat', a), ('joint', b)]:
        if arm in initial_states and tensor_digest(m.state_dict()) != tensor_digest(initial_states[arm]):
            raise ValueError('native mapped initializer must reproduce bitwise')
    matched = verify_matched_draws(logs)
    summary = dict(config=cfg, stages=stages, terminal_results=terminal, initialization=identity,
                   matched_data_draws=matched, input_hashes=inputs,
                   partial=len(terminal) != len(ARMS), omitted_arms=[a for a in ARMS if a not in arms],
                   visual_inspection_pending=True, not_promoted=True, no_confirmations_opened=True,
                   NOT_semantic_InkVAE_or_paper_reproduction=True,
                   requested_generic_handwriting_not_exemplar_reconstruction=True)
    # No report directory created before all requested evidence passes.
    out = p/('report-'+time.strftime('%Y%m%d-%H%M%S', time.gmtime()))
    out.mkdir(exist_ok=False)
    for name, content in pages.items():
        (out/name).write_text(content)
    for arm, rows in logs.items():
        (out/f'verified-update-prefix-{arm}.json').write_text(json.dumps(rows, sort_keys=True)+'\n')
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    table = []
    for stage, item in stages.items():
        arm, step = stage.split('/')
        for split, values in item['aggregate'].items():
            for g, v in values.items():
                stroke = item['strokes'][f'{split}/g{g}/correct']
                fields = [split, g, f"{100*v['cer']:.2f}%", v['exact'], v['evaluations'], v['missing_eoc'], stroke['median_generated_pen_lifts'], stroke['median_source_pen_lifts'], stroke['severe_underlifting_lines']]
                table.append(f'<tr><td><a href="gallery-{arm}-{step}.html">{stage}</a></td>'+''.join('<td>'+html.escape(str(f))+'</td>' for f in fields)+'</tr>')
    control_table = []
    for stage, item in stages.items():
        arm, step = stage.split('/')
        for c in item['controls']:
            fields = [c['split'], c['policy'], f"{100*c['requested_text_cer']:.2f}%", 'N/A' if c['supplied_text_cer'] is None else f"{100*c['supplied_text_cer']:.2f}%", c['exact_requested']]
            control_table.append(f'<tr><td><a href="controls-{arm}-{step}.html">{stage}</a></td>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in fields)+'</tr>')
    (out/'index.html').write_text(_head('Native conditioning controls')+'<h1>Native text-only conditioning — '+('PARTIAL snapshot' if summary['partial'] else 'completed two-arm study')+'</h1><p>Fresh same-AMD-backend controls: original concat versus joint text/trajectory attention. 8144 TRAIN lines /186 writers; frozen initialized transport, NOT semantic InkVAE. Joint has extra parameters, so not an attention-only causal experiment. No clean prefix, oracle length, target trajectory or writer ID at generation. DEV is exposed research data; no independent confirmation opened. Stroke counts are not content composition. Swapped outputs are scored against BOTH requested and actually supplied texts in the JSON. No quality promotion implied.</p><p><a href="summary.json">Verified metrics, controls, target-informed diagnostics and hashes</a>. Every declared output is in the galleries; report generation does not count as visual review.</p><table><tr><th>Arm/step</th><th>Split</th><th>Guidance</th><th>CER</th><th>Exact</th><th>Rows</th><th>No EOC</th><th>Generated lifts</th><th>IAM lifts</th><th>Severe rows</th></tr>'+''.join(table)+'</table><h2>Matched primary-seed guidance1 text controls</h2><table><tr><th>Arm/step paired gallery</th><th>Split</th><th>Policy</th><th>Requested CER</th><th>Supplied CER</th><th>Exact requested</th></tr>'+''.join(control_table)+'</table>')
    return str(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('relative')
    parser.add_argument('--root', default='data')
    parser.add_argument('--arms', nargs='+', choices=ARMS, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--steps', nargs='+', type=int)
    mode.add_argument('--terminal', action='store_true', help='require result files and all actual per-arm milestones, including differing wall stops')
    args = parser.parse_args()
    print(generate(args.relative, args.arms, args.steps, args.root))
