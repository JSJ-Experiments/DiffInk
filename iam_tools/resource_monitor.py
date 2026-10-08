"""Best-effort, phase-aware resource evidence; never an automatic scaler.

CPU cores = process-tree CPU seconds / wall second (not host /proc/stat).
Cgroup scope cannot be verified generically; its counters are contextual only.
GPU utilization is NVIDIA's device busy percentage, NOT VRAM occupancy.
"""
import contextlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


def proc_ticks(text):
    # comm may contain spaces and ')'; fields after final ')' start at field3.
    fields = text[text.rfind(')') + 2:].split()
    return int(fields[11]) + int(fields[12])


def read_number(path):
    value = Path(path).read_text().strip()
    return None if value == 'max' else int(value)


class LinuxSampler:
    def __init__(self, gpu=True, gpu_uuid=None):
        self.pid = os.getpid()
        self.hz = os.sysconf('SC_CLK_TCK')
        self.gpu = gpu
        self.gpu_uuid = gpu_uuid
        self.previous = None
        self.cgroup = self._cgroup()

    def _cgroup(self):
        try:
            line = next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
            relative = line[3:].lstrip('/')
            path = Path('/sys/fs/cgroup') / relative
            if not (path / 'cpu.stat').exists():
                path = Path('/sys/fs/cgroup')
            return dict(path=str(path), reported_membership=line[3:], scope='unverified; contextual counters only')
        except Exception as e:
            return dict(scope='unavailable', error=type(e).__name__ + ': ' + str(e))

    def _tree(self):
        pending = [self.pid]; seen = set(); ticks = {}; rss = 0; errors = []
        while pending:
            pid = pending.pop()
            if pid in seen:
                continue
            seen.add(pid)
            try:
                for task in (Path('/proc') / str(pid) / 'task').iterdir():
                    ticks[f'{pid}/{task.name}'] = proc_ticks((task / 'stat').read_text())
                    pending.extend(map(int, (task / 'children').read_text().split()))
                for line in (Path('/proc') / str(pid) / 'status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        rss += int(line.split()[1]) * 1024
            except (OSError, ValueError) as e:
                errors.append(str(e))
        # Sum RSS can double count shared pages; not actual container memory.
        return ticks, rss, len(seen), errors

    def __call__(self):
        now = time.monotonic(); ticks, rss, processes, errors = self._tree()
        result = dict(timestamp=time.time(), monotonic=now, process_tree_rss_bytes=rss,
                      rss_definition='sum process RSS; shared pages may be counted more than once',
                      process_count=processes, cpu_cores=None, hottest_thread_cores=None, errors=errors)
        if self.previous:
            old_time, old_ticks = self.previous
            elapsed = now - old_time
            rates = [max(0, v - old_ticks[k]) / self.hz / elapsed for k, v in ticks.items() if k in old_ticks]
            result.update(cpu_cores=sum(rates), hottest_thread_cores=max(rates, default=0.))
        self.previous = now, ticks
        cg = dict(self.cgroup)
        if 'path' in cg:
            path = Path(cg['path'])
            for name in ('cpu.stat', 'cpu.max', 'memory.current', 'memory.max'):
                try:
                    cg[name] = (path / name).read_text().strip()
                except OSError as e:
                    cg[name + '_error'] = str(e)
        result['cgroup'] = cg
        if self.gpu:
            command = ['nvidia-smi', '--query-gpu=uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw', '--format=csv,noheader,nounits']
            if self.gpu_uuid:
                command += ['-i', self.gpu_uuid]
            try:
                rows = subprocess.run(command, capture_output=True, text=True, timeout=2, check=True).stdout.strip().splitlines()
                if len(rows) != 1:
                    raise ValueError('ambiguous visible GPU devices; provide GPU UUID, refusing host-wide attribution')
                fields = [x.strip() for x in rows[0].split(',')]
                def number(x):
                    try:
                        v = float(x)
                        return v if math.isfinite(v) else None
                    except ValueError:
                        return None
                result['gpu'] = dict(zip(('uuid', 'busy_percent', 'memory_activity_percent', 'vram_used_mib', 'vram_total_mib', 'power_watts'), [fields[0]] + [number(x) for x in fields[1:]]))
            except Exception as e:
                result['gpu_error'] = str(e)
        return result


def recommendations(samples, cpu_request, memory_request_mib):
    """Called only on a sustained homogeneous training window by the monitor."""
    def avg(key):
        v = [s[key] for s in samples if s.get(key) is not None]
        return statistics.mean(v) if v else None
    cpu = avg('cpu_cores'); hot = avg('hottest_thread_cores'); alerts = []
    gpu_rows = [s['gpu'] for s in samples if s.get('gpu', {}).get('busy_percent') is not None]
    busy = statistics.mean(s['busy_percent'] for s in gpu_rows) if gpu_rows else None
    if busy is not None and busy < 50 and cpu is not None:
        if cpu >= .85 * cpu_request:
            alerts.append(dict(code='cpu_saturated_gpu_low', recommendation='CPU process tree near requested cores while GPU is underused. Profile input/dispatch; benchmark more CPU or vectorization before scaling. CPU request is a reservation, not a hard limit.'))
        elif hot is not None and hot >= .8 and cpu < max(1.8, .5 * cpu_request):
            alerts.append(dict(code='serial_cpu_gpu_low', recommendation='Hot serial thread plus low GPU activity. More reserved cores alone likely will not help: reduce CPU synchronizations, vectorize/cache input, profile dispatch, benchmark larger batches.'))
        else:
            alerts.append(dict(code='gpu_low_no_cpu_saturation', recommendation='GPU activity low without measured CPU saturation. Inspect I/O, synchronization, tiny kernels and batch size; do not infer a need for more CPUs.'))
    memory_rows = [s['gpu'] for s in samples if s.get('gpu', {}).get('vram_total_mib') and s['gpu'].get('vram_used_mib') is not None]
    ratios = [g['vram_used_mib'] / g['vram_total_mib'] for g in memory_rows]
    if ratios and statistics.mean(ratios) >= .9:
        alerts.append(dict(code='vram_pressure', recommendation='Mean GPU memory above 90%; inspect peak allocator usage, smaller physical batches/accumulation or a higher-memory GPU.'))
    rss = avg('process_tree_rss_bytes')
    if rss is not None and rss > .9 * memory_request_mib * 1024**2:
        alerts.append(dict(code='rss_pressure', recommendation='Summed process RSS near memory request (may double-count shared pages). Verify container memory scope before increasing memory reservation.'))
    return alerts


class ResourceMonitor:
    def __init__(self, folder, *, cpu_request=4, memory_request_mib=16384, gpu=True,
                 gpu_uuid=None, interval=5., sustained_seconds=30., sampler=None):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        self.cpu_request = cpu_request; self.memory_request_mib = memory_request_mib
        self.interval = interval; self.sustained = sustained_seconds
        if min(cpu_request, memory_request_mib, interval, sustained_seconds) <= 0:
            raise ValueError('positive resource requests and time windows required')
        self.sampler = sampler or LinuxSampler(gpu, gpu_uuid)
        self.lock = threading.RLock(); self.stop_event = threading.Event()
        self.phase = 'startup'; self.generation = 0; self.phase_start = time.monotonic()
        self.samples = []; self.window = []; self.alerts = []; self.alert_keys = set(); self.steps = {}
        self.thread = None

    def set_phase(self, name):
        with self.lock:
            name = str(name)
            # Training loops may announce the same phase every update. This is
            # NOT a transition: resetting would permanently suppress sustained
            # alerts and discard samples from the still-homogeneous window.
            if name == self.phase:
                return
            self.phase = name; self.generation += 1; self.phase_start = time.monotonic(); self.window = []

    @contextlib.contextmanager
    def in_phase(self, name):
        previous = self.phase; self.set_phase(name)
        try:
            yield
        finally:
            self.set_phase(previous)

    def step(self, seconds, examples=1):
        with self.lock:
            row = self.steps.setdefault(self.phase, dict(seconds=[], examples=0))
            row['seconds'].append(float(seconds)); row['examples'] += examples

    def _append(self, name, value):
        try:
            with (self.folder / name).open('a') as f:
                f.write(json.dumps(value) + '\n')
        except OSError as e:
            print('[resource telemetry storage unavailable] ' + str(e), flush=True)

    def record(self, row, generation=None):
        emitted = []
        with self.lock:
            if generation is not None and generation != self.generation:
                return  # sample straddled evaluation/train transition
            row = dict(row, phase=self.phase, phase_generation=self.generation, phase_age_seconds=row['monotonic'] - self.phase_start)
            self.samples.append(row)
            if self.phase.startswith('train') and row['monotonic'] - self.phase_start >= self.interval:
                self.window.append(row)
                cutoff = row['monotonic'] - self.sustained
                while len(self.window) > 1 and self.window[1]['monotonic'] <= cutoff:
                    self.window.pop(0)
                enough = len(self.window) >= max(2, math.ceil(self.sustained / (2 * self.interval)))
                duration = self.window[-1]['monotonic'] - self.window[0]['monotonic']
                gaps = any(b['monotonic'] - a['monotonic'] > 2.5 * self.interval for a, b in zip(self.window, self.window[1:]))
                if enough and duration >= self.sustained and not gaps:
                    for alert in recommendations(self.window, self.cpu_request, self.memory_request_mib):
                        key = (self.generation, alert['code'])
                        if key in self.alert_keys:
                            continue
                        self.alert_keys.add(key)
                        alert.update(timestamp=row.get('timestamp'), phase=self.phase, window_seconds=self.sustained)
                        self.alerts.append(alert); emitted.append(alert)
        # Disk/network writes must not hold the training step's timing lock.
        self._append('resources.jsonl', row)
        for alert in emitted:
            self._append('alerts.jsonl', alert)
            print('[resource alert] ' + json.dumps(alert), flush=True)

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    generation = self.generation
                self.record(self.sampler(), generation)
            except Exception as e:
                # Observability failure must not kill a GPU experiment.
                print('[resource monitor unavailable] ' + str(e), flush=True)
                try:
                    with (self.folder / 'monitor-errors.jsonl').open('a') as f:
                        f.write(json.dumps(dict(timestamp=time.time(), error=str(e))) + '\n')
                except OSError:
                    pass  # telemetry storage failure also must not affect training
            self.stop_event.wait(self.interval)

    def start(self):
        if self.thread is not None:
            raise RuntimeError('already started')
        self.thread = threading.Thread(target=self._loop, daemon=True, name='resource-monitor'); self.thread.start()
        return self

    def close(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=self.interval + 4)
        with self.lock:
            phases = {}
            for phase in sorted({s['phase'] for s in self.samples} | set(self.steps)):
                rows = [s for s in self.samples if s['phase'] == phase]
                d = dict(samples=len(rows))
                for key in ('cpu_cores', 'hottest_thread_cores', 'process_tree_rss_bytes'):
                    values = [s[key] for s in rows if s.get(key) is not None]
                    d[key + '_mean'] = statistics.mean(values) if values else None
                gs = [s['gpu']['busy_percent'] for s in rows if s.get('gpu', {}).get('busy_percent') is not None]
                d['gpu_busy_percent_mean'] = statistics.mean(gs) if gs else None
                steps = self.steps.get(phase, dict(seconds=[], examples=0)); values = sorted(steps['seconds'])
                if values:
                    d.update(steps=len(values), examples=steps['examples'], measured_step_seconds=sum(values),
                             examples_per_second=steps['examples']/max(sum(values), 1e-9),
                             latency_p50_seconds=statistics.median(values), latency_p90_seconds=values[int(.9*(len(values)-1))])
                phases[phase] = d
            result = dict(cpu_request=self.cpu_request, memory_request_mib=self.memory_request_mib,
                          cpu_definition='process tree CPU-seconds / wall-second; new/exited threads may be missed',
                          cgroup_scope='unverified contextual only; never drives core scaling alerts',
                          gpu_definition='NVIDIA device busy percent; memory activity is not VRAM occupancy',
                          interval_seconds=self.interval, sustained_seconds=self.sustained,
                          phases=phases, alerts=self.alerts, automatic_scaling=False)
        try:
            (self.folder / 'resource-summary.json').write_text(json.dumps(result, indent=2) + '\n')
        except OSError as e:
            print('[resource summary storage unavailable] ' + str(e), flush=True)
        return result

    def __enter__(self):
        return self.start()

    def __exit__(self, *args):
        self.close()
