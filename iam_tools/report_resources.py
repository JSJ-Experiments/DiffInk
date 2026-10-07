"""CPU-side report of measured utilization and bottleneck recommendations."""
import html,json
from pathlib import Path


def report(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=Path(directory);summary=json.loads((d/'resource-summary.json').read_text());rows=[json.loads(s) for s in (d/'resources.jsonl').read_text().splitlines()];out=d/'report';out.mkdir(exist_ok=True)
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    phases=[p for p in summary['phases'] if p.startswith('train')]
    fig,axs=plt.subplots(3,1,figsize=(15,10),sharex=True)
    first=min(r['monotonic'] for r in rows)
    for phase in phases:
        r=[r for r in rows if r['phase']==phase];x=[v['monotonic']-first for v in r]
        axs[0].plot(x,[v.get('gpu',{}).get('busy_percent') for v in r],'.-',label=phase)
        axs[1].plot(x,[v.get('cpu_cores') for v in r],'.-')
        axs[2].plot(x,[v.get('gpu',{}).get('vram_used_mib') for v in r],'.-')
    axs[0].set_ylabel('NVIDIA GPU busy %');axs[0].set_ylim(0,100);axs[0].legend(fontsize=8)
    axs[1].set_ylabel('Process-tree CPU cores');axs[2].set_ylabel('Device VRAM MiB');axs[2].set_xlabel('elapsed sample seconds (preparation/eval not connected)')
    for ax in axs:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'resources.png',dpi=130);plt.close(fig)
    body='<!doctype html><meta charset="utf-8"><title>Measured resource utilization</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:20px}img{width:100%}pre{white-space:pre-wrap}td,th{padding:8px;border:1px solid #ccc}table{border-collapse:collapse}</style><h1>Resource evidence, not “every core must be busy”</h1><p>Phase-tagged process CPU and NVIDIA GPU activity. No autoscaling. Cgroup scope unverified; missing metrics stay missing. VRAM is not GPU activity. Read raw evidence before changing resource reservations.</p><p><a href="../config.json">Configuration</a> · <a href="../resource-summary.json">Summary/recommendations</a> · <a href="../resources.jsonl">Raw samples</a></p><img src="resources.png"><table><tr><th>Phase</th><th>Examples/s (step wall)</th><th>GPU busy</th><th>CPU cores</th><th>Hottest thread</th><th>p50 / p90 update ms</th></tr>'
    def fmt(v):return 'unavailable' if v is None else f'{v:.2f}'
    for p in phases:
        s=summary['phases'][p];lat=[s.get('latency_p50_seconds'),s.get('latency_p90_seconds')]
        body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [p,fmt(s.get('examples_per_second')),fmt(s.get('gpu_busy_percent_mean')),fmt(s.get('cpu_cores_mean')),fmt(s.get('hottest_thread_cores_mean')),' / '.join(fmt(x*1000 if x is not None else None) for x in lat)])+'</tr>'
    body+='</table><h2>Sustained alerts</h2><pre>'+html.escape(json.dumps(summary['alerts'],indent=2))+'</pre>'
    (out/'index.html').write_text(body);return str(out/'index.html')
