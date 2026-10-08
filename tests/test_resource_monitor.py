import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch,Mock
from iam_tools.resource_monitor import LinuxSampler, ResourceMonitor, proc_ticks, recommendations


def sample(t, cpu=1., hot=.95, gpu=20., rss=1024):
    r = dict(monotonic=t, timestamp=t, cpu_cores=cpu, hottest_thread_cores=hot, process_tree_rss_bytes=rss)
    if gpu is not None:
        r['gpu'] = dict(busy_percent=gpu, vram_used_mib=100, vram_total_mib=10000)
    return r


class ResourceTests(unittest.TestCase):
    def test_proc_spaces_parentheses(self):
        fields = ['S'] + ['0']*10 + ['123', '456'] + ['0']*20
        self.assertEqual(proc_ticks('45 (a ) b worker) ' + ' '.join(fields)), 579)

    def test_serial_not_extra_cpu(self):
        a = recommendations([sample(1)], 4, 16384)
        self.assertEqual(a[0]['code'], 'serial_cpu_gpu_low')
        self.assertIn('More reserved cores alone', a[0]['recommendation'])

    def test_cpu_scaling_candidate(self):
        self.assertEqual(recommendations([sample(1, cpu=3.9)], 4, 16384)[0]['code'], 'cpu_saturated_gpu_low')

    def test_idle_cpu(self):
        self.assertEqual(recommendations([sample(1, cpu=.1, hot=.1)], 4, 16384)[0]['code'], 'gpu_low_no_cpu_saturation')

    def test_busy_gpu_idle_cores_not_alert(self):
        self.assertEqual(recommendations([sample(1, cpu=1, gpu=95)], 4, 16384), [])

    def test_no_gpu_is_not_zero_gpu(self):
        self.assertEqual(recommendations([sample(1, gpu=None)], 4, 16384), [])

    def test_unverified_cgroup_cannot_drive_alerts(self):
        r = sample(1, cpu=.1, hot=.1)
        r['cgroup'] = {'cpu.stat': 'usage_usec 999999999', 'scope': 'unverified'}
        self.assertEqual(recommendations([r], 4, 16384)[0]['code'], 'gpu_low_no_cpu_saturation')

    def test_memory_pressure(self):
        r = sample(1, gpu=95, rss=16000*1024**2);r['gpu']['vram_used_mib'] = 9500
        self.assertEqual({x['code'] for x in recommendations([r], 4, 16384)}, {'vram_pressure', 'rss_pressure'})

    def test_single_memory_spike_is_not_sustained(self):
        rows=[sample(t,gpu=95) for t in range(10)]
        rows[0]['gpu']['vram_used_mib']=9999
        self.assertEqual(recommendations(rows,4,16384),[])

    def test_gaps_do_not_prove_sustained(self):
        with tempfile.TemporaryDirectory() as folder:
            m=ResourceMonitor(folder,interval=5,sustained_seconds=30)
            m.set_phase('train');m.phase_start=0
            for t in [5,10,15,45,50,55,60]:m.record(sample(t))
            self.assertEqual(m.alerts,[])

    def test_phase_reset_sustained_and_dedup(self):
        with tempfile.TemporaryDirectory() as folder:
            m = ResourceMonitor(folder, interval=5, sustained_seconds=30)
            m.set_phase('eval');m.phase_start=0
            for t in range(5, 45, 5):m.record(sample(t))
            self.assertEqual(m.alerts, [])
            m.set_phase('train');m.phase_start=40
            for t in range(45, 75, 5):m.record(sample(t))
            self.assertEqual(m.alerts, [])
            m.record(sample(75));m.record(sample(80))
            self.assertEqual(len(m.alerts), 1)
            m.set_phase('eval');m.set_phase('train');m.phase_start=80
            m.record(sample(85));self.assertEqual(len(m.alerts),1)
            m.step(.1, 8);m.step(.2, 8)
            result=m.close();self.assertAlmostEqual(result['phases']['train']['examples_per_second'], 16/.3)
            self.assertTrue((Path(folder)/'resource-summary.json').exists())

    def test_repeated_train_phase_announcements_preserve_sustained_alert_window(self):
        with tempfile.TemporaryDirectory() as folder:
            m=ResourceMonitor(folder,interval=5,sustained_seconds=30)
            m.set_phase('train');m.phase_start=0;generation=m.generation
            for t in range(5,46,5):
                m.set_phase('train');m.record(sample(t),generation)
            self.assertEqual(m.generation,generation);self.assertEqual(m.phase_start,0)
            self.assertEqual(len(m.samples),9);self.assertEqual(len(m.alerts),1)
            self.assertEqual(m.alerts[0]['code'],'serial_cpu_gpu_low')

    def test_same_phase_context_is_not_a_transition_but_evaluation_resets_window(self):
        with tempfile.TemporaryDirectory() as folder:
            m=ResourceMonitor(folder,interval=5,sustained_seconds=30)
            m.set_phase('train');m.phase_start=0;generation=m.generation
            for t in range(5,26,5):m.record(sample(t))
            window=list(m.window)
            with m.in_phase('train'):self.assertEqual(m.window,window)
            self.assertEqual(m.generation,generation)
            with m.in_phase('eval'):
                self.assertEqual(m.window,[]);m.record(sample(30))
            self.assertEqual(m.phase,'train');self.assertEqual(m.window,[])
            self.assertGreater(m.generation,generation)
            self.assertEqual(m.alerts,[])

    def test_straddled_phase_sample_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            m=ResourceMonitor(folder);old=m.generation;m.set_phase('train');m.record(sample(5),old)
            self.assertEqual(m.samples, [])

    def test_shutdown_and_failed_sampler_nonfatal(self):
        with tempfile.TemporaryDirectory() as folder:
            def fail():raise OSError('synthetic unavailable')
            m=ResourceMonitor(folder,interval=.01,sustained_seconds=.1,sampler=fail).start()
            time.sleep(.03);m.close();self.assertFalse(m.thread.is_alive())
            self.assertIn('synthetic unavailable',(Path(folder)/'monitor-errors.jsonl').read_text())

    def test_linux_cpu_without_gpu(self):
        sampler=LinuxSampler(gpu=False);sampler();r=sampler()
        self.assertGreaterEqual(r['cpu_cores'],0);self.assertNotIn('gpu',r)
        self.assertGreater(r['process_tree_rss_bytes'],0)

    def test_storage_failure_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as folder:
            m=ResourceMonitor(folder)
            with patch.object(Path,'open',side_effect=OSError('disk unavailable')),patch.object(Path,'write_text',side_effect=OSError('disk unavailable')):
                m.record(sample(10));m.step(.2,8);result=m.close()
            self.assertEqual(result['phases']['startup']['steps'],1)

    def test_ambiguous_host_gpus_rejected(self):
        sampler=LinuxSampler(gpu=True)
        response=Mock(stdout='GPU-a,10,20,30,100,40\nGPU-b,10,20,30,100,40\n')
        with patch('iam_tools.resource_monitor.subprocess.run',return_value=response):r=sampler()
        self.assertNotIn('gpu',r);self.assertIn('ambiguous',r['gpu_error'])

    def test_missing_gpu_is_nonfatal(self):
        sampler=LinuxSampler(gpu=True)
        with patch('iam_tools.resource_monitor.subprocess.run',side_effect=FileNotFoundError('nvidia-smi')):r=sampler()
        self.assertNotIn('gpu',r);self.assertIn('nvidia-smi',r['gpu_error'])

    def test_positive_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):ResourceMonitor(folder,cpu_request=0)

if __name__=='__main__':unittest.main()
