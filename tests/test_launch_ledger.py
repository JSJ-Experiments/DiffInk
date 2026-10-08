import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from iam_tools.launch_ledger import calls_once

class LaunchLedgerTests(unittest.TestCase):
    def test_reentry_attaches_exact_calls_without_respawn(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spawned=[];attached=[];commits=[]
            def spawn(arm):
                # Intent must already be DURABLY committed before submit.
                self.assertEqual(commits[-1]['calls'][arm]['state'],'spawning')
                spawned.append(arm);return SimpleNamespace(object_id='fc-'+arm)
            def commit():commits.append(json.loads((p/'launch-ledger.json').read_text()))
            calls=calls_once(p,['a','b'],spawn,lambda c:attached.append(c) or c,commit)
            self.assertEqual(spawned,['a','b']);self.assertEqual([c.object_id for c in calls],['fc-a','fc-b'])
            resumed=calls_once(p,['a','b'],spawn,lambda c:attached.append(c) or c,commit)
            self.assertEqual(spawned,['a','b']);self.assertEqual(resumed,['fc-a','fc-b'])
    def test_crash_after_spawn_is_ambiguous_not_permission_to_resubmit(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spawned=[]
            def fail(arm):spawned.append(arm);raise RuntimeError('lost submit response')
            with self.assertRaises(RuntimeError):calls_once(p,['a'],fail,lambda c:c,lambda:None)
            with self.assertRaisesRegex(RuntimeError,'Ambiguous'):calls_once(p,['a'],fail,lambda c:c,lambda:None)
            self.assertEqual(spawned,['a'])
    def test_partial_committed_ledger_resumes_only_missing_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);spawned=[]
            (p/'launch-ledger.json').write_text(json.dumps(dict(schema=1,study_name=p.name,arms=['a','b'],calls={'a':dict(state='spawned',call_id='fc-a')})))
            calls=calls_once(p,['a','b'],lambda a:spawned.append(a) or SimpleNamespace(object_id='fc-'+a),lambda c:SimpleNamespace(object_id=c),lambda:None)
            self.assertEqual(spawned,['b']);self.assertEqual([c.object_id for c in calls],['fc-a','fc-b'])
    def test_untracked_artifacts_and_scope_changes_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'a').mkdir()
            with self.assertRaisesRegex(RuntimeError,'without launch identity'):calls_once(p,['a'],lambda a:None,lambda c:None,lambda:None)
            (p/'a').rmdir();calls_once(p,['a'],lambda a:SimpleNamespace(object_id='fc-a'),lambda c:c,lambda:None)
            with self.assertRaises(ValueError):calls_once(p,['a','b'],lambda a:None,lambda c:None,lambda:None)
    def test_invalid_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            for arms in [[],['a','a'],['../bad'],['.']]:
                with self.assertRaises(ValueError):calls_once(tmp,arms,lambda a:None,lambda c:None,lambda:None)


class LightweightPathTests(unittest.TestCase):
    def test_bounded_relative_study_guard(self):
        from iam_tools.launch_ledger import study_path
        self.assertEqual(str(study_path('/data','checkpoints/study/x','checkpoints/study/')),'/data/checkpoints/study/x')
        for relative in ['/tmp/x','checkpoints/study/../other','checkpoints/other/x','checkpoints/study2/x']:
            with self.assertRaises(ValueError):study_path('/data',relative,'checkpoints/study/')
    def test_module_import_does_not_pull_tensor_libraries(self):
        import subprocess,sys
        probe="import iam_tools.launch_ledger,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
        subprocess.run([sys.executable,'-c',probe],check=True)

if __name__=='__main__':unittest.main()
