import json,tempfile,unittest
from pathlib import Path
from iam_tools.weak_confirmation_alignment import require_opened

class WeakConfirmationTimingTests(unittest.TestCase):
    def test_no_blind_source_opening_before_one_shot_completion(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):require_opened(d)
            p=Path(d)/'confirmation/summary.json';p.parent.mkdir();p.write_text(json.dumps(dict(arms={'control':{}},preflight=[{}]*16)))
            with self.assertRaises(ValueError):require_opened(d)
            s=dict(arms={'control':{},'weak_alignment':{}},preflight=[{}]*16);p.write_text(json.dumps(s));self.assertEqual(require_opened(d),s)
