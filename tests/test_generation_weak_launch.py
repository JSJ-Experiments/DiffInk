import ast,unittest
from pathlib import Path

class WeakLaunchTests(unittest.TestCase):
    def fixture(self):
        p=Path(__file__).resolve().parent.parent/'modal_generation_weak_alignment.py'
        return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}

    def test_durable_single_parent_and_explicit_training_guard(self):
        main=self.fixture()['main'];self.assertEqual(ast.unparse(main.body[0].test),'not train')
        self.assertTrue(any(isinstance(n,ast.Return) for n in main.body[0].body))
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['remote','spawn']]
        self.assertEqual([ast.unparse(n.func) for n in calls],['coordinate.spawn']);self.assertIn('call.get()',ast.unparse(main))

    def test_two_children_owned_by_remote_parent(self):
        parent=ast.unparse(self.fixture()['coordinate'])
        self.assertIn("['control', 'weak_alignment']",parent);self.assertIn('research.spawn(a, relative)',parent);self.assertIn('call.get()',parent)
        self.assertIn('volume.reload()',parent);self.assertIn('volume.commit()',parent)

    def test_bounded_t4_and_same_engine(self):
        research=self.fixture()['research'];code=ast.unparse(research)
        self.assertIn('generation_weak_alignment_study import run',code);self.assertIn('finally:',code);self.assertIn('volume.commit()',code)
        options={k.arg:ast.literal_eval(k.value) for k in research.decorator_list[0].keywords if k.arg in ['gpu','cpu','memory','timeout','retries','max_containers']}
        self.assertEqual(options,dict(gpu='T4',cpu=2,memory=8192,timeout=7200,retries=0,max_containers=2))
