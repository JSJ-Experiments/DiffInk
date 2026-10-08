"""Guard launcher shape without submitting jobs or needing cloud credentials."""
import ast
from pathlib import Path
import unittest


class PrefixLaunchTests(unittest.TestCase):
    def fixture(self):
        source=Path(__file__).resolve().parent.parent/'modal_generation_prefix_contract.py'
        return {node.name:node for node in ast.parse(source.read_text()).body if isinstance(node,ast.FunctionDef)}

    def test_local_entrypoint_has_one_durable_remote_parent(self):
        nodes=self.fixture();main=nodes['main']
        calls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['remote','spawn']]
        self.assertEqual(len(calls),1)
        self.assertEqual(ast.unparse(calls[0].func),'coordinate.spawn')
        self.assertIn('call.get()',ast.unparse(main))
        self.assertIsInstance(main.body[0],ast.If)
        self.assertEqual(ast.unparse(main.body[0].test),'not train')
        self.assertTrue(any(isinstance(n,ast.Return) for n in main.body[0].body))

    def test_remote_parent_retains_both_children(self):
        nodes=self.fixture();code=ast.unparse(nodes['coordinate'])
        self.assertIn('prepare.remote()',code);self.assertIn('research.spawn(a, relative)',code)
        self.assertIn("['noncausal', 'causal']",code);self.assertIn('call.get()',code)
        self.assertIn('volume.reload()',code);self.assertIn('volume.commit()',code)
