import ast,copy,unittest
from pathlib import Path
from iam_tools.generation_weak_confirmation import fit_gate
from iam_tools.generation_weak_continuation import checked_path

class WeakContinuationTests(unittest.TestCase):
    def test_fit_gate_explicit_global_step_and_bounded_extension(self):
        cfg=dict(max_updates=60000,continuation_of='checkpoints/iam_generation_weak_alignment/run',continuation_updates=12000,parent_step=48000)
        def result(cer=.04):return dict(stop='budget_completed',last_step=60000,best_step=60000,history=[dict(step=60000,aggregate=dict(all_train256=dict(correct=dict(free_cer=cer))))])
        results={a:result() for a in ['control','weak_alignment']};fit_gate(cfg,results)
        for bad in [dict(cfg,continuation_updates=24000),dict(cfg,parent_step=0),dict(cfg,max_updates=48000)]:
            with self.assertRaises(ValueError):fit_gate(bad,results)
        bad=dict(results,weak_alignment=result(.051))
        with self.assertRaises(ValueError):fit_gate(cfg,bad)

    def test_continuation_paths_not_parent_or_traversal(self):
        self.assertEqual(str(checked_path('checkpoints/iam_generation_weak_continuation/stamp','data')),'data/checkpoints/iam_generation_weak_continuation/stamp')
        for rel in ['checkpoints/iam_generation_weak_alignment/stamp','checkpoints/iam_generation_weak_continuation/../oops','/tmp/anything']:
            with self.assertRaises(ValueError):checked_path(rel,'data')

    def test_actual_engine_restores_full_own_state_no_new_calibration_or_losses(self):
        p=Path(__file__).resolve().parent.parent/'iam_tools/generation_weak_continuation.py'
        nodes={n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)};run=ast.unparse(nodes['run']);prep=ast.unparse(nodes['prepare'])
        self.assertIn("opt.load_state_dict(saved['optimizer_state_dict'])",run)
        self.assertIn("model.load_state_dict(saved['model_state_dict'])",run)
        self.assertIn("torch.set_rng_state(saved['torch_rng_state'])",run);self.assertIn("torch.cuda.set_rng_state(saved['cuda_rng_state'])",run)
        self.assertIn("alignment_weight = saved['alignment_weight']",run);self.assertIn('step = 48000 + update',run);self.assertNotIn('calibrate(',run)
        self.assertIn("(p / 'confirmation').exists()",prep);self.assertIn('> 0.05',prep)
        self.assertIn("['reserved-confirmation', 'reserved-confirmation-corrected']",prep)

    def test_guarded_bounded_durable_launcher(self):
        p=Path(__file__).resolve().parent.parent/'modal_generation_weak_continuation.py';nodes={n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
        self.assertIn('coordinate.spawn()',ast.unparse(nodes['main']));self.assertEqual(ast.unparse(nodes['main'].body[0].test),'not train')
        self.assertIn("['control', 'weak_alignment']",ast.unparse(nodes['coordinate']))
        kwargs={k.arg:ast.literal_eval(k.value) for k in nodes['research'].decorator_list[0].keywords if k.arg in ['gpu','timeout','cpu','retries']}
        self.assertEqual(kwargs,dict(gpu='T4',timeout=3000,cpu=2,retries=0))
