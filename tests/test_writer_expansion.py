import copy
import unittest
from iam_tools.writer_expansion import partition, train_score, training_schedule, run


class WriterExpansionTests(unittest.TestCase):
    def test_disjoint_partition_retains_old_order(self):
        self.assertEqual(partition(['a','b','c'],['v'],['c','a']),
                         dict(old=['c','a'],new=['b'],train=['a','b','c'],held_out=['v']))
    def test_reject_leakage_duplicates_missing_old_and_empty(self):
        for args in [(['a'],['a'],['a']),(['a','a'],['v'],['a']),(['a'],['v'],['b']),(['a'],[],['a'])]:
            with self.assertRaises(ValueError):partition(*args)
    def test_checkpoint_selection_independent_of_validation(self):
        g=dict(mean_per_line_x_rmse=.01,mean_per_line_y_rmse=.02,mean_per_line_first_difference_relative=.1)
        row=dict(groups=dict(train=dict(mu=g,sampled=g,mu_macro_pen_f1=.9),held_out=dict(arbitrary='bad')))
        initial=train_score(row); changed=copy.deepcopy(row);changed['groups']['held_out']={'arbitrary':'excellent'}
        self.assertEqual(initial,train_score(changed)); self.assertAlmostEqual(initial,.045)
    def test_schedule_complete_epochs_and_reproducible(self):
        ids=['a','b','c']; a=list(training_schedule(ids,4,accumulation=2))
        self.assertEqual(a,list(training_schedule(ids,4,accumulation=2)))
        flat=[x for update in a for x in update]
        self.assertEqual(set(flat),set(ids));self.assertEqual(sorted(flat[:3]),ids)
        self.assertEqual(sorted(flat[3:6]),ids);self.assertTrue(all(len(u)==2 for u in a))
        with self.assertRaises(ValueError):list(training_schedule(['a','a'],1))

    def test_bounds_checked_before_gpu_or_data(self):
        with self.assertRaises(ValueError):run('missing','missing',steps=2001)
        with self.assertRaises(ValueError):run('missing','missing',lr=.001)




class PenRefitIsolationTests(unittest.TestCase):
    def test_mean_and_sampled_cache_change_only_pen_rows(self):
        import sys,tempfile
        from pathlib import Path
        import torch
        from iam_tools.writer_polish import refit_pen
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'third_party/DiffInk'))
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.encoder=torch.nn.Conv1d(5,4,8,stride=8)
                self.conv_mu=torch.nn.Conv1d(4,4,1);self.conv_logvar=torch.nn.Conv1d(4,4,1)
                self.transformer_decoder=torch.nn.Module();self.transformer_decoder.fc=torch.nn.Linear(4,123)
            def to_model_space(self,x):return x
            def decode(self,z,padding_mask=None):
                features=z.repeat_interleave(8,dim=2).transpose(1,2)
                return self.transformer_decoder.fc(features).transpose(1,2)
        torch.manual_seed(17);model=Tiny();before={k:v.clone() for k,v in model.state_dict().items()}
        raw=torch.zeros(1,5,16);raw[:,0]=torch.linspace(0,1,16);raw[:,2]=1;raw[:,2,7]=0;raw[:,3,7]=1;raw[:,2,-1]=0;raw[:,4,-1]=1
        batch=(raw,torch.ones(1,16,dtype=torch.bool),torch.tensor([[0,1]]))
        with tempfile.TemporaryDirectory() as tmp:result=refit_pen(model,[batch],Path(tmp)/'refit',updates=5)
        self.assertTrue(result['non_pen_parameters_bitwise_unchanged'])
        self.assertEqual(result['training_lines'],1)
        self.assertFalse(torch.equal(before['transformer_decoder.fc.weight'][:3],model.transformer_decoder.fc.weight[:3]))
        for k,v in model.state_dict().items():
            if k.startswith('transformer_decoder.fc.'):self.assertTrue(torch.equal(before[k][3:],v[3:]))
            else:self.assertTrue(torch.equal(before[k],v))



class CheckpointAuditTests(unittest.TestCase):
    def test_audit_uses_original_input_not_selected_output(self):
        import json,tempfile,torch
        from pathlib import Path
        from iam_tools.verify_writer_checkpoint import parent_from_provenance
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir()
            torch.save(dict(value=torch.tensor(1)),root/'original.pt')
            torch.save(dict(value=torch.tensor(2)),run/'checkpoint-best.pt')
            (run/'provenance.json').write_text(json.dumps(dict(source_rel='original.pt')))
            parent,provenance=parent_from_provenance(run,root)
            self.assertEqual(parent['value'].item(),1)
            (run/'provenance.json').write_text(json.dumps(dict(source_rel='run/checkpoint-best.pt')))
            with self.assertRaisesRegex(ValueError,'itself'):parent_from_provenance(run,root)




class BalancedObjectiveTests(unittest.TestCase):
    def test_pen_row_lr_matches_separate_adam_groups(self):
        import torch
        from iam_tools.writer_expansion import adam_step_with_pen_lr
        torch.manual_seed(7)
        whole=torch.nn.Linear(4,6);head=torch.nn.Linear(4,3);rest=torch.nn.Linear(4,3)
        with torch.no_grad():
            head.weight.copy_(whole.weight[:3]);head.bias.copy_(whole.bias[:3])
            rest.weight.copy_(whole.weight[3:]);rest.bias.copy_(whole.bias[3:])
        a=torch.optim.AdamW(whole.parameters(),lr=1e-4,betas=(.9,.99),weight_decay=0)
        b=torch.optim.AdamW([dict(params=head.parameters(),lr=1e-3),dict(params=rest.parameters(),lr=1e-4)],betas=(.9,.99),weight_decay=0)
        x=torch.randn(8,4);target=torch.randn(8,6)
        for _ in range(20):
            a.zero_grad();b.zero_grad()
            (whole(x)-target).square().mean().backward()
            (torch.cat([head(x),rest(x)],1)-target).square().mean().backward()
            adam_step_with_pen_lr(a,whole,10);b.step()
        torch.testing.assert_close(whole.weight[:3],head.weight,atol=5e-7,rtol=2e-6)
        torch.testing.assert_close(whole.bias[:3],head.bias,atol=5e-7,rtol=2e-6)
        torch.testing.assert_close(whole.weight[3:],rest.weight,atol=1e-7,rtol=1e-6)
        with self.assertRaises(ValueError):adam_step_with_pen_lr(a,whole,101)

    def test_linear_probe_recovers_known_map_and_rejects_bad_shapes(self):
        import numpy as np
        from iam_tools.xy_readout_probe import fit
        t=np.linspace(-1,1,40);features=np.column_stack([t,t*t,np.sin(t)])
        target=np.column_stack([2*t+.3*t*t+1,-t+.4*np.sin(t)])
        states=np.zeros(40,dtype=int);states[19]=1;states[-1]=2
        coef,meta=fit([features],[target],[states])
        np.testing.assert_allclose(np.column_stack([features,np.ones(40)])@coef,target,atol=1e-10)
        self.assertEqual(meta['rank'],4)
        with self.assertRaises(ValueError):fit([features],[target[:-1]],[states])
        with self.assertRaises(ValueError):fit([],[],[])




class ResumeTests(unittest.TestCase):
    def test_resume_preserves_original_budget_and_rejects_mismatch(self):
        from iam_tools.writer_expansion import validate_resume
        parent=dict(config=dict(profile='seen-writer-24-line-balanced-joint',base_lr=5e-6,max_optimizer_updates=1200),
                    optimizer_updates=200,optimizer_state_dict={},rng_state_cpu=[],rng_state_cuda=[],provenance={})
        validate_resume(parent,True,5e-6,1000)
        for balanced,lr,steps in [(False,5e-6,1000),(True,1e-5,1000),(True,5e-6,1200)]:
            with self.assertRaises(ValueError):validate_resume(parent,balanced,lr,steps)
        del parent['rng_state_cuda']
        with self.assertRaises(ValueError):validate_resume(parent,True,5e-6,1000)

    def test_reconstructed_schedule_matches_uninterrupted_suffix(self):
        ids=['a','b','c','d','e'];whole=list(training_schedule(ids,12,accumulation=8))
        resumed=[u for step,u in enumerate(training_schedule(ids,12,accumulation=8),1) if step>4]
        self.assertEqual(resumed,whole[4:])




class PhaseProbeTests(unittest.TestCase):
    def test_zero_centered_phase_fit_preserves_global_offset(self):
        import numpy as np
        from iam_tools.phase_probe import fit_phase_bias
        b=np.column_stack([np.arange(8)-3.5,np.linspace(-1,1,8)])*.01;g=np.array([.2,-.1])
        row=dict(mu=dict(geometry=dict(point_index_mod8=[dict(phase=i,count=2,mean_residual=(b[i]+g).tolist()) for i in range(8)])))
        fitted,global_bias=fit_phase_bias([row]);np.testing.assert_allclose(fitted,b,atol=1e-15)
        np.testing.assert_allclose(global_bias,g,atol=1e-15)
        with self.assertRaises(ValueError):fit_phase_bias([])




class ResumeAuditTests(unittest.TestCase):
    def test_nested_optimizer_and_rng_comparison_detects_changed_tensor(self):
        import torch
        from iam_tools.verify_writer_checkpoint import equal_nested
        x=dict(state={0:dict(step=torch.tensor(2.),exp_avg=torch.tensor([.1,.2]))},groups=[dict(lr=5e-6)])
        y=copy.deepcopy(x);self.assertTrue(equal_nested(x,y));y['state'][0]['exp_avg'][0]=.3
        self.assertFalse(equal_nested(x,y));self.assertFalse(equal_nested([1],(1,)))


if __name__=='__main__':unittest.main()
