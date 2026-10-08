import copy,tempfile,unittest,json
from unittest.mock import patch
from pathlib import Path
from iam_tools.continuous_recovery import validate_checkpoint,copy_snapshot,snapshot_info
from iam_tools.pen_ab import file_sha
from iam_tools.writer_expansion import training_schedule
class ContinuousRecoveryTests(unittest.TestCase):
    def test_resume_keeps_original_training_contract_and_requires_optimizer_rng(self):
        cfg=dict(eval_steps=[0,250,500,1000],max_updates=1000,lr=1e-5,source_archive_sha256='new',recovery=dict(arms={}))
        saved=dict(arm='continuous_xy',step=250,parent_body_step=4000,parent_head_updates=6000,config=dict(cfg,source_archive_sha256='old'),model_state_dict={},optimizer_state_dict=dict(state={1:{'step':250}}),rng_cpu='cpu',rng_cuda=['gpu'])
        validate_checkpoint(saved,cfg,'continuous_xy',250)
        for key,value in [('arm','detached_xy'),('step',500),('parent_body_step',3000),('optimizer_state_dict',dict(state={}))]:
            bad=copy.deepcopy(saved);bad[key]=value
            with self.assertRaises(ValueError):validate_checkpoint(bad,cfg,'continuous_xy',250)
        bad=copy.deepcopy(saved);bad['config']['lr']=1e-4
        with self.assertRaises(ValueError):validate_checkpoint(bad,cfg,'continuous_xy',250)
        for key in ['rng_cpu','rng_cuda']:
            bad=copy.deepcopy(saved);del bad[key]
            with self.assertRaises(ValueError):validate_checkpoint(bad,cfg,'continuous_xy',250)
        with self.assertRaises(ValueError):validate_checkpoint(dict(saved,step=1000),cfg,'continuous_xy',1000)
    def test_recovery_prefix_order_hashes_and_prior_selection_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);arm='continuous_xy';ids=list('abcdefgh');cfg=dict(eval_steps=[0,2,4],max_updates=4,batch=8,schedule_seed=64142,max_train_wall_seconds=1800,recovery=dict(arms={arm:dict(step=2)}));data=dict(training_ids=ids)
            for name in ('checkpoint-last.pt','checkpoint-best.pt'):
                (p/name).write_bytes(name.encode());cfg['recovery']['arms'][arm]['last_sha256' if 'last' in name else 'best_sha256']=file_sha(p/name)
            schedule=list(training_schedule(ids,4,8,64142));rows=[dict(step=i+1,sample_ids=batch,clipped=False) for i,batch in enumerate(schedule[:2])]
            (p/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));gate=dict(passed=True,failures=[])
            for step,cer in [(0,.2),(2,.3)]:
                (p/f'evaluation-{step}.h5').write_bytes(b'actualpackedbytes');ev=dict(teacher=dict(offset_mse=.4),free=dict(correct=dict(cer=cer)),source_rollout={},capacity_gate=gate,packed_h5_sha256=file_sha(p/f'evaluation-{step}.h5'));(p/f'eval-{step}.json').write_text(json.dumps(ev));(p/f'teacher-reading-{step}.json').write_text(json.dumps(dict(capacity_gate=gate)))
            (p/'resource-summary.json').write_text(json.dumps(dict(phases={('train/'+arm):dict(measured_step_seconds=10.)})))
            with patch('iam_tools.continuous_recovery.teacher_gate',return_value=gate):
                info=snapshot_info(p,cfg,data,arm,{})
                self.assertEqual((info['step'],info['best_step'],info['seconds']),(2,0,10.))
                self.assertEqual([h['step'] for h in info['history']],[0,2])
                rows[1]['step']=1;(p/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):snapshot_info(p,cfg,data,arm,{})
    def test_copy_never_overwrites_and_separates_prior_stall_telemetry(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);src=p/'src';out=p/'out';src.mkdir();out.mkdir()
            for name in ['metrics.jsonl','checkpoint-last.pt','resources.jsonl','resource-summary.json','alerts.jsonl']:(src/name).write_bytes(name.encode())
            copy_snapshot(src,out);self.assertEqual(sorted(f.name for f in out.iterdir()),['checkpoint-last.pt','metrics.jsonl'])
            with self.assertRaises(ValueError):copy_snapshot(src,out)
if __name__=='__main__':unittest.main()
