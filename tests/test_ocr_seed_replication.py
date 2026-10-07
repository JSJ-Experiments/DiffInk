from pathlib import Path
import sys,copy,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'third_party/DiffInk' if (ROOT/'third_party/DiffInk').exists() else ROOT
sys.path.insert(0,str(REPO))
from iam_tools.ocr_seed_replication import compare,publish,SEEDS,FIXED_FIELDS,POOL_SHA,SHA,REFERENCE


def fixture():
    records={'d1':dict(writer_id='1',text='abcd'),'d2':dict(writer_id='2',text='abcdef'),'h':dict(writer_id='3',text='abcd')}
    summaries=[]
    for seed in SEEDS:
        groups=dict(dev=dict(mu=dict(cer=.3)),held_out=dict(mu=dict(cer=.25)))
        four=dict(dev=dict(mu=dict(cer=.1)),held_out=dict(mu=dict(cer=0.)))
        configs={};results={}
        for arm,g in [('points8',groups),('points4',four)]:
            c=dict.fromkeys(FIXED_FIELDS);c.update(seed=seed,train_ids=['train'],dev_ids=['d1','d2'],held_out_ids=['h'],schedule_sha256='schedule')
            configs[arm]=c;results[arm]=dict(last_step=8000,stop_reason='budget_completed',entire_codec_bitwise_unchanged=True,best_step=8000,selected=g,selected_sha256='checkpoint',initial_head_tensor_sha256=str(seed),sample_schedule_sha256='schedule')
        summaries.append(dict(pool_manifest_sha256=POOL_SHA,source_sha256=SHA,results=results,configs=configs,paired=dict(equal_updates=True),
            cpu_reload={a:dict(mean_cpu_gpu_transcript_differences=[]) for a in results},cpu_line_error_changes={
            'dev':dict(lines=[dict(sample_id='d2',errors8=2,errors4=1),dict(sample_id='d1',errors8=1,errors4=0)]),
            'held_out':dict(lines=[dict(sample_id='h',errors8=1,errors4=0)])}))
    return summaries,records

class OCRReplicationTests(unittest.TestCase):
    def test_writer_counts_and_averages_repeat_same_lines_not_independent_pool(self):
        (a,b),records=fixture();r=compare(a,b,records)
        self.assertTrue(r['both_seeds_four_point_improves']['dev']);self.assertEqual(r['mean_cer_across_seeds']['dev']['points4'],.1)
        self.assertEqual(r['per_writer']['137']['dev']['1']['characters'],4);self.assertEqual(r['per_writer']['42']['dev']['2']['cer8'],2/6)
        self.assertEqual(r['paired_line_changes']['137']['dev']['characters'],10);self.assertIn('NOT256',r['caveats'])
    def test_reject_uncontrolled_lr_labels_features_schedule_and_same_initialization(self):
        (a,b),records=fixture()
        for field,value in [('base_lr',.1),('feature_stats',{'mean':[1,2]}),('dev_ids',['d2']),('train_ids',['dev'])]:
            c=copy.deepcopy(b);c['configs']['points8'][field]=value
            with self.assertRaises(ValueError):compare(a,c,records)
        c=copy.deepcopy(b);c['results']['points8']['initial_head_tensor_sha256']='42'
        with self.assertRaises(ValueError):compare(a,c,records)
        c=copy.deepcopy(b);c['results']['points4']['sample_schedule_sha256']='changed'
        with self.assertRaises(ValueError):compare(a,c,records)
    def test_reject_cherry_picked_seed_incomplete_budget_or_reload_disagreement(self):
        (a,b),records=fixture()
        for mutation in [lambda c:c['configs']['points8'].update(seed=138),lambda c:c['results']['points4'].update(last_step=7999),
            lambda c:c['results']['points8'].update(stop_reason='wall_limit'),lambda c:c['paired'].update(equal_updates=False),
            lambda c:c['cpu_reload']['points4'].update(mean_cpu_gpu_transcript_differences=['d1'])]:
            c=copy.deepcopy(b);mutation(c)
            with self.assertRaises(ValueError):compare(a,c,records)
    def test_reject_duplicate_counts_negative_errors_and_cer_mismatch(self):
        (a,b),records=fixture()
        for mutation in [lambda c:c['cpu_line_error_changes']['dev']['lines'].append(dict(sample_id='d1',errors8=1,errors4=0)),
            lambda c:c['cpu_line_error_changes']['dev']['lines'][0].update(errors4=-1),
            lambda c:c['results']['points4']['selected']['dev']['mu'].update(cer=.2)]:
            c=copy.deepcopy(b);mutation(c)
            with self.assertRaises(ValueError):compare(a,c,records)
    def test_publisher_rejects_changed_immutable_reference_before_data_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/REFERENCE;p.parent.mkdir(parents=True);p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'immutable seed42'):publish(root/'new',root)

if __name__=='__main__':unittest.main()
