import unittest
from iam_tools.report_fullset_comparison import phase_diagnostic


class PhaseDiagnosticsTests(unittest.TestCase):
    def rows(self, values):
        return dict(lines=[dict(sample_id=sid,mu=dict(geometry=dict(point_index_mod8=[dict(vector_rmse=x) for x in v]))) for sid,v in values.items()])

    def test_explicit_training_subset_excludes_held_out(self):
        data=self.rows({'train':[1]*8,'val':[100]*8})
        result=phase_diagnostic(data,['train'])
        self.assertEqual(result['mean_per_line_phase_vector_rmse'],[1]*8)
        self.assertEqual(result['largest_to_smallest_mean_phase_ratio'],1.)

    def test_equal_per_line_average_and_zero_error(self):
        data=self.rows({'a':[1,2,1,2,1,2,1,2],'b':[3,4,3,4,3,4,3,4]})
        result=phase_diagnostic(data,['a','b'])
        self.assertEqual(result['mean_per_line_phase_vector_rmse'],[2,3,2,3,2,3,2,3])
        self.assertEqual(result['largest_to_smallest_mean_phase_ratio'],1.5)
        self.assertEqual(phase_diagnostic(self.rows({'a':[0]*8}),['a'])['largest_to_smallest_mean_phase_ratio'],1.)

    def test_invalid_subset_and_nonfinite_phases_fail(self):
        data=self.rows({'a':[1]*8})
        for ids in ([],['a','a'],['missing']):
            with self.assertRaises(ValueError):phase_diagnostic(data,ids)
        for values in ([1]*7,[float('nan')]*8,[-1]*8):
            with self.assertRaises(ValueError):phase_diagnostic(self.rows({'a':values}),['a'])

class ReportPointerTests(unittest.TestCase):
    def test_pointer_is_atomic_dated_and_preserves_old_gallery(self):
        import tempfile
        from pathlib import Path
        from iam_tools.report_pointer import publish_pointer
        with tempfile.TemporaryDirectory() as tmp:
            family=Path(tmp);report=family/'20261006-000001/report';report.mkdir(parents=True)
            (report/'index.html').write_text('dated')
            old=family/'latest';old.mkdir();(old/'old.png').write_bytes(b'unchanged')
            self.assertEqual(publish_pointer(family,report),'../20261006-000001/report/index.html')
            self.assertIn('20261006-000001/report/index.html',(old/'index.html').read_text())
            self.assertEqual((old/'old.png').read_bytes(),b'unchanged')
            self.assertEqual((report/'index.html').read_text(),'dated')
            self.assertFalse(list(old.glob('*.tmp')))
            with self.assertRaises(ValueError):publish_pointer(family,old)

class DiversityContractTests(unittest.TestCase):
    def test_writer_scope_is_explicit_and_survives_reload(self):
        from iam_tools.writer_expansion import resolve_writer_scope
        self.assertEqual(resolve_writer_scope({}),'10174')
        self.assertIsNone(resolve_writer_scope({},None))
        self.assertIsNone(resolve_writer_scope(dict(config=dict(research_writer_id=None))))
        self.assertEqual(resolve_writer_scope(dict(config=dict(research_writer_id='42'))),'42')
        with self.assertRaises(ValueError):resolve_writer_scope({},'bad')

    def test_gallery_pagination_retains_every_line_in_order(self):
        from iam_tools.report_writer_expansion import paginate_ids
        ids=list(map(str,range(192)));pages=paginate_ids(ids)
        self.assertEqual(len(pages),12);self.assertEqual([i for p in pages for i in p],ids)
        self.assertTrue(all(len(p)<=16 for p in pages))
        with self.assertRaises(ValueError):paginate_ids(ids,0)
        with self.assertRaises(ValueError):paginate_ids(['x','x'])

    def test_unknown_study_family_fails_before_gpu_or_data(self):
        from iam_tools.conditioning_study import run
        with self.assertRaises(ValueError):run('missing','missing',family='../../elsewhere')

    def test_retained_source_groups_survive_selected_checkpoint_reload(self):
        from iam_tools.writer_expansion import retention_groups
        parent=dict(sample_ids=['a','b'],provenance=dict(splits=dict(train=['a','b'])))
        train=['a','b','c'];val={'v':'10174','w':'42'}
        initial=retention_groups(parent,train,val)
        selected=dict(sample_ids=train,provenance=dict(splits=dict(train=train,**initial)))
        self.assertEqual(retention_groups(selected,train,val),initial)
        self.assertEqual(initial['added'],['c']);self.assertEqual(initial['source_trained'],['a','b'])

    def test_empty_added_group_omitted_and_reference_leakage_rejected(self):
        from iam_tools.writer_expansion import retention_groups
        parent=dict(sample_ids=['a'])
        self.assertNotIn('added',retention_groups(parent,['a'],{'v':'10174'}))
        with self.assertRaises(ValueError):retention_groups(parent,['b'],{'a':'10174'})
        with self.assertRaises(ValueError):retention_groups(parent,['a'],{'a':'10174'})
