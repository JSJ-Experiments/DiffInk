import json
from pathlib import Path
import tempfile
import unittest
from iam_tools.preview import transcript_lines, parse_line, writer_map, preview, render


class ParserTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, content):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return p

    def test_csr_not_ocr_and_spelling_preserved(self):
        p = self.write('page.txt', 'OCR:\nwrong\nCSR:\n\nA MOVE\nfrom nomnating\n')
        self.assertEqual(transcript_lines(p), ['A MOVE', 'from nomnating'])

    def test_no_csr_or_ambiguous_markers(self):
        for body in ('OCR:\nfoo', 'CSR:\nfoo %%%%% bar', 'CSR:\n'):
            with self.assertRaises(ValueError):
                transcript_lines(self.write('page.txt',body))

    def test_raw_order_and_duplicate_times(self):
        p = self.write('a01-000u-01.xml', '<Session><StrokeSet><Stroke><Point x="10" y="20" time="1"/><Point x="9" y="21" time="1"/></Stroke><Stroke><Point x="30" y="40" time="2"/></Stroke></StrokeSet></Session>')
        s = parse_line(p, 'hello', '00100')
        self.assertEqual(s['strokes'], [[[10.,20.,1.],[9.,21.,1.]],[[30.,40.,2.]]])
        self.assertEqual(s['writer_id'], '00100')
        render(s, self.root / 'render')
        self.assertTrue((self.root / 'render.png').exists())

    def test_invalid_strokes(self):
        for content in ('<Session/>', '<Session><Stroke/></Session>',
                        '<Session><Stroke><Point x="nan" y="1" time="1"/></Stroke></Session>',
                        '<Session><Stroke><Point x="1" y="1" time="2"/><Point x="2" y="2" time="1"/></Stroke></Session>'):
            with self.assertRaises(ValueError):
                parse_line(self.write('line.xml',content), 'text','001')

    def fixture(self):
        self.write('raw/forms.txt','# comment\na01-000u 00100 0\n')
        self.write('raw/writers.xml','<opt><Writer name="00100"/></opt>')
        self.write('raw/ascii/a01-000u.txt','OCR:\nwrong\nCSR:\ncorrect\n')
        self.write('raw/lineStrokes/a01-000u-01.xml','<Session><Stroke><Point x="1" y="2" time="3"/></Stroke></Session>')

    def test_pipeline(self):
        self.fixture()
        report = preview(self.root/'raw', self.root/'out',1)
        sample = json.loads((self.root/'out/a01-000u-01.json').read_text())
        self.assertEqual(sample['text'],'correct')
        self.assertEqual(sample['writer_id'],'00100')
        self.assertEqual(len(report['samples']),1)
        self.assertFalse(report['training'])

    def test_alignment_mismatch_fails_closed(self):
        self.fixture()
        self.write('raw/ascii/a01-000u.txt','CSR:\nline 1\nline 2\n')
        with self.assertRaises(ValueError):
            preview(self.root/'raw', self.root/'out')
        report=json.loads((self.root/'out/manifest.json').read_text())
        self.assertEqual(len(report['rejected_forms']),1)

    def test_writer_mapping_preserves_zeroes(self):
        p = self.write('forms.txt','# comment\na01-000u 00100 0\n')
        self.assertEqual(writer_map(p),{'a01-000u':'00100'})

if __name__ == '__main__':
    unittest.main()
