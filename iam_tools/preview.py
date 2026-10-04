"""Convert a small, strictly paired IAM sample into canonical JSON and previews."""
import argparse
import html
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def transcript_lines(path):
    # CSR is the handwritten transcription; OCR describes the printed prompt.
    source = Path(path).read_text(encoding='latin-1').splitlines()
    try:
        start = next(i for i, line in enumerate(source) if line.strip() == 'CSR:')
    except StopIteration:
        raise ValueError(f'{path}: missing CSR section') from None
    lines = [line.strip() for line in source[start + 1:] if line.strip()]
    if not lines:
        raise ValueError(f'{path}: empty CSR section')
    # Do not silently invent a pairing for transcription control markers.
    if any('%' in line for line in lines):
        raise ValueError(f'{path}: CSR contains % markers; needs manual alignment review')
    return lines


def writer_map(path):
    result = {}
    for line in Path(path).read_text(encoding='latin-1').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        if len(fields) < 2 or fields[0] in result:
            raise ValueError(f'{path}: malformed/duplicate form: {line}')
        result[fields[0]] = fields[1]  # Keep leading zeroes, if any.
    return result


def parse_line(xml_path, text, writer_id):
    root = ET.parse(xml_path).getroot()
    strokes = []
    tags = set()
    for node in root.iter():
        tags.add(node.tag.rsplit('}', 1)[-1])
    for stroke in root.iter():
        if stroke.tag.rsplit('}', 1)[-1] != 'Stroke':
            continue
        points = [[float(p.attrib[key]) for key in ('x', 'y', 'time')]
                  for p in stroke if p.tag.rsplit('}', 1)[-1] == 'Point']
        if not points or not all(math.isfinite(v) for p in points for v in p):
            raise ValueError(f'{xml_path}: empty stroke or non-finite coordinate')
        if any(a[2] > b[2] for a, b in zip(points, points[1:])):
            raise ValueError(f'{xml_path}: timestamps decrease within a stroke')
        strokes.append(points)
    if not strokes:
        raise ValueError(f'{xml_path}: no strokes')
    return {'id': Path(xml_path).stem, 'writer_id': writer_id, 'text': text,
            'strokes': strokes, 'source': {'xml': str(xml_path), 'xml_tags': sorted(tags),
            'transcript_section': 'CSR', 'coordinate_system': 'raw IAM; y increases down',
            'character_alignment': 'not inferred'}}


def render(sample, destination):
    # Aspect-preserving display transformation only: canonical coordinates stay raw.
    from PIL import Image, ImageDraw
    strokes = sample['strokes']
    points = [p for stroke in strokes for p in stroke]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = 1500 / max(xmax - xmin, 1)
    width, height = 1560, max(150, math.ceil((ymax - ymin) * scale) + 100)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), f"{sample['id']} | writer {sample['writer_id']}", fill='black')
    draw.text((20, 32), sample['text'], fill='black')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="20" y="22" font-size="14">{html.escape(sample["id"])} | writer {html.escape(sample["writer_id"])}</text>',
           f'<text x="20" y="44" font-size="14">{html.escape(sample["text"])}</text>']
    for stroke in strokes:
        xy = [((p[0]-xmin)*scale+30, (p[1]-ymin)*scale+75) for p in stroke]
        if len(xy) == 1:
            x,y=xy[0]
            draw.ellipse((x-1.5,y-1.5,x+1.5,y+1.5), fill='black')
            svg.append(f'<circle cx="{x}" cy="{y}" r="1.5"/>')
        else:
            draw.line(xy, fill='black', width=2)
            coords=' '.join(f'{x:.4f},{y:.4f}' for x,y in xy)
            svg.append(f'<polyline points="{coords}" fill="none" stroke="black" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
    destination = Path(destination)
    canvas.save(destination.with_suffix('.png'))
    destination.with_suffix('.svg').write_text('\n'.join(svg+['</svg>']))


def preview(raw, output, limit=6):
    raw, output = Path(raw), Path(output)
    if limit <= 0:
        raise ValueError('limit must be positive')
    writers = writer_map(raw / 'forms.txt')
    known = {w.attrib['name'] for w in ET.parse(raw / 'writers.xml').getroot().iter('Writer')}
    transcripts = {}
    for p in sorted((raw / 'ascii').rglob('*.txt')):
        if p.stem in transcripts:
            raise ValueError(f'duplicate transcript: {p.stem}')
        transcripts[p.stem] = p
    groups = {}
    for p in sorted((raw / 'lineStrokes').rglob('*.xml')):
        form, number = p.stem.rsplit('-', 1)
        groups.setdefault(form, []).append((int(number), p))
    output.mkdir(parents=True, exist_ok=True)
    records, rejected = [], []
    for form, files in sorted(groups.items()):
        try:
            lines = transcript_lines(transcripts[form])
            writer = writers[form]
            if writer not in known:
                raise ValueError(f'writer {writer} absent from writers.xml')
            if sorted(n for n,p in files) != list(range(1, len(lines)+1)):
                raise ValueError('stroke IDs do not exactly match 1-based CSR lines')
            samples = [parse_line(p, lines[n-1], writer) for n,p in sorted(files)]
        except (KeyError, ValueError, ET.ParseError) as exc:
            rejected.append({'form': form, 'reason': str(exc)})
            continue
        for sample in samples:
            sample['source']['transcript'] = str(transcripts[form])
            (output / f'{sample["id"]}.json').write_text(json.dumps(sample, indent=2, ensure_ascii=False)+'\n')
            render(sample, output / sample['id'])
            records.append({'id':sample['id'], 'writer_id':sample['writer_id'], 'text':sample['text'],
                            'strokes':len(sample['strokes']), 'points':sum(map(len,sample['strokes']))})
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    manifest = {'stage':'raw-canonical-preview', 'normalization':False, 'training':False,
                'samples':records, 'rejected_forms':rejected}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    items = ''.join(f'<article><h2>{html.escape(r["id"])}</h2><p>{html.escape(r["text"])}</p><img style="max-width:100%" src="{r["id"]}.svg"></article>' for r in records)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>IAM raw preview</title><h1>Raw IAM strokes — no normalization</h1>'+items)
    if not records:
        raise ValueError(f'No safely paired samples; see {output / "manifest.json"}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', default='data/raw/iam')
    parser.add_argument('--out', default='data/canonical/iam/preview')
    parser.add_argument('--limit', type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(preview(args.raw, args.out, args.limit), indent=2))

if __name__ == '__main__':
    main()
