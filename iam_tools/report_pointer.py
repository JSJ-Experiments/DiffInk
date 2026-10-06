"""Publish a small atomic link to immutable dated reports, not mixed galleries."""
import html
import os
from pathlib import Path
import uuid


def publish_pointer(family, report):
    family, report = Path(family), Path(report)
    relative = report.resolve().relative_to(family.resolve())
    if relative.parts[0] == 'latest' or not (report/'index.html').is_file():
        raise ValueError('existing dated report under the same family required')
    latest = family/'latest'; latest.mkdir(exist_ok=True)
    href = '../'+relative.as_posix()+'/index.html'
    escaped = html.escape(href, quote=True)
    page = f'<meta charset="utf-8"><title>Latest research report</title><meta http-equiv="refresh" content="0;url={escaped}"><p><a href="{escaped}">Open the dated report</a>. Latest published is not necessarily the best model.</p>'
    tmp = latest/('index-'+uuid.uuid4().hex+'.tmp')
    try:
        tmp.write_text(page); os.replace(tmp, latest/'index.html')
    finally:
        if tmp.exists(): tmp.unlink()
    return href
