#!/usr/bin/env python3
"""Normalize pdfTeX's random document ID for release reproducibility.

The byte length is preserved so xref offsets remain valid.  This script only
rewrites the two hexadecimal IDs in the trailer; page content and metadata are
unchanged.
"""
from pathlib import Path
import re, sys
if len(sys.argv) != 3:
    raise SystemExit('usage: normalize_pdf_metadata.py INPUT OUTPUT')
raw = Path(sys.argv[1]).read_bytes()
pat = re.compile(rb'/ID \[<[0-9A-Fa-f]{32}> <[0-9A-Fa-f]{32}>\]')
replacement = b'/ID [<00000000000000000000000000000000> <00000000000000000000000000000000>]'
out, n = pat.subn(replacement, raw)
if n != 1:
    raise SystemExit(f'expected one PDF ID trailer, found {n}')
Path(sys.argv[2]).write_bytes(out)
print(f'normalized {n} PDF ID trailer')
