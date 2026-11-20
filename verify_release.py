# SPDX-License-Identifier: GPL-3.0-only
"""Verify the shipped-file checksums (generated files are ignored)."""
import hashlib
from pathlib import Path

root=Path(__file__).resolve().parent
for line in (root/'MANIFEST.sha256').read_text(encoding='utf-8').splitlines():
    expected,name=line.split('  ',1)
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:
        raise SystemExit('Checksum mismatch: '+name)
print('Release checksums: PASS')
