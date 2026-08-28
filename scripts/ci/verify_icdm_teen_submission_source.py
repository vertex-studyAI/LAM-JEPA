#!/usr/bin/env python3
from pathlib import Path
import sys

TEX = Path('paper/icdm_teen_2026.tex')
GATE = Path('paper/ICDM_TEEN_SUBMISSION_GATE_20260828.md')

errors = []
if not TEX.exists():
    errors.append(f'missing {TEX}')
if not GATE.exists():
    errors.append(f'missing {GATE}')

if not errors:
    tex = TEX.read_text(encoding='utf-8')
    gate = GATE.read_text(encoding='utf-8')

    required_tex = {
        'IEEE conference class': r'\\documentclass[10pt,conference]{IEEEtran}',
        'High School Student marker': 'High School Student',
        'negative result full': '0.2549',
        'matched result': '0.2664',
        'paired adverse difference': '-0.0115',
        'frozen seed set': r'\\{1,2,3,4,5\\}',
        'locked test boundary': 'locked confirmatory test',
        'no superiority boundary': 'no claim of ARC superiority',
        'external reproduction pending': 'Independent external reproduction',
    }
    for label, token in required_tex.items():
        if token.lower() not in tex.lower():
            errors.append(f'missing required source boundary: {label}: {token}')

    forbidden_tex = {
        'false superiority': 'LAM-JEPA outperforms the matched supervised',
        'false transformer claim': 'LAM-JEPA is a Transformer',
        'false external reproduction': 'independently externally reproduced',
    }
    for label, token in forbidden_tex.items():
        if token.lower() in tex.lower():
            errors.append(f'forbidden claim present: {label}: {token}')

    required_gate = [
        '2026-08-30 AoE',
        'Maximum 5 pages total',
        'Single-blind review',
        'first-author affiliation must clearly include `High School Student`',
        'GO only if every hard gate above is closed',
    ]
    for token in required_gate:
        if token.lower() not in gate.lower():
            errors.append(f'missing venue/release gate token: {token}')

    owner_placeholders = [
        '[FIRST AUTHOR -- OWNER APPROVAL REQUIRED]',
        '[School / city / country -- OWNER APPROVAL REQUIRED]',
        '[Email -- OWNER APPROVAL REQUIRED]',
    ]
    if not all(p in tex for p in owner_placeholders):
        errors.append('owner-controlled metadata placeholders were altered before approval')

if errors:
    print('ICDM Teen submission-source verification: FAIL')
    for error in errors:
        print(f'- {error}')
    sys.exit(1)

print('ICDM Teen submission-source verification: PASS')
print('Static source/gate boundaries are present. This does NOT prove PDF page count, compilation, citation validity, or scientific rerun success.')
