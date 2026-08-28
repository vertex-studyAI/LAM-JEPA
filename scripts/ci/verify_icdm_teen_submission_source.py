#!/usr/bin/env python3
from pathlib import Path
import re
import sys

TEX = Path('paper/icdm_teen_2026.tex')
BIB = Path('paper/references.bib')
GATE = Path('paper/ICDM_TEEN_SUBMISSION_GATE_20260828.md')
CITATION_AUDIT = Path('paper/ICDM_TEEN_CITATION_AUDIT_20260828.md')
PROVENANCE = Path('MANUSCRIPT_PROVENANCE.md')

errors = []
for path in (TEX, BIB, GATE, CITATION_AUDIT, PROVENANCE):
    if not path.exists():
        errors.append(f'missing {path}')

if not errors:
    tex = TEX.read_text(encoding='utf-8')
    bib = BIB.read_text(encoding='utf-8')
    gate = GATE.read_text(encoding='utf-8')
    citation_audit = CITATION_AUDIT.read_text(encoding='utf-8')
    provenance = PROVENANCE.read_text(encoding='utf-8')

    required_tex = {
        'IEEE conference class': r'\documentclass[10pt,conference]{IEEEtran}',
        'High School Student marker': 'High School Student',
        'negative result full': '0.2549',
        'matched result': '0.2664',
        'paired adverse difference': '-0.0115',
        'frozen seed set': r'\{1,2,3,4,5\}',
        'training eligibility count': '1,117/1,119',
        'validation eligibility count': '295/299',
        'frozen epoch count': '20 epochs',
        'frozen batch size': 'batch size 32',
        'frozen learning rate': r'3\times10^{-4}',
        'locked test boundary': 'locked confirmatory test',
        'no superiority boundary': 'no claim of ARC superiority',
        'external reproduction pending': 'Independent external reproduction',
        'scientific source SHA': '760aa7f9a73a177d5ff4ba7eb470f7e68ace63cb',
        'active parameter count LAM': '86,372',
        'active parameter count matched': '86,644',
        'sample standard deviation statement': 'sample standard deviations',
        'bootstrap interval statement': 'bootstrap 95\\% intervals',
    }
    for label, token in required_tex.items():
        if token.lower() not in tex.lower():
            errors.append(f'missing required source boundary: {label}: {token}')

    required_sections = [
        'Introduction',
        'Related Work and Claim Boundary',
        'Frozen Method',
        'Results',
        'Reproducibility and Failure Analysis',
        'Limitations and Conclusion',
    ]
    for section in required_sections:
        if f'\\section{{{section}}}' not in tex:
            errors.append(f'missing required manuscript section: {section}')

    forbidden_tex = {
        'false superiority': 'LAM-JEPA outperforms the matched supervised',
        'false transformer claim': 'LAM-JEPA is a Transformer',
        'false planner benefit': 'planner improves ARC accuracy',
        'false target benefit': 'target path improves ARC accuracy',
        'false external reproduction': 'independently externally reproduced',
        'test-set result claim': 'ARC test accuracy',
    }
    for label, token in forbidden_tex.items():
        if token.lower() in tex.lower():
            errors.append(f'forbidden claim present: {label}: {token}')

    # Every citation used by the submission source must exist in the shared BibTeX
    # and must have an explicit primary-source audit row.
    cite_groups = re.findall(r'\\cite\{([^}]+)\}', tex)
    cited_keys = sorted({key.strip() for group in cite_groups for key in group.split(',') if key.strip()})
    bib_keys = set(re.findall(r'@\w+\{([^,]+),', bib))
    for key in cited_keys:
        if key not in bib_keys:
            errors.append(f'citation key used in TeX but absent from BibTeX: {key}')
        if f'`{key}`' not in citation_audit:
            errors.append(f'citation key lacks primary-source audit entry: {key}')
    if not cited_keys:
        errors.append('submission source contains no citations')

    required_citations = {
        'assran2023ijepa',
        'vandenOord2017vqvae',
        'ye2024lapa',
        'clark2018arc',
        'he2023debertav3',
        'pineau2021reproducibility',
    }
    missing_required_citations = sorted(required_citations - set(cited_keys))
    if missing_required_citations:
        errors.append('required related-work citations absent from compact source: ' + ', '.join(missing_required_citations))

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

    required_provenance = [
        'artifact ID: **`9003785715`**',
        'test_split_accessed=false',
        'INTERNAL_PROVENANCE_GREEN',
        'SCIENTIFIC_RESULT_NEGATIVE',
    ]
    for token in required_provenance:
        if token.lower() not in provenance.lower():
            errors.append(f'missing provenance boundary: {token}')

    # These placeholders are deliberately required until owner-controlled metadata
    # is truthfully supplied. Their presence means the source is NOT upload-ready.
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
print(f'Cross-checked {len(cited_keys)} cited keys against BibTeX and the primary-source citation audit.')
print('Static scientific, protocol, venue, citation, and provenance boundaries are present.')
print('This does NOT prove successful LaTeX compilation, PDF page count, font/overflow quality, owner metadata, external reproduction, or submission.')
