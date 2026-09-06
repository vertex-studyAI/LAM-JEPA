#!/usr/bin/env python3
from pathlib import Path
import re
import sys

TEX = Path('paper/icdm_teen_2026.tex')
BIB = Path('paper/references.bib')
GATE = Path('paper/ICDM_TEEN_SUBMISSION_GATE_20260828.md')
CITATION_AUDIT = Path('paper/ICDM_TEEN_CITATION_AUDIT_20260828.md')
PROVENANCE = Path('MANUSCRIPT_PROVENANCE.md')
PAPER_README = Path('paper/README.md')
EXTERNAL_REVIEW = Path('paper/EXTERNAL_REVIEW_VQ_COLLAPSE_CORRECTION_20260831.md')
CLAIM_LEDGER = Path('CLAIM_LEDGER.md')

errors = []
for path in (
    TEX,
    BIB,
    GATE,
    CITATION_AUDIT,
    PROVENANCE,
    PAPER_README,
    EXTERNAL_REVIEW,
    CLAIM_LEDGER,
):
    if not path.exists():
        errors.append(f'missing {path}')

if not errors:
    tex = TEX.read_text(encoding='utf-8')
    bib = BIB.read_text(encoding='utf-8')
    gate = GATE.read_text(encoding='utf-8')
    citation_audit = CITATION_AUDIT.read_text(encoding='utf-8')
    provenance = PROVENANCE.read_text(encoding='utf-8')
    paper_readme = PAPER_README.read_text(encoding='utf-8')
    external_review = EXTERNAL_REVIEW.read_text(encoding='utf-8')
    claim_ledger = CLAIM_LEDGER.read_text(encoding='utf-8')

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
        'bounded external reproduction': 'One independent frozen-protocol external rerun/review',
        'external collapse diagnosis': 'single-code vector-quantizer collapse',
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
        'Frozen Method and Decision Gates',
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
        'false broad external reproduction': 'independently externally reproduced across multiple sites',
        'false peer-review claim': 'peer-reviewed publication validates',
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

    required_external_review = [
        'A genuinely external reviewer independently reran the frozen ARC protocol',
        'all retained full/control runs collapse to constant classifiers',
        'VQ assignment | **1 of 32 codes**',
        'Quantizer removal solves the ARC task.',
        'The external review constitutes peer-reviewed publication or broad independent replication.',
    ]
    for token in required_external_review:
        if token.lower() not in external_review.lower():
            errors.append(f'missing external-review evidence boundary: {token}')

    required_claim_ledger = [
        'C21 | One genuinely external frozen-protocol rerun/review',
        'VERIFIED, bounded external reproduction',
        'ONE_EXTERNAL_FROZEN_PROTOCOL_REPRODUCTION',
        'BROAD_INDEPENDENT_REPLICATION',
    ]
    for token in required_claim_ledger:
        if token.lower() not in claim_ledger.lower():
            errors.append(f'missing claim-ledger external-review boundary: {token}')

    required_readme = [
        'One genuinely external frozen-protocol rerun/review has already reproduced the retained headline metrics',
        'it is not broad multi-site replication or peer review',
        'A second independent rerun/reviewer remains a stronger promotion gate',
    ]
    for token in required_readme:
        if token.lower() not in paper_readme.lower():
            errors.append(f'missing paper README external-review boundary: {token}')

    stale_readme_claims = [
        'Independent external reproduction is also still pending',
        'no genuinely independent reproduction/review report has been returned yet',
    ]
    for token in stale_readme_claims:
        if token.lower() in paper_readme.lower():
            errors.append(f'stale external-review state present in paper README: {token}')

    # These TeX-safe placeholders are deliberately required until owner-controlled
    # metadata is truthfully supplied. Their presence means the source is NOT upload-ready.
    # Keep them bracket-free inside IEEEtran author blocks to avoid parser ambiguity.
    owner_placeholders = [
        r'\textit{FIRST AUTHOR -- OWNER APPROVAL REQUIRED}',
        r'\textit{School / city / country -- OWNER APPROVAL REQUIRED}',
        r'\textit{Email -- OWNER APPROVAL REQUIRED}',
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
print('Static scientific, protocol, venue, citation, provenance, and bounded external-review boundaries are present.')
print('This does NOT prove successful LaTeX compilation, PDF page count, font/overflow quality, owner metadata, broad independent replication, peer review, or submission.')
