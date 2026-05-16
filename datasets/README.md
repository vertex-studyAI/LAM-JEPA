# LAM-JEPA Real Dataset Bundle

This bundle is a reproducible, real-data-only acquisition and preprocessing layer.
It does **not** redistribute third-party corpora inside the zip. Instead, it gives you a fully working downloader/build system for the official public sources.

## What is included
- dataset registry YAMLs
- download scripts
- preprocessing scripts
- student-trace builders
- OOD split builders
- curriculum/misconception metadata
- reproducibility manifests

## What is not included
- the raw third-party corpora themselves, because those are large and subject to their own licenses / access rules

## Quickstart
1. Install requirements from `requirements-datasets.txt`
2. Run `bash scripts/data/full_build_pipeline.sh`
3. Point your repo at `datasets/`

## Notes
- GSM8K and MATH are referenced from their official GitHub repositories.
- ASSISTments and EdNet may require manual download / registration depending on the release you use.
