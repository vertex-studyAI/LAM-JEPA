#!/usr/bin/env bash
set -euo pipefail

python scripts/data/fetch_real_datasets.py
python scripts/data/build_curriculum_graph.py

printf "
Real dataset acquisition complete.
"
