#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
work_root="$(mktemp -d "${RUNNER_TEMP:-/tmp}/lam-jepa-pdf-determinism.XXXXXX")"

export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-946684800}"
export FORCE_SOURCE_DATE="${FORCE_SOURCE_DATE:-1}"
export TZ="${TZ:-UTC}"

build_variant() {
  local variant="$1"
  local build_root="${work_root}/${variant}"

  mkdir -p "${build_root}"
  git -C "${repo_root}" archive HEAD paper | tar -x -C "${build_root}"

  (
    cd "${build_root}/paper"
    for document in main icdm_teen_2026; do
      pdflatex -interaction=nonstopmode -halt-on-error "${document}.tex" >/dev/null
      bibtex "${document}" >/dev/null
      pdflatex -interaction=nonstopmode -halt-on-error "${document}.tex" >/dev/null
      pdflatex -interaction=nonstopmode -halt-on-error "${document}.tex" >/dev/null
      test -s "${document}.pdf"
    done
  )
}

build_variant first
build_variant second

for document in main icdm_teen_2026; do
  first="${work_root}/first/paper/${document}.pdf"
  second="${work_root}/second/paper/${document}.pdf"
  cmp "${first}" "${second}"
  printf '%s_sha256=%s\n' "${document}" "$(sha256sum "${first}" | awk '{print $1}')"
done

echo "PDF_DETERMINISM_VERIFIED"
