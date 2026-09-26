#!/usr/bin/env bash
# Run one explicit, locally pinned E01 research-v2 recipe on an existing host.
# Provisioning, network checkouts, implicit control jobs and partial-cohort
# summaries from the historical launcher are retired. Compare verified runs
# with plot_nanofold_experiment_metrics.py after separate explicit executions.
set -euo pipefail
for argument in "$@"; do
  case "${argument}" in
    --revision|--revision=*)
      echo "This entrypoint pins e01; use run_research_experiment.py for other revisions." >&2
      exit 2
      ;;
  esac
done
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/../source_provenance.json" ]]; then
  for argument in "$@"; do
    case "${argument}" in
      --export-dir|--export-dir=*)
        echo "Use the existing E01 runtime without a second export directory." >&2
        exit 2
        ;;
    esac
  done
  "${PYTHON_BIN:-python}" - "${SCRIPT_DIR}/../source_provenance.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    receipt = json.load(handle)
if receipt.get("revision") != "e01" or receipt.get("commit") != "a596b6b5c234e44aed79d66766f1515b9bc86b1b":
    raise SystemExit("This launcher requires a pinned E01 runtime.")
PY
  exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/run_research_experiment.py" "$@"
fi
exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/run_research_experiment.py" --revision e01 "$@"
