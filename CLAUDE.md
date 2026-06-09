# CLAUDE.md

Guidance for AI agents (and humans) working in this repo. Keep it short and current.

## Environment
- Python **3.13** venv at `.venv`. Run code via `.venv\Scripts\python.exe` or `uv run`.
- **Do not let numpy resolve to 1.26.x.** It has no cp313 wheel, so it compiles via
  MINGW-W64 and segfaults on import ("CRASHES ARE TO BE EXPECTED"), taking pandas down
  with it. Pin `numpy>=2.1` (ships a working cp313 wheel).
- Keep `pandas==2.3.3`. pandas 3.0 removed the `"5S"` freq alias — use lowercase `"5s"`.
- `uv.lock` is committed; install with `uv sync` to reproduce the pinned environment.

## Data & conventions
- Telemetry is **long-form**: columns `[timestamp, variable, value]`. This is the native
  SMAP / OPS-SAT format, ingest real data without schema transforms.
- Detectors follow the **PyOD / scikit-learn convention**: `fit` / `decision_function` /
  `predict` / `is_anomaly`, with `decision_scores_`, `threshold_`, `labels_` as post-fit
  attributes.
- **Fit and transform stay separate.** `normalize_fit()` returns a stats dict and does not
  mutate data; reuse that dict at inference to prevent leakage. It is intentionally *not*
  wired into `pipeline()` (which always returns a DataFrame).
- Use lowercase pandas freq aliases (`"5s"`, not `"5S"`).

## Workflow
- Don't commit unless asked.
- Personal/local files are gitignored: `memory/`, `CLAUDE.local.md`,
  `.claude/settings.local.json`. Shared guidance (this file) is committed.
