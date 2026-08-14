# CLAUDE.md

Guidance for AI agents (and humans) working in this repo. Keep it short and current.

## Environment
- Python **3.13** venv at `.venv`. Run code via `.venv\Scripts\python.exe` or `uv run`.
- **Do not let numpy resolve to 1.26.x.** It has no cp313 wheel, so it compiles via
  MINGW-W64 and segfaults on import ("CRASHES ARE TO BE EXPECTED"), taking pandas down
  with it. Pin `numpy>=2.1` (ships a working cp313 wheel).
- Keep `pandas==2.3.3`. pandas 3.0 removed the `"5S"` freq alias — use lowercase `"5s"`.
- `uv.lock` is committed; install with `uv sync` to reproduce the pinned environment.
- **torch lives in the `deep` extra** and is imported lazily at fit time. Use
  `uv sync --extra deep` when touching `models/deep/`; without it those tests drop out of
  collection instead of failing, so the run looks green while covering nothing.
- The codegen tests compile the emitted C, so they need a C compiler on PATH. They skip
  themselves without one.

## Data & conventions
- Telemetry is **long-form**: columns `[timestamp, variable, value]`. This is the native
  SMAP / MSL format, ingest real data without schema transforms.
- Detectors follow the **PyOD / scikit-learn convention**: `fit` / `decision_function` /
  `predict` / `is_anomaly`, with `decision_scores_`, `threshold_`, `labels_` as post-fit
  attributes.
- **Fit and transform stay separate.** `normalize_fit()` returns a stats dict and does not
  mutate data; reuse that dict at inference to prevent leakage. It is intentionally *not*
  wired into `pipeline()` (which always returns a DataFrame).
- Use lowercase pandas freq aliases (`"5s"`, not `"5S"`).
- **`score_channels` separates context from alarms.** Every channel feeds the model; only
  the named ones contribute to the deviation score.

## Evaluation
- **Never quote point-adjusted F1 alone.** It credits a whole labelled segment to one
  flagged sample, so on SMAP a uniform random detector beats every trained configuration
  while scoring 0.000 event-level. Report event-level metrics beside it, and keep the
  benchmark's `random@all` row.
- Thresholds are chosen **without labels** (`threshold_for_budget`, `dynamic_threshold`).
  An oracle-threshold number is not one anyone can deploy.
- Measured effect of each choice: `docs/source/user_guide/anomaly_scoring.rst`.

## Generated C
- Follows the **NASA/JPL Power of Ten** rules: no allocation, no recursion, fixed loop
  bounds, status codes everywhere, no file-scope mutable state.
- **Deterministic, with no timestamp**, so the same fitted detector reproduces the sources
  byte for byte. Do not add one; a test enforces it.
- Files are stamped with the generator version (`KANGDN_VERSION`), and the banner says the
  interface is unstable until 1.0.0.
- Platform code lives in `targets/`. **Nothing in `targets/` may leak into the generated
  sources.**

## Docs
- Sphinx with `pydata-sphinx-theme`. The navbar is built from **section landing pages**
  (`user_guide/index`, `api/index`, ...), so add a new page to its section index, not to
  the root toctree.
- Build with `-E -a` when checking: incremental builds cache over real docstring errors.
  The build must be warning-free.
- Update `docs/source/_extra/llms.txt` whenever the public API changes.

## Workflow
- Personal/local files are gitignored: `memory/`, `CLAUDE.local.md`,
  `.claude/settings.local.json`. Shared guidance (this file) is committed.
- Four checks gate CI; run all of them before pushing:
  ```
  uv run ruff check src tests examples
  uv run ruff format --check src tests examples
  uv run pytest -q --cov
  uv run --group docs sphinx-build -b html docs/source build/docs
  ```
