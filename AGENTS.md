# Repository agent rules

## Blind test boundary

- Do not open, read, search, list, screenshot, summarize, or inspect test-split media, annotations, ROI files, locked-manifest contents, or per-sample test artifacts.
- Do not inspect files below `runs/test/` except `summary.json` and `metrics/evaluation.json`; those two files must contain aggregate-only results.
- Run the test split only when the user explicitly requests it. An agent may report aggregate TP, FP, FN, precision, recall, and F1, but must not report or derive per-sample test information.
- Never tune code, thresholds, or configuration from test results. Diagnose and implement changes using validation/development data only.
- Synthetic fixtures that exercise blind-test tooling are allowed and must not contain identifiers or values copied from the real test split.
