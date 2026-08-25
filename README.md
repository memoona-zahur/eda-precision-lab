# EDA Precision Lab — Week 06 · Day 1

**Real Relationships vs. Real Comparisons, and Verified Findings**

Dataset: `students.csv` — 600 rows × 6 columns, verbatim assignment generator (seed=21), SHA-256 fingerprinted.

## Deliverables

| # | Kata step | File |
|---|---|---|
| 1 | Genuine relationship + Pearson r + trend (study hours) | `chart_relationship_study.png` |
| 2 | Null relationship written up as a real finding (sleep hours) | `chart_null_sleep.png` |
| 3 | Categorical comparison with shuffle-test justification | `chart_comparison_section.png` |
| 4 | Line-chart trap built deliberately + false impression named | `chart_trap_line.png` |
| 5 | Layout collision: built → confirmed in saved PNG → fixed | `chart_collision_broken.png` / `chart_collision_fixed.png` |
| 6 | Arbitrary vs deliberate color; natural order explained | `chart_color_arbitrary.png` / `chart_color_deliberate.png` |
| 7 | Self-audit: 3 claimed numbers vs 3 independent recomputations | in-notebook table |

All 8 charts element-collision-guarded before save.

## How to reproduce

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python generate_data.py               # students.csv (SHA-256 recorded in notebook)
.venv/bin/jupyter nbconvert --execute --inplace \
    week6_day1_precision_lab.ipynb              # all outputs + PNGs
.venv/bin/python -m pytest test_precision_checks.py -v   # 8/8 pass
```

## Tests

`test_precision_checks.py` — 8 checks:

1. Dataset shape + exact columns (600×6)
2. Generator deterministic (re-run reproduces committed SHA)
3. Study↔score relationship strong (r > 0.6)
4. Sleep↔score genuinely null (|r| < 0.10)
5. Section C bonus visible (mean_C − mean_A ≈ +4)
6. All 8 PNGs exist and are valid images
7. Collision pair: broken vs fixed differ (fix actually changed something)
8. Executed notebook outputs contain `AUDIT: ALL THREE MATCH`
