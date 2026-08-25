"""Adversarial verification suite — Week 06 Day 1 EDA Precision Lab.

Design principle: catch work that LOOKS correct but is wrong.
Split into two tiers:
  - Structural checks (dataset, generator, PNGs, notebook)
  - Adversarial checks (magic-byte, stats canary, ranges, pin enforcement)
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats

ROOT = Path(__file__).parent
REQUIRED_PNGS = [
    "chart_relationship_study.png",
    "chart_null_sleep.png",
    "chart_comparison_section.png",
    "chart_trap_line.png",
    "chart_collision_broken.png",
    "chart_collision_fixed.png",
    "chart_color_arbitrary.png",
    "chart_color_deliberate.png",
]
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def load_df():
    return pd.read_csv(ROOT / "students.csv")


# ──────────────────────────────────────────────────────────────────────
# Tier 1 — Structural checks (given-style)
# ──────────────────────────────────────────────────────────────────────

def test_dataset_shape_and_exact_columns():
    df = load_df()
    expected = ["student_id", "class_section", "study_hours_per_week",
                "sleep_hours_per_night", "attendance_pct", "exam_score"]
    assert df.shape == (600, 6), f"shape {df.shape}"
    assert list(df.columns) == expected
    assert df["student_id"].is_unique and df.notna().all().all()


def test_generator_is_deterministic_and_unmodified():
    """Re-running the verbatim generator must reproduce students.csv byte-for-byte."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, str(ROOT / "generate_data.py")],
                       cwd=tmp, check=True, capture_output=True)
        fresh = hashlib.sha256((Path(tmp) / "students.csv").read_bytes()).hexdigest()
    committed = hashlib.sha256((ROOT / "students.csv").read_bytes()).hexdigest()
    assert fresh == committed, "generator output drifted from the committed dataset"


def test_real_relationship_study_hours_is_strong():
    df = load_df()
    r, p = stats.pearsonr(df["study_hours_per_week"], df["exam_score"])
    assert r > 0.60 and p < 1e-30, f"designed strong relationship missing: r={r:.3f}"


def test_null_relationship_sleep_is_genuinely_null():
    df = load_df()
    r, _ = stats.pearsonr(df["sleep_hours_per_night"], df["exam_score"])
    assert abs(r) < 0.10, f"sleep should be unrelated by design: r={r:.3f}"


def test_section_c_carries_the_designed_bonus():
    df = load_df()
    means = df.groupby("class_section")["exam_score"].mean()
    bonus = means["C"] - means["A"]
    assert abs(bonus - 4.0) < 1.5, f"designed +4 section-C effect not visible: {bonus:+.2f}"
    assert abs(means["A"] - means["B"]) < 2.0, "A and B are twins by design"


def test_all_required_pngs_exist_and_are_valid_images():
    for name in REQUIRED_PNGS:
        path = ROOT / name
        assert path.exists(), f"missing deliverable: {name}"
        im = Image.open(path)
        im.verify()
        assert path.stat().st_size > 5_000, f"suspiciously tiny image: {name}"


def test_collision_pair_kept_and_genuinely_different():
    """Both Step-6 versions must survive — the lesson is the pair, not just the fix."""
    broken = (ROOT / "chart_collision_broken.png").read_bytes()
    fixed = (ROOT / "chart_collision_fixed.png").read_bytes()
    assert len(broken) > 0 and len(fixed) > 0
    assert broken != fixed, "broken and fixed versions are identical — the fix did nothing"


def test_executed_notebook_contains_passing_self_audit():
    """The self-audit is only real if the executed outputs show ALL THREE MATCH."""
    nb = json.loads((ROOT / "week6_day1_precision_lab.ipynb").read_text())
    streams = "".join(
        "".join(out.get("text", []))
        for cell in nb["cells"] if cell["cell_type"] == "code"
        for out in cell.get("outputs", []) if out.get("output_type") == "stream"
    )
    counts = [c.get("execution_count") for c in nb["cells"] if c["cell_type"] == "code"]
    assert "AUDIT: ALL THREE MATCH" in streams, "self-audit did not pass in executed outputs"
    errors = [o for c in nb["cells"] if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, "executed notebook contains error outputs"
    non_empty = [c for c in counts if c is not None]
    assert non_empty == sorted(non_empty) and len(non_empty) > 0, "execution order broken"


# ──────────────────────────────────────────────────────────────────────
# Tier 2 — Adversarial checks (self-invented)
# ──────────────────────────────────────────────────────────────────────

def test_pngs_are_real_pngs_not_renamed_text_files():
    """Magic-byte check: PIL verify accepts renamed text files; this does not."""
    for name in REQUIRED_PNGS:
        head = (ROOT / name).read_bytes()[:8]
        assert head == PNG_MAGIC, f"{name} is not a valid PNG (head={head!r})"


def test_requirements_pins_the_stack():
    """Every core library used in the notebook must be pinned in requirements.txt."""
    reqs = (ROOT / "requirements.txt").read_text()
    for pkg in ("pandas==", "numpy==", "matplotlib==", "scipy=="):
        assert pkg in reqs, f"{pkg} version not pinned in requirements.txt"


def test_study_correlation_within_plausible_range():
    """Statistical canary: if the correlation drifts far from 0.69, something changed."""
    df = load_df()
    r, _ = stats.pearsonr(df["study_hours_per_week"], df["exam_score"])
    assert 0.55 < r < 0.80, f"study-hours r outside plausible range: {r:.3f}"


def test_sleep_correlation_within_plausible_range():
    """Statistical canary: sleep r should be near zero, not moderate."""
    df = load_df()
    r, _ = stats.pearsonr(df["sleep_hours_per_night"], df["exam_score"])
    assert abs(r) < 0.15, f"sleep-hours r drifted from null: {r:.3f}"


def test_exam_scores_within_physical_plausibility():
    """Scores were clipped to [0, 100] by the generator — no value should escape."""
    df = load_df()
    assert df["exam_score"].min() >= 0, "exam score below 0"
    assert df["exam_score"].max() <= 100, "exam score above 100"


def test_study_hours_nonnegative_and_clipped():
    """Generator clips study hours at 0 — no negatives survive."""
    df = load_df()
    assert df["study_hours_per_week"].min() >= 0, "negative study hours"
    assert df["study_hours_per_week"].max() < 30, "study hours implausibly high"


def test_sleep_hours_within_clip_bounds():
    """Generator clips sleep to [3, 10]."""
    df = load_df()
    assert df["sleep_hours_per_night"].min() >= 3, "sleep below clip floor"
    assert df["sleep_hours_per_night"].max() <= 10, "sleep above clip ceiling"


def test_attendance_within_clip_bounds():
    """Generator clips attendance to [40, 100]."""
    df = load_df()
    assert df["attendance_pct"].min() >= 40, "attendance below clip floor"
    assert df["attendance_pct"].max() <= 100, "attendance above clip ceiling"


def test_section_sizes_roughly_equal():
    """Design: p=[0.34, 0.33, 0.33] — no section should deviate by more than 15%."""
    df = load_df()
    counts = df["class_section"].value_counts()
    total = len(df)
    for sec in ["A", "B", "C"]:
        frac = counts[sec] / total
        assert 0.28 < frac < 0.40, f"section {sec} fraction {frac:.2f} outside design range"


def test_collision_broken_file_is_smaller_or_different_layout():
    """The broken version has overlapping elements; the fixed version has more whitespace.
    At minimum they must differ in byte count (different rendering)."""
    broken = (ROOT / "chart_collision_broken.png").stat().st_size
    fixed = (ROOT / "chart_collision_fixed.png").stat().st_size
    assert broken != fixed, f"identical file sizes suggest no layout change: {broken} == {fixed}"


def test_arbitrary_color_chart_exists_and_differs_from_deliberate():
    """The two color-variant charts must be different files (not duplicated)."""
    arb = (ROOT / "chart_color_arbitrary.png").read_bytes()
    delib = (ROOT / "chart_color_deliberate.png").read_bytes()
    assert arb != delib, "arbitrary and deliberate color charts are identical"


def test_notebook_contains_finding_with_actual_number():
    """The notebook must contain a finding that cites a specific Pearson r value,
    not just a vague 'correlated' impression."""
    nb = json.loads((ROOT / "week6_day1_precision_lab.ipynb").read_text())
    all_md = " ".join(
        "".join(c.get("source", []))
        for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    has_r_value = any(tok in all_md for tok in ("r = 0.", "r =", "r ≈", "r="))
    assert has_r_value, "no specific r value found in findings"


def test_notebook_shuffle_test_appears():
    """The shuffle-test justification must appear in the notebook text."""
    nb = json.loads((ROOT / "week6_day1_precision_lab.ipynb").read_text())
    all_md = " ".join(
        "".join(c.get("source", []))
        for c in nb["cells"] if c["cell_type"] == "markdown"
    )
    assert "shuffle" in all_md.lower(), "shuffle-test justification missing from notebook"
