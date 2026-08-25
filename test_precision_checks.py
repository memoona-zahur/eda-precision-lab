"""Own verification suite — Week 06 Day 1 EDA Precision Lab.

Beyond the kata requirements: these checks pin the dataset's designed truths,
the generator's determinism, the saved charts' integrity, and — the check this
lab exists to invent — that the shipped notebook actually contains a PASSING
self-audit table rather than just claiming one.
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


def load_df():
    return pd.read_csv(ROOT / "students.csv")


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
