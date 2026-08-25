"""Week 06 Day 1 - EDA Precision Lab dataset generator.

Generation code below is copied VERBATIM from the assignment spec
(week6-day1.md, Part A). The spec forbids modifying it: the relationships
inside are known in advance, which is what makes today's self-audit possible.
The only additions are the save call and the fingerprint print at the end -
no parameter, order, or expression has been touched.
"""
import hashlib

import numpy as np
import pandas as pd

# ---- BEGIN VERBATIM SPEC CODE ---------------------------------------------
rng = np.random.default_rng(seed=21)
n = 600

class_section = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33])
study_hours = rng.normal(10, 3.5, size=n).clip(0, None).round(1)
sleep_hours = rng.normal(7, 1.2, size=n).clip(3, 10).round(1)
attendance_pct = rng.normal(85, 10, size=n).clip(40, 100).round(1)

noise = rng.normal(0, 8, size=n)
section_bonus = pd.Series(class_section).map({"A": 0, "B": 0, "C": 4}).values
exam_score = (50 + 2.6 * study_hours + 0.15 * attendance_pct + section_bonus + noise).clip(0, 100).round(1)

students = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "class_section": class_section,
    "study_hours_per_week": study_hours,
    "sleep_hours_per_night": sleep_hours,
    "attendance_pct": attendance_pct,
    "exam_score": exam_score,
})
# ---- END VERBATIM SPEC CODE -----------------------------------------------

students.to_csv("students.csv", index=False)

sha = hashlib.sha256(open("students.csv", "rb").read()).hexdigest()
print(f"students.csv saved | shape {students.shape}")
print(f"sha256 = {sha[:16]}…{sha[-8:]}")
