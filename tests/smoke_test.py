"""Structural checks that do not require participant-level data."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "BRAINAGE_OUTPUT_DIR",
    str(Path(tempfile.gettempdir()) / "brainage_multiscale_smoke_results"),
)
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import brain_age_multiscale as pipeline  # noqa: E402


def main() -> None:
    expected_counts = {
        "WholeBrain_All5": 5,
        "Hemisphere_All5": 10,
        "Lobe_All5": 30,
        "ROI_All5": 340,
        "Multiscale_All5": 385,
        "WholeBrain_NoFD": 4,
        "Hemisphere_NoFD": 8,
        "Lobe_NoFD": 24,
        "ROI_NoFD": 272,
        "Multiscale_NoFD": 308,
        "ROI_volume": 68,
        "ROI_area": 68,
        "ROI_thickness": 68,
        "ROI_curvature": 68,
        "ROI_FD": 68,
    }

    assert pipeline.__version__ == "1.0.0"
    assert pipeline.OUTER_FOLDS == 5
    assert pipeline.N_REPEATS == 5
    assert pipeline.INNER_FOLDS == 5
    assert pipeline.N_BOOTSTRAP == 2000
    assert pipeline.N_PAIRED_PERM == 5000
    assert set(pipeline.FEATURE_SETS) == set(expected_counts)

    for name, expected in expected_counts.items():
        columns = pipeline.FEATURE_SETS[name]
        assert len(columns) == expected, (name, len(columns), expected)
        assert len(columns) == len(set(columns)), f"Duplicate feature in {name}"

    adjusted = pipeline.bh_fdr([0.01, 0.04, 0.03, 0.002])
    assert len(adjusted) == 4
    assert all(0.0 <= value <= 1.0 for value in adjusted)

    print("Smoke test passed: release structure and analysis constants are valid.")


if __name__ == "__main__":
    main()
