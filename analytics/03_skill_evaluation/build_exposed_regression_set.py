"""Build the complete exposed regression manifest for skill extraction.

This manifest contains every usable record that has already been inspected or
used for tuning/evaluation:
- 359 exposed development records
- 360 exposed final-QA records
- 719 total usable exposed records

The script does NOT include the 11-record diagnostic manifest separately,
because those 11 IDs are already contained in the exposed samples above.

It validates every ID against the corrected permanent development/final pools,
which automatically excludes source-ineligible records such as job_064682.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/processed")

DEVELOPMENT_POOL = DATA_DIR / "skill_development_pool.csv"
FINAL_TEST_POOL = DATA_DIR / "skill_final_test_pool.csv"
OUTPUT_PATH = DATA_DIR / "skill_exposed_regression_719.csv"

DEVELOPMENT_MANIFESTS = [
    DATA_DIR / "skill_pilot_sample.csv",
    DATA_DIR / "skill_holdout_batch_b.csv",
    DATA_DIR / "skill_dev_validation_72.csv",
    DATA_DIR / "skill_dev_validation_72_v2.csv",
    DATA_DIR / "skill_dev_validation_72_v3.csv",
    DATA_DIR / "skill_dev_validation_72_v4.csv",
]

FINAL_MANIFESTS = [
    DATA_DIR / "skill_final_validation_144.csv",
    DATA_DIR / "skill_final_validation_72_v2.csv",
    DATA_DIR / "skill_final_validation_144_v3.csv",
]

EXPECTED_DEVELOPMENT_COUNT = 359
EXPECTED_FINAL_COUNT = 360
EXPECTED_TOTAL_COUNT = 719
KNOWN_INELIGIBLE_ID = "job_064682"


def read_ids(path: Path) -> list[str]:
    """Read and validate record IDs from one manifest."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required manifest not found: {path}"
        )

    frame = pd.read_csv(
        path,
        usecols=["record_id"],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )

    frame["record_id"] = (
        frame["record_id"]
        .astype(str)
        .str.strip()
    )

    if frame["record_id"].eq("").any():
        raise ValueError(
            f"Blank record_id found in {path}"
        )

    if not frame["record_id"].is_unique:
        raise ValueError(
            f"Duplicate record_id values found inside {path}"
        )

    return frame["record_id"].tolist()


def read_pool_ids(path: Path) -> set[str]:
    """Read permanent pool IDs."""
    return set(
        read_ids(path)
    )


def collect_group(
    manifests: list[Path],
    allowed_ids: set[str],
    group_name: str,
) -> pd.DataFrame:
    """Collect exposed IDs from one permanent pool."""
    rows: list[dict[str, str]] = []

    for manifest in manifests:
        for record_id in read_ids(manifest):
            if record_id not in allowed_ids:
                # Source-ineligible rows from historical exposed samples are
                # intentionally excluded from usable regression.
                continue

            rows.append(
                {
                    "record_id": record_id,
                    "source_group": group_name,
                    "source_manifest": manifest.name,
                }
            )

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise RuntimeError(
            f"No usable IDs collected for {group_name}."
        )

    duplicate_mask = frame["record_id"].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_ids = sorted(
            frame.loc[
                duplicate_mask,
                "record_id",
            ].unique()
        )

        raise RuntimeError(
            f"Unexpected duplicate exposed IDs in {group_name}: "
            + ", ".join(duplicate_ids[:10])
        )

    return frame


def main() -> None:
    """Build and validate the 719-record exposed regression manifest."""
    for required_path in [
        DEVELOPMENT_POOL,
        FINAL_TEST_POOL,
        *DEVELOPMENT_MANIFESTS,
        *FINAL_MANIFESTS,
    ]:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required file not found: {required_path}"
            )

    development_pool_ids = read_pool_ids(
        DEVELOPMENT_POOL
    )
    final_pool_ids = read_pool_ids(
        FINAL_TEST_POOL
    )

    pool_overlap = (
        development_pool_ids
        & final_pool_ids
    )

    if pool_overlap:
        raise RuntimeError(
            "Permanent development/final pools overlap."
        )

    development = collect_group(
        DEVELOPMENT_MANIFESTS,
        development_pool_ids,
        "development",
    )

    final_qa = collect_group(
        FINAL_MANIFESTS,
        final_pool_ids,
        "final_qa",
    )

    if len(development) != EXPECTED_DEVELOPMENT_COUNT:
        raise RuntimeError(
            "Unexpected usable exposed development count: "
            f"expected {EXPECTED_DEVELOPMENT_COUNT}, "
            f"found {len(development)}."
        )

    if len(final_qa) != EXPECTED_FINAL_COUNT:
        raise RuntimeError(
            "Unexpected exposed final-QA count: "
            f"expected {EXPECTED_FINAL_COUNT}, "
            f"found {len(final_qa)}."
        )

    regression = pd.concat(
        [
            development,
            final_qa,
        ],
        ignore_index=True,
    )

    if len(regression) != EXPECTED_TOTAL_COUNT:
        raise RuntimeError(
            "Unexpected total exposed regression count: "
            f"expected {EXPECTED_TOTAL_COUNT}, "
            f"found {len(regression)}."
        )

    if not regression["record_id"].is_unique:
        raise RuntimeError(
            "Exposed regression record IDs are not unique."
        )

    if (
        KNOWN_INELIGIBLE_ID
        in set(regression["record_id"])
    ):
        raise RuntimeError(
            f"Source-ineligible ID {KNOWN_INELIGIBLE_ID} "
            "entered the usable regression manifest."
        )

    regression.insert(
        0,
        "pilot_order",
        range(
            1,
            len(regression) + 1,
        ),
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    regression.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "Complete exposed regression manifest built."
    )
    print(
        f"Development rows: {len(development):,}"
    )
    print(
        f"Final-QA rows: {len(final_qa):,}"
    )
    print(
        f"Total rows: {len(regression):,}"
    )
    print(
        "Unique record IDs: "
        f"{regression['record_id'].nunique():,}"
    )
    print(
        f"{KNOWN_INELIGIBLE_ID} included: "
        f"{KNOWN_INELIGIBLE_ID in set(regression['record_id'])}"
    )
    print(
        f"Output: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
