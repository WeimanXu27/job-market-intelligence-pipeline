"""Freeze an untouched replacement blind holdout after the first blind failure.

This script is READ-ONLY with respect to MySQL and does NOT call OpenAI.

Goal
----
Create a replacement 288-record blind holdout after the original v5 blind failed.
The original 288 IDs are now exposed and are excluded here. This replacement
holdout must remain untouched while v6 is developed only on exposed cases.

Holdout composition
-------------------
Stress set (144):
  - 48 untouched extraction-zero stress records
  - 24 untouched role-screen-zero stress records
  - 72 untouched low-count/rich-description records

General controls (144):
  - 36 records from each current Patch-28 skill-count band:
      1-10, 11-20, 21-40, 41-60

Exposure safeguards
-------------------
The script excludes:
  1. All 935 records already completed before the production run
     (`already_processed == True` in the exact 64,283 manifest). This covers
     the earlier exposed regression + fresh final QA sets.
  2. Every record in the 144-record final zero priority-review sample.
  3. Every record in the 12-record targeted full-run zero diagnostic manifest.
  4. Every record in the top-200 zero-candidate export.
  5. The known manually inspected max-count example job_004941.
  6. All 288 IDs from the failed Patch-29 v5 blind holdout.

The output manifest deliberately contains NO description text and NO current
extracted_skills. Do not open descriptions/results for these IDs until the
Patch-29 v6 candidate has been frozen.

Outputs
-------
data/processed/skill_patch29_v6_blind_holdout_manifest.csv
data/processed/skill_patch29_v6_blind_holdout_summary.csv
data/processed/skill_patch29_v6_blind_holdout_excluded_ids.csv
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import URL, bindparam, create_engine, text

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data" / "processed"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402

SEED = 20260907
EXPECTED_USABLE = 64_283
EXPECTED_PRECOMPLETED = 935
EXPECTED_HOLDOUT = 288
EXPECTED_EXPOSED_BEFORE_V6 = 1_509

FULL_MANIFEST = DATA_DIR / "skill_full_run_manifest_64283.csv"
ZERO_CAUSE = DATA_DIR / "skill_full_run_zero_with_log_cause.csv"
ZERO_REVIEW = DATA_DIR / "skill_full_run_zero_priority_review_sample.csv"
ZERO_TOP200 = DATA_DIR / "skill_full_run_zero_candidates_top200.csv"
TARGETED_MANIFEST = DATA_DIR / "skill_patch28_fullrun_zero_stage_manifest.csv"
OLD_FAILED_BLIND = DATA_DIR / "skill_patch29_blind_holdout_manifest.csv"

OUT_MANIFEST = DATA_DIR / "skill_patch29_v6_blind_holdout_manifest.csv"
OUT_SUMMARY = DATA_DIR / "skill_patch29_v6_blind_holdout_summary.csv"
OUT_EXCLUDED = DATA_DIR / "skill_patch29_v6_blind_holdout_excluded_ids.csv"

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]

RICHNESS_MARKERS = [
    "requirements",
    "qualifications",
    "responsibilities",
    "required qualifications",
    "preferred qualifications",
    "minimum qualifications",
    "experience",
    "skills",
    "proficiency",
    "proficient",
    "knowledge of",
    "ability to",
    "must have",
    "you will",
    "what you'll do",
    "what you will do",
    "what we're looking for",
    "what we are looking for",
]

MANUAL_EXPOSED_IDS = {
    "job_004941",  # max-count example discussed during final QA
}


def build_engine():
    db = DATABASE_CONFIG
    url = URL.create(
        drivername="mysql+mysqlconnector",
        username=db["username"],
        password=db["password"],
        host=db["host"],
        port=int(db["port"]),
        database=db["database"],
        query={"charset": db.get("charset", "utf8mb4")},
    )
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )


def parse_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    normalized = series.astype(str).str.strip().str.lower()
    bad = sorted(set(normalized) - set(mapping))
    if bad:
        raise ValueError(
            "Unexpected already_processed values: " + ", ".join(bad[:10])
        )
    return normalized.map(mapping)


def read_ids_if_exists(path: Path) -> set[str]:
    if not path.exists():
        return set()

    df = pd.read_csv(
        path,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    if "record_id" not in df.columns:
        raise ValueError(f"{path.name} has no record_id column.")

    ids = df["record_id"].astype(str).str.strip()
    return set(ids[ids.ne("")])


def count_skills(value) -> int:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0

    text_value = str(value).strip()
    if not text_value:
        return 0

    lines = [line.strip() for line in text_value.splitlines() if line.strip()]
    numbered = [
        line
        for line in lines
        if re.match(r"^\s*\d+[\.\)]\s*\S+", line)
    ]
    return len(numbered) if numbered else len(lines)


def description_metrics(value) -> tuple[int, int, int]:
    text_value = "" if value is None else str(value)
    lowered = text_value.lower()
    char_count = len(text_value)
    word_count = len(re.findall(r"\b[\w+#./-]+\b", text_value))
    marker_count = sum(1 for marker in RICHNESS_MARKERS if marker in lowered)
    return char_count, word_count, marker_count


def fetch_rows(engine, ids: list[str]) -> pd.DataFrame:
    query = text(
        f"SELECT {ID_COLUMN} AS record_id, "
        f"{DESCRIPTION_COLUMN} AS description, "
        f"{SKILLS_COLUMN} AS extracted_skills "
        f"FROM {TABLE_NAME} "
        f"WHERE {ID_COLUMN} IN :record_ids"
    ).bindparams(bindparam("record_ids", expanding=True))

    frames = []
    chunk_size = 800

    with engine.connect() as conn:
        for start in range(0, len(ids), chunk_size):
            chunk = ids[start:start + chunk_size]
            rows = conn.execute(
                query, {"record_ids": chunk}
            ).mappings().all()
            if rows:
                frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame(
            columns=["record_id", "description", "extracted_skills"]
        )

    result = pd.concat(frames, ignore_index=True)
    result["record_id"] = result["record_id"].astype(str).str.strip()
    return result


def deterministic_sample(
    frame: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(frame) < n:
        raise RuntimeError(
            f"Not enough eligible rows: need {n}, found {len(frame)}."
        )

    idx = rng.choice(frame.index.to_numpy(), size=n, replace=False)
    return frame.loc[idx].copy()


def sha256_csv(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    if not FULL_MANIFEST.exists():
        raise FileNotFoundError(f"Missing {FULL_MANIFEST}")
    if not ZERO_CAUSE.exists():
        raise FileNotFoundError(
            f"Missing {ZERO_CAUSE}; run zero-cause audit first."
        )

    manifest = pd.read_csv(
        FULL_MANIFEST,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )

    required = {"record_id", "run_order", "pool", "already_processed"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(
            f"Full manifest missing columns: {sorted(missing)}"
        )

    manifest["record_id"] = manifest["record_id"].astype(str).str.strip()
    manifest["run_order"] = pd.to_numeric(
        manifest["run_order"], errors="raise"
    ).astype(int)
    manifest["already_processed_bool"] = parse_bool(
        manifest["already_processed"]
    )

    if len(manifest) != EXPECTED_USABLE:
        raise RuntimeError(
            f"Expected {EXPECTED_USABLE:,} usable rows; found {len(manifest):,}."
        )
    if manifest["record_id"].nunique() != EXPECTED_USABLE:
        raise RuntimeError("Usable manifest IDs are not unique.")

    precompleted_ids = set(
        manifest.loc[
            manifest["already_processed_bool"], "record_id"
        ]
    )
    if len(precompleted_ids) != EXPECTED_PRECOMPLETED:
        raise RuntimeError(
            f"Expected {EXPECTED_PRECOMPLETED:,} precompleted/exposed IDs; "
            f"found {len(precompleted_ids):,}."
        )

    failed_v5_blind_ids = read_ids_if_exists(OLD_FAILED_BLIND)
    if len(failed_v5_blind_ids) != 288:
        raise RuntimeError(
            "Expected exactly 288 IDs in the failed v5 blind manifest; "
            f"found {len(failed_v5_blind_ids):,}."
        )

    exposure_sources = {
        "preproduction_935": precompleted_ids,
        "zero_priority_review_144": read_ids_if_exists(ZERO_REVIEW),
        "zero_top200_export": read_ids_if_exists(ZERO_TOP200),
        "targeted_zero_diagnostic_12": read_ids_if_exists(TARGETED_MANIFEST),
        "manual_inspection": MANUAL_EXPOSED_IDS,
        "failed_v5_blind_288": failed_v5_blind_ids,
    }

    exposed_union = set().union(*exposure_sources.values())

    if len(exposed_union) != EXPECTED_EXPOSED_BEFORE_V6:
        raise RuntimeError(
            f"Expected {EXPECTED_EXPOSED_BEFORE_V6:,} exposed IDs before v6; "
            f"found {len(exposed_union):,}."
        )

    exclusion_rows = []
    for reason, ids in exposure_sources.items():
        for rid in sorted(ids):
            exclusion_rows.append(
                {"record_id": rid, "exposure_reason": reason}
            )

    pd.DataFrame(exclusion_rows).to_csv(
        OUT_EXCLUDED,
        index=False,
        encoding="utf-8-sig",
    )

    zero_cause = pd.read_csv(
        ZERO_CAUSE,
        dtype={"record_id": str},
        keep_default_na=False,
        low_memory=False,
    )
    zero_cause["record_id"] = zero_cause["record_id"].astype(str).str.strip()

    zero_required = {
        "record_id",
        "final_log_cause",
        "rich_source_candidate",
        "richness_score",
    }
    missing = zero_required - set(zero_cause.columns)
    if missing:
        raise ValueError(
            f"Zero-cause file missing columns: {sorted(missing)}"
        )

    if zero_cause["rich_source_candidate"].dtype != bool:
        zero_cause["rich_source_candidate"] = (
            zero_cause["rich_source_candidate"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": True, "false": False})
        )

    # Exclude everything we consider exposed before candidate development.
    eligible_manifest = manifest.loc[
        ~manifest["record_id"].isin(exposed_union)
    ].copy()

    # Read DB only for automated selection metrics. No descriptions or current
    # skills will be written to the holdout manifest.
    engine = build_engine()
    try:
        db = fetch_rows(
            engine,
            eligible_manifest["record_id"].tolist(),
        )
    finally:
        engine.dispose()

    db = db.drop_duplicates("record_id", keep="first")

    eligible = eligible_manifest.merge(
        db,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    if eligible["description"].isna().any():
        raise RuntimeError(
            "Some eligible holdout candidates are missing from MySQL."
        )

    eligible["skill_count"] = eligible["extracted_skills"].map(count_skills)
    desc_metrics = eligible["description"].map(description_metrics)
    eligible["description_chars"] = desc_metrics.map(lambda x: x[0])
    eligible["description_words"] = desc_metrics.map(lambda x: x[1])
    eligible["richness_marker_count"] = desc_metrics.map(lambda x: x[2])

    zero_meta = zero_cause[
        [
            "record_id",
            "final_log_cause",
            "rich_source_candidate",
            "richness_score",
        ]
    ].copy()

    eligible = eligible.merge(
        zero_meta,
        on="record_id",
        how="left",
        validate="one_to_one",
    )

    # ------------------------------------------------------------------
    # Stress 1: untouched extraction-returned-zero records
    # Prefer richer candidates but do not rely only on the previously exported
    # top-200 because those IDs are explicitly excluded above.
    # ------------------------------------------------------------------
    extraction_zero_pool = eligible.loc[
        eligible["skill_count"].eq(0)
        & eligible["final_log_cause"].eq("extraction_returned_zero")
    ].copy()

    extraction_zero_pool["richness_score_numeric"] = pd.to_numeric(
        extraction_zero_pool["richness_score"],
        errors="coerce",
    ).fillna(0)

    extraction_zero_pool = extraction_zero_pool.sort_values(
        [
            "richness_score_numeric",
            "richness_marker_count",
            "description_chars",
        ],
        ascending=False,
    )

    # Take from the high-risk half of the still-untouched pool, but randomize
    # within that high-risk region so the exact records are not hand-picked.
    top_region_n = max(48, len(extraction_zero_pool) // 2)
    extraction_zero_region = extraction_zero_pool.head(top_region_n)
    stress_extraction_zero = deterministic_sample(
        extraction_zero_region,
        48,
        rng,
    )
    stress_extraction_zero["blind_group"] = "stress_extraction_zero"

    selected_ids = set(stress_extraction_zero["record_id"])

    # ------------------------------------------------------------------
    # Stress 2: untouched role-screen-zero records
    # ------------------------------------------------------------------
    role_zero_pool = eligible.loc[
        eligible["skill_count"].eq(0)
        & eligible["final_log_cause"].eq("role_screened_non_substantive")
        & ~eligible["record_id"].isin(selected_ids)
    ].copy()

    stress_role_zero = deterministic_sample(
        role_zero_pool,
        24,
        rng,
    )
    stress_role_zero["blind_group"] = "stress_role_screen_zero"
    selected_ids.update(stress_role_zero["record_id"])

    # ------------------------------------------------------------------
    # Stress 3: low-count + rich-description non-zero records
    # This is specifically meant to catch under-recall that a zero-only audit
    # cannot reveal.
    # ------------------------------------------------------------------
    low_rich_pool = eligible.loc[
        eligible["skill_count"].between(1, 5)
        & eligible["description_chars"].ge(1500)
        & eligible["richness_marker_count"].ge(2)
        & ~eligible["record_id"].isin(selected_ids)
    ].copy()

    stress_low_rich = deterministic_sample(
        low_rich_pool,
        72,
        rng,
    )
    stress_low_rich["blind_group"] = "stress_low_count_rich"
    selected_ids.update(stress_low_rich["record_id"])

    # ------------------------------------------------------------------
    # 144 general controls across the non-zero output distribution.
    # ------------------------------------------------------------------
    control_specs = [
        ("control_count_01_10", 1, 10, 36),
        ("control_count_11_20", 11, 20, 36),
        ("control_count_21_40", 21, 40, 36),
        ("control_count_41_60", 41, 60, 36),
    ]

    control_parts = []

    for group, low, high, target in control_specs:
        pool = eligible.loc[
            eligible["skill_count"].between(low, high)
            & ~eligible["record_id"].isin(selected_ids)
        ].copy()

        chosen = deterministic_sample(pool, target, rng)
        chosen["blind_group"] = group
        control_parts.append(chosen)
        selected_ids.update(chosen["record_id"])

    holdout = pd.concat(
        [
            stress_extraction_zero,
            stress_role_zero,
            stress_low_rich,
            *control_parts,
        ],
        ignore_index=True,
    )

    if len(holdout) != EXPECTED_HOLDOUT:
        raise RuntimeError(
            f"Expected {EXPECTED_HOLDOUT} holdout rows; got {len(holdout)}."
        )
    if holdout["record_id"].nunique() != EXPECTED_HOLDOUT:
        raise RuntimeError("Holdout IDs are not unique.")
    if set(holdout["record_id"]) & exposed_union:
        raise RuntimeError("Blind holdout overlaps exposed IDs.")

    # Deterministic random review order. Still no content/results in output.
    holdout = holdout.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    holdout["blind_order"] = np.arange(1, len(holdout) + 1)

    public_cols = [
        "blind_order",
        "record_id",
        "blind_group",
        "pool",
    ]

    holdout[public_cols].to_csv(
        OUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    manifest_sha = sha256_csv(OUT_MANIFEST)

    group_counts = (
        holdout["blind_group"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    summary_rows = [
        ("selection_seed", SEED),
        ("usable_manifest_rows", len(manifest)),
        ("exposed_union_unique_ids", len(exposed_union)),
        ("eligible_after_exposure_exclusion", len(eligible)),
        ("blind_holdout_rows", len(holdout)),
        ("blind_holdout_unique_ids", holdout["record_id"].nunique()),
        ("exposed_overlap", len(set(holdout["record_id"]) & exposed_union)),
        ("failed_v5_blind_overlap", len(set(holdout["record_id"]) & failed_v5_blind_ids)),
        ("manifest_sha256", manifest_sha),
        (
            "blind_rule",
            "Do not inspect descriptions or any extractor outputs for these IDs until the Patch-29 v6 candidate is frozen.",
        ),
    ]

    for group, count in group_counts.items():
        summary_rows.append((f"group_{group}", int(count)))

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )
    summary.to_csv(
        OUT_SUMMARY,
        index=False,
        encoding="utf-8-sig",
    )

    print("Patch-29 v6 replacement blind holdout freeze complete.")
    print()
    print(f"Usable universe: {len(manifest):,}")
    print(f"Excluded exposed IDs: {len(exposed_union):,}")
    print(f"Eligible untouched IDs: {len(eligible):,}")
    print(f"Blind holdout: {len(holdout):,} unique IDs")
    print(f"Exposure overlap: 0")
    print()
    print("Holdout composition:")
    for group, count in group_counts.items():
        print(f"  {group}: {count}")
    print()
    print(f"Manifest SHA-256: {manifest_sha}")
    print()
    print(f"Manifest: {OUT_MANIFEST}")
    print(f"Summary: {OUT_SUMMARY}")
    print(f"Exclusion audit: {OUT_EXCLUDED}")
    print()
    print("BLINDNESS RULE:")
    print(
        "Do NOT inspect descriptions or extractor outputs for these "
        "288 IDs until the Patch-29 v6 candidate is frozen."
    )
    print()
    print("No MySQL rows were modified. No OpenAI API calls were made.")


if __name__ == "__main__":
    main()
