from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUT_DIR = DATA_DIR / "tableau_page4"

BASE_PATH = DATA_DIR / "skill_patch28_postings_base_page4_final.csv"
CAP_PATH = DATA_DIR / "skill_patch28_capabilities_long_page4_final.csv"
SUPPORT_PATH = DATA_DIR / "skill_patch28_capability_support_page4_final.csv"

# Core 1 intentionally has no pre-ranked serving table:
# Tableau should use SUPPORT_PATH directly and control Top N itself.

CORE2_PATH = OUT_DIR / "page4_core2_segment_stats_all.csv"
CORE3_PATH = OUT_DIR / "page4_core3_county_stats_all.csv"
CORE4_PATH = OUT_DIR / "page4_core4_cooccurrence_top100.csv"
SELECTOR_PATH = OUT_DIR / "page4_selected_capability_options.csv"
SEGMENT_PATH = OUT_DIR / "page4_search_segment_options_all.csv"
COUNTY_PATH = OUT_DIR / "page4_county_options_all.csv"
SUMMARY_PATH = OUT_DIR / "page4_tableau_flexible_inputs_summary.csv"

EXPECTED_POSTINGS = 64_283
EXPECTED_CAPABILITY_ROWS = 984_259
EXPECTED_UNIQUE_CAPABILITIES = 254_773

# This is an analytics/selector stability scope, not a visual Top-N rule.
SELECTOR_MIN_SUPPORT = 250
EXPECTED_SELECTOR_OPTIONS = 291

# Core 4 is the one place we pre-limit candidates for performance.
# Tableau can choose any user-facing Top N <= this ceiling.
CORE4_PRECOMPUTE_TOP_K = 100


def main() -> None:
    for path in (BASE_PATH, CAP_PATH, SUPPORT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    base = pd.read_csv(BASE_PATH, low_memory=False)
    caps = pd.read_csv(CAP_PATH, low_memory=False)
    support = pd.read_csv(SUPPORT_PATH, low_memory=False)

    if len(base) != EXPECTED_POSTINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_POSTINGS:,} postings, found {len(base):,}"
        )
    if base["record_id"].nunique() != EXPECTED_POSTINGS:
        raise RuntimeError("Posting record_id uniqueness failed")

    if len(caps) != EXPECTED_CAPABILITY_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_CAPABILITY_ROWS:,} capability rows, "
            f"found {len(caps):,}"
        )
    if caps["final_skill_key"].nunique() != EXPECTED_UNIQUE_CAPABILITIES:
        raise RuntimeError("Unexpected unique capability count")
    if caps.duplicated(["record_id", "final_skill_key"]).any():
        raise RuntimeError("Duplicate posting-capability rows found")

    display_map = (
        support[["final_skill_key", "skill_display"]]
        .drop_duplicates("final_skill_key")
        .set_index("final_skill_key")["skill_display"]
        .to_dict()
    )
    overall_count_map = (
        support.set_index("final_skill_key")["posting_count"].to_dict()
    )
    overall_prev_map = (
        support.set_index("final_skill_key")["overall_prevalence_pct"].to_dict()
    )

    # ------------------------------------------------------------
    # Shared Selected Capability options (Core 3 + Core 4)
    # ------------------------------------------------------------
    selector = (
        support.loc[support["posting_count"] >= SELECTOR_MIN_SUPPORT]
        .copy()
        .sort_values(
            ["posting_count", "skill_display"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if len(selector) != EXPECTED_SELECTOR_OPTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_SELECTOR_OPTIONS} selector options, "
            f"found {len(selector)}"
        )

    selector["selector_rank_by_support"] = range(1, len(selector) + 1)
    selector["selector_min_support_used_for_serving"] = SELECTOR_MIN_SUPPORT
    selector_keys = set(selector["final_skill_key"])

    # ------------------------------------------------------------
    # Core 2: ALL segment x capability statistics.
    # No Search-Segment minimum, no Top N, no view-mode filtering here.
    # Tableau will control all of those.
    # ------------------------------------------------------------
    base["search_query_title"] = (
        base["search_query_title"].fillna("").astype(str).str.strip()
    )

    segment_options = (
        base.loc[base["search_query_title"].ne("")]
        .groupby("search_query_title", as_index=False)
        .agg(segment_postings=("record_id", "nunique"))
        .sort_values(
            ["segment_postings", "search_query_title"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    cap_segment = caps.merge(
        base[["record_id", "search_query_title"]],
        on="record_id",
        how="left",
        validate="many_to_one",
    )

    core2 = (
        cap_segment.loc[cap_segment["search_query_title"].ne("")]
        .groupby(
            ["search_query_title", "final_skill_key", "skill_display"],
            as_index=False,
        )
        .agg(selected_skill_postings=("record_id", "nunique"))
    )

    core2 = core2.merge(
        segment_options,
        on="search_query_title",
        how="left",
        validate="many_to_one",
    )

    core2["overall_skill_postings"] = (
        core2["final_skill_key"].map(overall_count_map)
    )
    core2["rest_market_postings"] = (
        EXPECTED_POSTINGS - core2["segment_postings"]
    )
    core2["rest_skill_postings"] = (
        core2["overall_skill_postings"] - core2["selected_skill_postings"]
    )
    core2["selected_prevalence_pct"] = (
        core2["selected_skill_postings"] / core2["segment_postings"] * 100
    )
    core2["rest_prevalence_pct"] = (
        core2["rest_skill_postings"] / core2["rest_market_postings"] * 100
    )
    core2["prevalence_gap_pp"] = (
        core2["selected_prevalence_pct"] - core2["rest_prevalence_pct"]
    )

    core2 = core2[
        [
            "search_query_title",
            "segment_postings",
            "final_skill_key",
            "skill_display",
            "selected_skill_postings",
            "selected_prevalence_pct",
            "rest_market_postings",
            "rest_skill_postings",
            "rest_prevalence_pct",
            "prevalence_gap_pp",
            "overall_skill_postings",
        ]
    ].sort_values(
        ["search_query_title", "selected_prevalence_pct", "final_skill_key"],
        ascending=[True, False, True],
        kind="stable",
    )

    # ------------------------------------------------------------
    # Core 3: ALL 37 counties x 291 selector capabilities.
    # No county-size filter and no Top N here.
    # Tableau will control county eligibility / visible Top N.
    # ------------------------------------------------------------
    base["job_county"] = (
        base["job_county"].fillna("").astype(str).str.strip()
    )

    county_options = (
        base.loc[base["job_county"].ne("")]
        .groupby("job_county", as_index=False)
        .agg(county_postings=("record_id", "nunique"))
        .sort_values(
            ["county_postings", "job_county"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    all_county_postings = int(county_options["county_postings"].sum())

    base_county = base.loc[
        base["job_county"].ne(""),
        ["record_id", "job_county"],
    ]

    selector_occ = caps.loc[
        caps["final_skill_key"].isin(selector_keys),
        ["record_id", "final_skill_key"],
    ].merge(
        base_county,
        on="record_id",
        how="inner",
        validate="many_to_one",
    )

    county_skill_counts = (
        selector_occ.groupby(
            ["final_skill_key", "job_county"],
            as_index=False,
        )
        .agg(capability_postings=("record_id", "nunique"))
    )

    selector_small = selector[
        ["final_skill_key", "skill_display", "posting_count"]
    ].rename(columns={"posting_count": "overall_postings"})
    selector_small["_join_key"] = 1

    county_small = county_options.copy()
    county_small["_join_key"] = 1

    core3 = (
        selector_small.merge(county_small, on="_join_key")
        .drop(columns="_join_key")
        .merge(
            county_skill_counts,
            on=["final_skill_key", "job_county"],
            how="left",
            validate="one_to_one",
        )
    )

    core3["capability_postings"] = (
        core3["capability_postings"].fillna(0).astype(int)
    )
    core3["county_prevalence_pct"] = (
        core3["capability_postings"] / core3["county_postings"] * 100
    )

    # Store the full county-resolved benchmark. In Tableau, when a county-size
    # filter/parameter is applied, a matching benchmark can be calculated from
    # the filtered rows if desired.
    selector_county_totals = (
        selector_occ.groupby("final_skill_key", as_index=False)
        .agg(
            county_resolved_capability_postings=(
                "record_id",
                "nunique",
            )
        )
    )

    core3 = core3.merge(
        selector_county_totals,
        on="final_skill_key",
        how="left",
        validate="many_to_one",
    )

    core3["county_resolved_capability_postings"] = (
        core3["county_resolved_capability_postings"].fillna(0).astype(int)
    )
    core3["all_county_resolved_postings"] = all_county_postings
    core3["all_county_resolved_benchmark_pct"] = (
        core3["county_resolved_capability_postings"]
        / all_county_postings
        * 100
    )

    core3 = core3.sort_values(
        ["final_skill_key", "county_prevalence_pct", "county_postings"],
        ascending=[True, False, False],
        kind="stable",
    )

    # ------------------------------------------------------------
    # Core 4: precompute a generous Top 100 ceiling per selector capability.
    # Tableau controls the displayed Top N <= 100.
    # ------------------------------------------------------------
    posting_skills = (
        caps.groupby("record_id")["final_skill_key"]
        .agg(list)
    )

    pair_counts: dict[str, Counter] = defaultdict(Counter)

    for skills in posting_skills:
        selected_present = selector_keys.intersection(skills)
        if not selected_present:
            continue

        for selected_key in selected_present:
            counter = pair_counts[selected_key]
            for other_key in skills:
                if other_key != selected_key:
                    counter[other_key] += 1

    core4_rows = []

    for selected_row in selector.itertuples(index=False):
        selected_key = selected_row.final_skill_key
        selected_display = selected_row.skill_display
        selected_n = int(selected_row.posting_count)

        ranked = sorted(
            pair_counts.get(selected_key, Counter()).items(),
            key=lambda kv: (-kv[1], kv[0]),
        )[:CORE4_PRECOMPUTE_TOP_K]

        for precompute_rank, (other_key, co_postings) in enumerate(
            ranked,
            start=1,
        ):
            overall_prev = float(overall_prev_map.get(other_key, 0.0))

            core4_rows.append(
                {
                    "selected_capability_key": selected_key,
                    "selected_capability_display": selected_display,
                    "selected_capability_postings": selected_n,
                    "precompute_rank": precompute_rank,
                    "co_capability_key": other_key,
                    "co_capability_display": display_map.get(
                        other_key,
                        other_key,
                    ),
                    "co_postings": co_postings,
                    "cooccurrence_pct": co_postings / selected_n * 100,
                    "co_capability_overall_postings": int(
                        overall_count_map.get(other_key, 0)
                    ),
                    "co_capability_overall_prevalence_pct": overall_prev,
                    "excess_vs_overall_pp": (
                        co_postings / selected_n * 100 - overall_prev
                    ),
                }
            )

    core4 = pd.DataFrame(core4_rows)

    if core4["selected_capability_key"].nunique() != EXPECTED_SELECTOR_OPTIONS:
        raise RuntimeError(
            "Not every selector capability received Core 4 rows"
        )

    # ------------------------------------------------------------
    # Write flexible Tableau-serving inputs.
    # ------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    core2.to_csv(CORE2_PATH, index=False)
    core3.to_csv(CORE3_PATH, index=False)
    core4.to_csv(CORE4_PATH, index=False)
    selector.to_csv(SELECTOR_PATH, index=False)
    segment_options.to_csv(SEGMENT_PATH, index=False)
    county_options.to_csv(COUNTY_PATH, index=False)

    summary_rows = [
        ("processed_postings", EXPECTED_POSTINGS),
        ("final_capability_rows", EXPECTED_CAPABILITY_ROWS),
        ("final_unique_capabilities", EXPECTED_UNIQUE_CAPABILITIES),
        (
            "core1_source",
            "skill_patch28_capability_support_page4_final.csv",
        ),
        (
            "core1_note",
            "No Top N is prefiltered. Tableau should control Top N.",
        ),
        ("core2_search_segments_all", len(segment_options)),
        ("core2_rows_all_segment_skill_pairs", len(core2)),
        (
            "core2_note",
            "No segment-size threshold, Top N, or Most Prevalent/Most Distinctive view is prefiltered. Tableau controls them.",
        ),
        ("selected_capability_min_support_serving_scope", SELECTOR_MIN_SUPPORT),
        ("selected_capability_options", len(selector)),
        ("core3_counties_all", len(county_options)),
        ("core3_rows_all_selector_x_county", len(core3)),
        (
            "core3_note",
            "All county-resolved counties are included. Tableau controls minimum county support and visible Top N.",
        ),
        ("core4_precompute_top_k_ceiling", CORE4_PRECOMPUTE_TOP_K),
        ("core4_rows", len(core4)),
        (
            "core4_note",
            "Pair counts are precomputed for performance, but Tableau controls displayed Top N up to the precomputed ceiling.",
        ),
        (
            "design_principle",
            "Use code for expensive/reproducible analytical preparation; keep presentation choices such as Top N, view mode, and eligibility display filters in Tableau whenever practical.",
        ),
    ]

    pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    ).to_csv(SUMMARY_PATH, index=False)

    print("Flexible Page 4 Tableau input build complete.")
    print()
    print("Core 1:")
    print("  Use full skill_patch28_capability_support_page4_final.csv")
    print("  Tableau controls Top N")
    print()
    print(f"Core 2 all Search Segments:       {len(segment_options):,}")
    print(f"Core 2 all segment-skill rows:    {len(core2):,}")
    print(f"Selected Capability options:      {len(selector):,}")
    print(f"Core 3 all counties:              {len(county_options):,}")
    print(f"Core 3 selector x county rows:    {len(core3):,}")
    print(f"Core 4 precompute ceiling:        Top {CORE4_PRECOMPUTE_TOP_K}")
    print(f"Core 4 rows:                      {len(core4):,}")
    print()
    print("Wrote:")
    print(f"  {CORE2_PATH}")
    print(f"  {CORE3_PATH}")
    print(f"  {CORE4_PATH}")
    print(f"  {SELECTOR_PATH}")
    print(f"  {SEGMENT_PATH}")
    print(f"  {COUNTY_PATH}")
    print(f"  {SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: The earlier Top-15/Top-10 serving files are QA previews only. "
        "Do not connect those prefiltered files to the final Tableau dashboard."
    )


if __name__ == "__main__":
    main()
