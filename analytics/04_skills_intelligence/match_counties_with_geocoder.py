from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

PREVIEW_PATH = DATA_DIR / "skill_patch28_ca_county_official_match_preview.csv"
UNRESOLVED_PATH = DATA_DIR / "skill_patch28_ca_county_unresolved_places.csv"

OUTPUT_COMBINED_PATH = DATA_DIR / "skill_patch28_ca_county_geocode_match_preview.csv"
OUTPUT_UNRESOLVED_PATH = DATA_DIR / "skill_patch28_ca_county_geocode_unresolved.csv"
OUTPUT_SUMMARY_PATH = DATA_DIR / "skill_patch28_ca_county_geocode_summary.csv"

GEOCODER_URL = (
    "https://geocode.arcgis.com/arcgis/rest/services/"
    "World/GeocodeServer/findAddressCandidates"
)

EXPECTED_ALL_PLACES = 346
EXPECTED_ALL_POSTINGS = 5_098
EXPECTED_FIRST_PASS_UNRESOLVED_PLACES = 116
EXPECTED_FIRST_PASS_UNRESOLVED_POSTINGS = 1_222

MIN_AUTO_SCORE = 95.0
AMBIGUOUS_SCORE_WINDOW = 3.0


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.casefold()
    text = text.replace("&", " and ")
    text = re.sub(r"[’'`]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_county(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+County$", "", text, flags=re.IGNORECASE)
    return text


def fetch_json(params: dict, retries: int = 3) -> dict:
    request = Request(
        f"{GEOCODER_URL}?{urlencode(params)}",
        headers={"User-Agent": "AllSumJobs-Portfolio-County-Audit/1.0"},
    )

    last_error = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if "error" in payload:
                raise RuntimeError(str(payload["error"]))
            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError("ArcGIS geocoder request failed") from last_error


def candidate_rows(place: str) -> list[dict]:
    payload = fetch_json(
        {
            "SingleLine": f"{place}, CA",
            "f": "json",
            "maxLocations": 5,
            "forStorage": "false",
            "outFields": (
                "Match_addr,Addr_type,City,Neighborhood,"
                "Subregion,Region,Postal,Country"
            ),
            "countryCode": "USA",
        }
    )

    rows = []

    for candidate in payload.get("candidates", []):
        attrs = candidate.get("attributes", {})
        score = float(candidate.get("score", 0) or 0)
        region = str(attrs.get("Region") or "").strip()
        country = str(attrs.get("Country") or "").strip()
        county = clean_county(attrs.get("Subregion"))

        if region not in {"CA", "California"}:
            continue
        if country not in {"USA", "US", "United States"}:
            continue
        if not county:
            continue

        rows.append(
            {
                "score": score,
                "county": county,
                "match_addr": attrs.get("Match_addr") or "",
                "addr_type": attrs.get("Addr_type") or "",
                "city": attrs.get("City") or "",
                "neighborhood": attrs.get("Neighborhood") or "",
                "postal": attrs.get("Postal") or "",
            }
        )

    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def resolve_place(place: str) -> dict:
    candidates = candidate_rows(place)

    result = {
        "geocode_status": "unresolved",
        "geocode_county": "",
        "geocode_score": "",
        "geocode_match_addr": "",
        "geocode_addr_type": "",
        "geocode_review_note": "",
    }

    if not candidates:
        result["geocode_review_note"] = "No California candidate with county returned"
        return result

    top = candidates[0]

    result.update(
        {
            "geocode_county": top["county"],
            "geocode_score": top["score"],
            "geocode_match_addr": top["match_addr"],
            "geocode_addr_type": top["addr_type"],
        }
    )

    if top["score"] < MIN_AUTO_SCORE:
        result["geocode_status"] = "manual_review"
        result["geocode_review_note"] = (
            f"Top score below {MIN_AUTO_SCORE:.0f}"
        )
        return result

    close_other_counties = sorted(
        {
            row["county"]
            for row in candidates[1:]
            if row["county"] != top["county"]
            and row["score"] >= top["score"] - AMBIGUOUS_SCORE_WINDOW
        }
    )

    if close_other_counties:
        result["geocode_status"] = "manual_review"
        result["geocode_review_note"] = (
            "Competing high-score county candidates: "
            + " | ".join(close_other_counties)
        )
        return result

    result["geocode_status"] = "auto_matched"
    result["geocode_review_note"] = (
        "High-confidence ArcGIS geocoder result; "
        "no close competing county candidate"
    )
    return result


def main() -> None:
    for path in (PREVIEW_PATH, UNRESOLVED_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    first_pass = pd.read_csv(PREVIEW_PATH)
    unresolved = pd.read_csv(UNRESOLVED_PATH)

    if len(first_pass) != EXPECTED_ALL_PLACES:
        raise RuntimeError(
            f"Expected {EXPECTED_ALL_PLACES} first-pass places, found {len(first_pass)}"
        )

    if int(first_pass["posting_count"].sum()) != EXPECTED_ALL_POSTINGS:
        raise RuntimeError("First-pass posting total mismatch")

    if len(unresolved) != EXPECTED_FIRST_PASS_UNRESOLVED_PLACES:
        raise RuntimeError(
            "Unexpected first-pass unresolved place count: "
            f"{len(unresolved)}"
        )

    if int(unresolved["posting_count"].sum()) != EXPECTED_FIRST_PASS_UNRESOLVED_POSTINGS:
        raise RuntimeError("First-pass unresolved posting total mismatch")

    geocoded_rows = []

    print("Running conservative second-pass geocoding...")
    print(
        f"Auto-accept threshold: score >= {MIN_AUTO_SCORE:.0f}; "
        f"reject if another county is within {AMBIGUOUS_SCORE_WINDOW:.0f} score points"
    )
    print()

    for i, row in enumerate(unresolved.itertuples(index=False), start=1):
        place = str(row.city).strip()
        resolution = resolve_place(place)

        geocoded_rows.append(
            {
                **row._asdict(),
                **resolution,
            }
        )

        if i % 20 == 0 or i == len(unresolved):
            print(f"Processed {i:,} / {len(unresolved):,} places")

        time.sleep(0.08)

    second_pass = pd.DataFrame(geocoded_rows)

    combined = first_pass.copy()

    second_map = second_pass.set_index("city")

    for idx, row in combined.iterrows():
        city = row["city"]

        if row["match_status"] == "auto_matched":
            combined.loc[idx, "final_mapped_county"] = row["mapped_county"]
            combined.loc[idx, "final_match_status"] = "auto_matched"
            combined.loc[idx, "final_match_method"] = row["match_method"]
            combined.loc[idx, "final_match_score"] = ""
            combined.loc[idx, "final_match_detail"] = row["official_place_name"]
            continue

        if city not in second_map.index:
            raise RuntimeError(f"Missing second-pass row for {city}")

        geo = second_map.loc[city]

        if geo["geocode_status"] == "auto_matched":
            combined.loc[idx, "final_mapped_county"] = geo["geocode_county"]
            combined.loc[idx, "final_match_status"] = "auto_matched"
            combined.loc[idx, "final_match_method"] = "arcgis_world_geocoder_high_confidence"
            combined.loc[idx, "final_match_score"] = geo["geocode_score"]
            combined.loc[idx, "final_match_detail"] = geo["geocode_match_addr"]
        else:
            combined.loc[idx, "final_mapped_county"] = ""
            combined.loc[idx, "final_match_status"] = geo["geocode_status"]
            combined.loc[idx, "final_match_method"] = "arcgis_world_geocoder_review"
            combined.loc[idx, "final_match_score"] = geo["geocode_score"]
            combined.loc[idx, "final_match_detail"] = (
                str(geo["geocode_match_addr"])
                + " | "
                + str(geo["geocode_review_note"])
            )

    remaining = combined.loc[
        combined["final_match_status"].ne("auto_matched")
    ].copy()

    matched_postings = int(
        combined.loc[
            combined["final_match_status"].eq("auto_matched"),
            "posting_count",
        ].sum()
    )

    second_pass_matched_postings = int(
        second_pass.loc[
            second_pass["geocode_status"].eq("auto_matched"),
            "posting_count",
        ].sum()
    )

    summary_rows = [
        ("input_ca_places", len(combined)),
        ("input_ca_postings", int(combined["posting_count"].sum())),
        (
            "first_pass_official_matched_places",
            int(first_pass["match_status"].eq("auto_matched").sum()),
        ),
        (
            "first_pass_official_matched_postings",
            int(
                first_pass.loc[
                    first_pass["match_status"].eq("auto_matched"),
                    "posting_count",
                ].sum()
            ),
        ),
        (
            "second_pass_geocoder_matched_places",
            int(second_pass["geocode_status"].eq("auto_matched").sum()),
        ),
        (
            "second_pass_geocoder_matched_postings",
            second_pass_matched_postings,
        ),
        (
            "combined_auto_matched_places",
            int(combined["final_match_status"].eq("auto_matched").sum()),
        ),
        ("combined_auto_matched_postings", matched_postings),
        (
            "combined_auto_matched_posting_pct",
            round(matched_postings / EXPECTED_ALL_POSTINGS * 100, 4),
        ),
        ("remaining_review_places", len(remaining)),
        (
            "remaining_review_postings",
            int(remaining["posting_count"].sum()),
        ),
        (
            "method_note",
            "Second pass uses ArcGIS World Geocoding only for unresolved "
            "California place labels. Auto-accept requires score >= 95, "
            "a California result with a county, and no different-county "
            "candidate within 3 score points. Lower-confidence cases remain "
            "for manual review.",
        ),
    ]

    summary = pd.DataFrame(summary_rows, columns=["metric", "value"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_COMBINED_PATH, index=False)
    remaining.to_csv(OUTPUT_UNRESOLVED_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print()
    print("Second-pass county geocoding complete.")
    print()
    print(
        "Second-pass matched postings: "
        f"{second_pass_matched_postings:,} / "
        f"{EXPECTED_FIRST_PASS_UNRESOLVED_POSTINGS:,}"
    )
    print(
        "Combined matched postings: "
        f"{matched_postings:,} / {EXPECTED_ALL_POSTINGS:,}"
    )
    print(
        "Remaining review postings: "
        f"{int(remaining['posting_count'].sum()):,}"
    )
    print()
    print("Wrote:")
    print(f"  {OUTPUT_COMBINED_PATH}")
    print(f"  {OUTPUT_UNRESOLVED_PATH}")
    print(f"  {OUTPUT_SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: This is still a preview. "
        "No source or Page 4 county values are modified."
    )


if __name__ == "__main__":
    main()
