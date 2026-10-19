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

INPUT_PATH = DATA_DIR / "skill_patch28_ca_missing_county_city_summary.csv"

OUTPUT_MAP_PATH = DATA_DIR / "skill_patch28_ca_county_official_match_preview.csv"
OUTPUT_UNRESOLVED_PATH = DATA_DIR / "skill_patch28_ca_county_unresolved_places.csv"
OUTPUT_SUMMARY_PATH = DATA_DIR / "skill_patch28_ca_county_official_match_summary.csv"

CITY_QUERY_URL = (
    "https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/"
    "California_Cities_and_Identifiers_Blue_Version_view/FeatureServer/2/query"
)

CDP_QUERY_URL = (
    "https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/"
    "California_Census_Designated_Places_%28CDPs%29/FeatureServer/1/query"
)

COUNTY_QUERY_URL = (
    "https://services3.arcgis.com/uknczv4rpevve42E/arcgis/rest/services/"
    "California_County_Boundaries_and_Identifiers_Blue_Version_view/"
    "FeatureServer/1/query"
)

EXPECTED_CA_PLACE_ROWS = 346
EXPECTED_CA_POSTINGS = 5_098


def fetch_json(url: str, params: dict, retries: int = 3) -> dict:
    query = urlencode(params)
    request = Request(
        f"{url}?{query}",
        headers={"User-Agent": "AllSumJobs-Portfolio-County-Audit/1.0"},
    )

    last_error = None
    for attempt in range(retries):
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))

            if "error" in payload:
                raise RuntimeError(
                    f"ArcGIS API error: {payload['error']}"
                )

            return payload
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Failed to query {url} after {retries} attempts"
    ) from last_error


def normalize_place_name(value: object) -> str:
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
    text = re.sub(r"\s+", " ", text).strip()

    # Conservative place-name spelling equivalence.
    text = re.sub(r"^saint\s+", "st ", text)

    return text


def clean_county_name(value: object) -> str:
    text = "" if value is None else str(value).strip()
    text = re.sub(r"\s+County$", "", text, flags=re.IGNORECASE)
    return text


def build_incorporated_city_index() -> dict[str, list[dict]]:
    payload = fetch_json(
        CITY_QUERY_URL,
        {
            "where": "1=1",
            "outFields": (
                "CDTFA_CITY,CDTFA_COUNTY,CENSUS_PLACE_NAME,"
                "GNIS_PLACE_NAME,CDT_NAME_SHORT"
            ),
            "returnGeometry": "false",
            "resultRecordCount": 2000,
            "f": "json",
        },
    )

    features = payload.get("features", [])
    index: dict[str, list[dict]] = {}

    for feature in features:
        attrs = feature.get("attributes", {})
        county = clean_county_name(attrs.get("CDTFA_COUNTY"))

        names = {
            attrs.get("CDTFA_CITY"),
            attrs.get("CENSUS_PLACE_NAME"),
            attrs.get("CDT_NAME_SHORT"),
        }

        gnis_name = attrs.get("GNIS_PLACE_NAME")
        if gnis_name:
            names.add(
                re.sub(
                    r"^(?:City|Town) of\s+",
                    "",
                    str(gnis_name),
                    flags=re.IGNORECASE,
                )
            )

        for name in names:
            key = normalize_place_name(name)
            if not key or not county:
                continue

            index.setdefault(key, []).append(
                {
                    "official_name": attrs.get("CDT_NAME_SHORT")
                    or attrs.get("CENSUS_PLACE_NAME")
                    or attrs.get("CDTFA_CITY"),
                    "county": county,
                }
            )

    return index


def build_cdp_index() -> dict[str, list[dict]]:
    payload = fetch_json(
        CDP_QUERY_URL,
        {
            "where": "1=1",
            "outFields": "NAME,NAMELSAD,INTPTLAT,INTPTLON",
            "returnGeometry": "false",
            "resultRecordCount": 2000,
            "f": "json",
        },
    )

    features = payload.get("features", [])
    index: dict[str, list[dict]] = {}

    for feature in features:
        attrs = feature.get("attributes", {})

        names = {
            attrs.get("NAME"),
            attrs.get("NAMELSAD"),
        }

        for name in list(names):
            if name:
                names.add(
                    re.sub(
                        r"\s+CDP$",
                        "",
                        str(name),
                        flags=re.IGNORECASE,
                    )
                )

        for name in names:
            key = normalize_place_name(name)
            if not key:
                continue

            index.setdefault(key, []).append(
                {
                    "official_name": attrs.get("NAME"),
                    "lat": attrs.get("INTPTLAT"),
                    "lon": attrs.get("INTPTLON"),
                }
            )

    return index


def county_for_point(lat: object, lon: object) -> str | None:
    if lat in (None, "") or lon in (None, ""):
        return None

    lat_value = float(lat)
    lon_value = float(lon)

    payload = fetch_json(
        COUNTY_QUERY_URL,
        {
            "where": "1=1",
            "geometry": f"{lon_value},{lat_value}",
            "geometryType": "esriGeometryPoint",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "CDT_NAME_SHORT,CDTFA_COUNTY",
            "returnGeometry": "false",
            "f": "json",
        },
    )

    counties = set()

    for feature in payload.get("features", []):
        attrs = feature.get("attributes", {})
        value = (
            attrs.get("CDT_NAME_SHORT")
            or attrs.get("CDTFA_COUNTY")
        )
        county = clean_county_name(value)
        if county:
            counties.add(county)

    if len(counties) == 1:
        return next(iter(counties))

    return None


def dedupe_records(records: list[dict]) -> list[dict]:
    seen = set()
    result = []

    for record in records:
        signature = tuple(
            (key, str(record.get(key)))
            for key in sorted(record)
        )
        if signature not in seen:
            seen.add(signature)
            result.append(record)

    return result


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_PATH}")

    places = pd.read_csv(INPUT_PATH)

    if len(places) != EXPECTED_CA_PLACE_ROWS:
        raise RuntimeError(
            f"Expected {EXPECTED_CA_PLACE_ROWS} California place rows, "
            f"found {len(places)}"
        )

    if int(places["posting_count"].sum()) != EXPECTED_CA_POSTINGS:
        raise RuntimeError(
            f"Expected {EXPECTED_CA_POSTINGS:,} California postings, "
            f"found {int(places['posting_count'].sum()):,}"
        )

    print("Downloading official California geography reference data...")
    city_index = build_incorporated_city_index()
    cdp_index = build_cdp_index()

    print(
        f"Official incorporated-city name keys: {len(city_index):,}"
    )
    print(f"Official CDP name keys: {len(cdp_index):,}")
    print()

    county_point_cache: dict[tuple[str, str], str | None] = {}

    output_rows = []

    for row in places.itertuples(index=False):
        city = str(row.city).strip()
        key = normalize_place_name(city)

        result = {
            "city": city,
            "posting_count": int(row.posting_count),
            "mapped_county": "",
            "match_status": "unresolved",
            "match_method": "",
            "official_place_name": "",
            "review_note": "",
            "location_examples": row.location_examples,
            "search_market_examples": row.search_market_examples,
        }

        city_matches = dedupe_records(city_index.get(key, []))

        city_counties = sorted(
            {match["county"] for match in city_matches}
        )

        if len(city_counties) == 1:
            result.update(
                {
                    "mapped_county": city_counties[0],
                    "match_status": "auto_matched",
                    "match_method": "official_incorporated_city",
                    "official_place_name": " | ".join(
                        sorted(
                            {
                                str(match["official_name"])
                                for match in city_matches
                                if match.get("official_name")
                            }
                        )
                    ),
                }
            )
            output_rows.append(result)
            continue

        if len(city_counties) > 1:
            result.update(
                {
                    "match_status": "manual_review",
                    "match_method": "ambiguous_incorporated_city_name",
                    "review_note": " | ".join(city_counties),
                }
            )
            output_rows.append(result)
            continue

        cdp_matches = dedupe_records(cdp_index.get(key, []))

        resolved_cdp = []

        for match in cdp_matches:
            cache_key = (
                str(match.get("lat")),
                str(match.get("lon")),
            )

            if cache_key not in county_point_cache:
                county_point_cache[cache_key] = county_for_point(
                    match.get("lat"),
                    match.get("lon"),
                )
                time.sleep(0.05)

            county = county_point_cache[cache_key]

            if county:
                resolved_cdp.append(
                    {
                        **match,
                        "county": county,
                    }
                )

        cdp_counties = sorted(
            {match["county"] for match in resolved_cdp}
        )

        if len(cdp_counties) == 1:
            result.update(
                {
                    "mapped_county": cdp_counties[0],
                    "match_status": "auto_matched",
                    "match_method": "official_cdp_internal_point",
                    "official_place_name": " | ".join(
                        sorted(
                            {
                                str(match["official_name"])
                                for match in resolved_cdp
                                if match.get("official_name")
                            }
                        )
                    ),
                    "review_note": (
                        "County derived from the official CDP internal "
                        "point intersected with the official California "
                        "county boundary layer."
                    ),
                }
            )
        elif len(cdp_counties) > 1:
            result.update(
                {
                    "match_status": "manual_review",
                    "match_method": "ambiguous_cdp_name",
                    "review_note": " | ".join(cdp_counties),
                }
            )
        elif cdp_matches:
            result.update(
                {
                    "match_status": "manual_review",
                    "match_method": "cdp_match_county_unresolved",
                    "official_place_name": " | ".join(
                        sorted(
                            {
                                str(match["official_name"])
                                for match in cdp_matches
                                if match.get("official_name")
                            }
                        )
                    ),
                }
            )

        output_rows.append(result)

    output = pd.DataFrame(output_rows)

    unresolved = output.loc[
        output["match_status"].ne("auto_matched")
    ].copy()

    method_summary = (
        output.groupby(
            ["match_status", "match_method"],
            dropna=False,
        )
        .agg(
            unique_places=("city", "nunique"),
            postings=("posting_count", "sum"),
        )
        .reset_index()
        .sort_values(
            ["match_status", "postings"],
            ascending=[True, False],
        )
    )

    matched_postings = int(
        output.loc[
            output["match_status"].eq("auto_matched"),
            "posting_count",
        ].sum()
    )

    summary_rows = [
        ("input_unique_ca_places", len(output)),
        ("input_ca_postings", int(output["posting_count"].sum())),
        (
            "auto_matched_unique_places",
            int(
                output.loc[
                    output["match_status"].eq("auto_matched"),
                    "city",
                ].nunique()
            ),
        ),
        ("auto_matched_postings", matched_postings),
        (
            "auto_matched_posting_pct",
            round(
                matched_postings
                / int(output["posting_count"].sum())
                * 100,
                4,
            ),
        ),
        ("unresolved_unique_places", len(unresolved)),
        (
            "unresolved_postings",
            int(unresolved["posting_count"].sum()),
        ),
        (
            "source_note",
            "Auto matches use California Department of Technology "
            "official incorporated-city, CDP, and county boundary "
            "ArcGIS services. No fuzzy city-name matching is used.",
        ),
        (
            "cdp_note",
            "CDP county is derived from the official CDP internal "
            "point intersected with the official county boundary layer. "
            "Unresolved names remain for manual review.",
        ),
    ]

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "value"],
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output.to_csv(OUTPUT_MAP_PATH, index=False)
    unresolved.to_csv(OUTPUT_UNRESOLVED_PATH, index=False)
    summary.to_csv(OUTPUT_SUMMARY_PATH, index=False)

    print("Official California county-match preview complete.")
    print()
    print(method_summary.to_string(index=False))
    print()
    print(f"Auto-matched postings: {matched_postings:,} / {EXPECTED_CA_POSTINGS:,}")
    print(
        "Unresolved postings: "
        f"{int(unresolved['posting_count'].sum()):,}"
    )
    print()
    print("Wrote:")
    print(f"  {OUTPUT_MAP_PATH}")
    print(f"  {OUTPUT_UNRESOLVED_PATH}")
    print(f"  {OUTPUT_SUMMARY_PATH}")
    print()
    print(
        "IMPORTANT: This script does not modify recovered_job_postings.csv "
        "or any frozen Page 4 table. It only creates a mapping preview."
    )


if __name__ == "__main__":
    main()
