"""Recover job-posting records from the historical scrape CSV backup.

The historical export contains job descriptions with embedded commas, quotes,
and line breaks that were not always escaped consistently. A standard CSV
parser therefore splits some descriptions into extra rows/columns.

This script reconstructs one logical job posting at a time, validates the
record shape, removes historical deduplication keys from the public-facing
output, and writes a clean CSV plus a compact recovery report.

It does not perform analytics feature engineering or NLP extraction. Those
steps belong downstream.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple


RECORD_START_PATTERN = re.compile(
    r"^[^,\r\n]+,\d{4}-\d{2}-\d{2},"
)

ONET_CODE_PATTERN = re.compile(
    r"^\d{2}-\d{4}\.\d{2}$"
)

PREFIX_FIELD_COUNT = 9

OUTPUT_COLUMNS = [
    "record_id",
    "date",
    "company",
    "title",
    "search_title",
    "location",
    "search_location",
    "salary_job_type",
    "hourly_salary",
    "description",
    "url",
    "onet_code",
    "city",
    "county",
    "min_salary",
    "max_salary",
    "salary_unit",
]

NULL_TOKENS = {
    "",
    "NULL",
    "null",
    "None",
    "none",
    "N/A",
    "NA",
}


def normalize_null(value: str) -> str:
    """Convert historical null tokens to an empty CSV value."""
    cleaned = value.strip()
    return "" if cleaned in NULL_TOKENS else cleaned


def iter_record_blocks(input_path: Path) -> Iterator[str]:
    """Yield one logical scrape record at a time.

    Logical records are identified by the original deduplication key followed
    by an ISO date. Continuation lines inside job descriptions are appended to
    the current record.
    """
    current_block: Optional[str] = None

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as source:
        # Skip the historical header. The recovery logic below does not depend
        # on the exact historical deduplication-key names.
        source.readline()

        for line in source:
            if RECORD_START_PATTERN.match(line):
                if current_block is not None:
                    yield current_block
                current_block = line
            elif current_block is not None:
                current_block += line

        if current_block is not None:
            yield current_block


def split_prefix_fields(
    record_text: str,
    field_count: int = PREFIX_FIELD_COUNT,
) -> Tuple[List[str], str]:
    """Parse the stable fields before the free-text description.

    The first nine historical columns precede the description field. This
    scanner respects quoted commas in fields such as company and location,
    while avoiding a full CSV parse of the malformed description body.
    """
    fields: List[str] = []
    buffer: List[str] = []
    in_quotes = False
    index = 0

    while index < len(record_text) and len(fields) < field_count:
        char = record_text[index]

        if char == '"':
            if (
                in_quotes
                and index + 1 < len(record_text)
                and record_text[index + 1] == '"'
            ):
                # Historical malformed CSV occasionally ended a quoted field
                # with two quotes directly before the delimiter. In valid CSV,
                # an escaped trailing quote would require a third closing quote
                # before the comma. Treat this pattern as the end of the field
                # and discard the extra quote.
                if (
                    index + 2 < len(record_text)
                    and record_text[index + 2] == ","
                ):
                    in_quotes = False
                    index += 2
                    continue

                buffer.append('"')
                index += 2
                continue

            in_quotes = not in_quotes
            index += 1
            continue

        if char == "," and not in_quotes:
            fields.append("".join(buffer))
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    return fields, record_text[index:]


def parse_record_block(
    record_text: str,
) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    """Recover one logical record.

    Returns:
        (record, None) on success.
        (None, reason) when the record cannot be reconstructed safely.
    """
    prefix_fields, remainder = split_prefix_fields(record_text)

    if len(prefix_fields) != PREFIX_FIELD_COUNT:
        return None, "invalid_prefix"

    # The URL is the first structured field after the free-text description.
    # Using the final ",http" boundary avoids commas and URLs that may appear
    # earlier inside the description.
    url_boundary = remainder.rfind(",http")
    if url_boundary < 0:
        return None, "url_boundary_not_found"

    description = remainder[:url_boundary]
    tail_text = remainder[url_boundary + 1 :].rstrip("\r\n")

    try:
        tail_fields = next(csv.reader([tail_text]))
    except csv.Error:
        return None, "invalid_tail_csv"

    # Tail fields:
    # url, onet_code, city, county, min_salary, max_salary,
    # salary_unit, historical_global_dedup_key
    if len(tail_fields) != 8:
        return None, "invalid_tail_shape"

    if description.startswith('"'):
        description = description[1:]

    if description.endswith('"'):
        description = description[:-1]

    description = (
        description.replace('""', '"')
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
        .strip()
    )

    (
        _historical_window_key,
        date_value,
        company,
        title,
        search_title,
        location,
        search_location,
        salary_job_type,
        hourly_salary,
    ) = prefix_fields

    (
        url,
        onet_code,
        city,
        county,
        min_salary,
        max_salary,
        salary_unit,
        _historical_global_key,
    ) = tail_fields

    record = {
        "date": normalize_null(date_value),
        "company": normalize_null(company),
        "title": normalize_null(title),
        "search_title": normalize_null(search_title),
        "location": normalize_null(location),
        "search_location": normalize_null(search_location),
        "salary_job_type": normalize_null(salary_job_type),
        "hourly_salary": normalize_null(hourly_salary),
        "description": description,
        "url": normalize_null(url),
        "onet_code": normalize_null(onet_code),
        "city": normalize_null(city),
        "county": normalize_null(county),
        "min_salary": normalize_null(min_salary),
        "max_salary": normalize_null(max_salary),
        "salary_unit": normalize_null(salary_unit),
    }

    if record["date"]:
        try:
            datetime.strptime(record["date"], "%Y-%m-%d")
        except ValueError:
            return None, "invalid_date"

    if (
        record["onet_code"]
        and not ONET_CODE_PATTERN.match(record["onet_code"])
    ):
        return None, "invalid_onet_code"

    return record, None


def recover_file(
    input_path: Path,
    output_path: Path,
    report_path: Path,
) -> Dict:
    """Recover all job-posting records and write validation outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    total_blocks = 0
    recovered_rows = 0
    drop_reasons: Counter[str] = Counter()
    dropped_record_numbers: List[int] = []
    non_null_counts: Counter[str] = Counter()
    search_title_counts: Counter[str] = Counter()

    min_date: Optional[str] = None
    max_date: Optional[str] = None

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as destination:
        writer = csv.DictWriter(
            destination,
            fieldnames=OUTPUT_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()

        for block_number, block in enumerate(
            iter_record_blocks(input_path),
            start=1,
        ):
            total_blocks += 1
            record, error = parse_record_block(block)

            if error or record is None:
                drop_reasons[error or "unknown_error"] += 1
                dropped_record_numbers.append(block_number)
                continue

            recovered_rows += 1
            record_id = f"job_{recovered_rows:06d}"

            output_record = {
                "record_id": record_id,
                **record,
            }
            writer.writerow(output_record)

            for column, value in output_record.items():
                if column != "record_id" and value:
                    non_null_counts[column] += 1

            if record["search_title"]:
                search_title_counts[record["search_title"]] += 1

            if record["date"]:
                min_date = (
                    record["date"]
                    if min_date is None
                    else min(min_date, record["date"])
                )
                max_date = (
                    record["date"]
                    if max_date is None
                    else max(max_date, record["date"])
                )

    recovery_rate = (
        recovered_rows / total_blocks
        if total_blocks
        else 0.0
    )

    report = {
        "input_file": input_path.name,
        "total_logical_record_blocks": total_blocks,
        "recovered_rows": recovered_rows,
        "dropped_rows": total_blocks - recovered_rows,
        "recovery_rate": recovery_rate,
        "drop_reasons": dict(drop_reasons),
        "dropped_record_numbers": dropped_record_numbers,
        "date_range": {
            "min": min_date,
            "max": max_date,
        },
        "non_null_counts": dict(non_null_counts),
        "unique_search_titles": len(search_title_counts),
        "top_search_titles": search_title_counts.most_common(20),
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recover logical job-posting records from the "
            "historical scrape CSV backup."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to the historical scrape CSV backup.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/recovered_job_postings.csv"
        ),
        help="Recovered CSV output path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "data/processed/recovery_report.json"
        ),
        help="Recovery report JSON output path.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the historical scrape-data recovery step."""
    args = parse_args()

    report = recover_file(
        input_path=args.input_csv,
        output_path=args.output,
        report_path=args.report,
    )

    print("Historical scrape recovery complete.")
    print(
        f"Logical record blocks: "
        f"{report['total_logical_record_blocks']}"
    )
    print(
        f"Recovered rows: {report['recovered_rows']}"
    )
    print(
        f"Dropped rows: {report['dropped_rows']}"
    )
    print(
        f"Recovery rate: "
        f"{report['recovery_rate']:.4%}"
    )
    print(
        f"Date range: "
        f"{report['date_range']['min']} "
        f"to {report['date_range']['max']}"
    )


if __name__ == "__main__":
    main()
