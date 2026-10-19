"""Load recovered job descriptions into the local MySQL skill-extraction table.

The loader uses the same database/table configuration as src/skill_extraction.py.
It loads only the stable record ID and job description, leaves extracted_skills
NULL for new records, and preserves any existing extracted_skills on reruns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import Column, MetaData, String, Table, create_engine, func, select
from sqlalchemy.dialects.mysql import LONGTEXT, insert as mysql_insert
from sqlalchemy.engine import URL


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import DATABASE_CONFIG, DATA_CONFIG  # noqa: E402


DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "recovered_job_postings.csv"
DEFAULT_CHUNK_SIZE = 1000

INVALID_DESCRIPTION_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "not available",
    "not applicable",
    "-",
    "--",
    ".",
    ":",
    "unknown",
}

TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


def build_engine():
    """Create a MySQL engine using the same configuration as skill_extraction.py."""
    connection_url = URL.create(
        drivername="mysql+mysqlconnector",
        username=DATABASE_CONFIG["username"],
        password=DATABASE_CONFIG["password"],
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        database=DATABASE_CONFIG["database"],
        query={"charset": DATABASE_CONFIG["charset"]},
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True,
        echo=False,
    )


def define_job_postings_table(metadata: MetaData) -> Table:
    """Define the minimal table required by the skill-extraction pipeline."""
    return Table(
        TABLE_NAME,
        metadata,
        Column(ID_COLUMN, String(255), primary_key=True),
        Column(DESCRIPTION_COLUMN, LONGTEXT, nullable=True),
        Column(SKILLS_COLUMN, LONGTEXT, nullable=True),
        mysql_charset=DATABASE_CONFIG["charset"],
    )


def load_source_rows(
    input_path: Path,
    engine,
    table: Table,
    chunk_size: int,
) -> tuple[int, int, int, int]:
    """Load record IDs/descriptions and return source validation counts."""
    source_rows = 0
    nonblank_descriptions = 0
    blank_descriptions = 0
    seen_ids: set[str] = set()

    for chunk in pd.read_csv(
        input_path,
        usecols=[ID_COLUMN, DESCRIPTION_COLUMN],
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        chunksize=chunk_size,
    ):
        chunk[ID_COLUMN] = chunk[ID_COLUMN].astype(str).str.strip()

        if chunk[ID_COLUMN].eq("").any():
            raise ValueError("Blank record_id found in the source file.")

        duplicate_ids = set(chunk[ID_COLUMN]).intersection(seen_ids)
        if duplicate_ids:
            example = sorted(duplicate_ids)[0]
            raise ValueError(
                f"Duplicate record_id found in source data: {example}"
            )

        seen_ids.update(chunk[ID_COLUMN].tolist())

        descriptions = (
            chunk[DESCRIPTION_COLUMN]
            .astype(str)
        )

        normalized_descriptions = (
            descriptions
            .str.strip()
            .str.lower()
        )

        usable_mask = (
            ~normalized_descriptions.isin(
                INVALID_DESCRIPTION_VALUES
            )
        )

        nonblank_descriptions += int(
            usable_mask.sum()
        )
        blank_descriptions += int(
            (~usable_mask).sum()
        )
        source_rows += len(chunk)

        chunk[DESCRIPTION_COLUMN] = (
            descriptions.where(
                usable_mask,
                None,
            )
        )

        records = chunk.to_dict(orient="records")
        insert_stmt = mysql_insert(table).values(records)
        upsert_stmt = insert_stmt.on_duplicate_key_update(
            **{
                DESCRIPTION_COLUMN: getattr(
                    insert_stmt.inserted,
                    DESCRIPTION_COLUMN,
                )
            }
        )

        with engine.begin() as conn:
            conn.execute(upsert_stmt)

        print(f"Loaded {source_rows:,} source rows...")

    return (
        source_rows,
        len(seen_ids),
        nonblank_descriptions,
        blank_descriptions,
    )


def validate_database(engine, table: Table) -> dict[str, int]:
    """Return database counts needed before skill extraction begins."""
    description_is_nonblank = (
        table.c[DESCRIPTION_COLUMN].is_not(None)
        & (func.trim(table.c[DESCRIPTION_COLUMN]) != "")
    )

    with engine.connect() as conn:
        total_rows = conn.execute(
            select(func.count()).select_from(table)
        ).scalar_one()

        unique_record_ids = conn.execute(
            select(func.count(func.distinct(table.c[ID_COLUMN])))
        ).scalar_one()

        nonblank_descriptions = conn.execute(
            select(func.count())
            .select_from(table)
            .where(description_is_nonblank)
        ).scalar_one()

        pending_extraction = conn.execute(
            select(func.count())
            .select_from(table)
            .where(
                description_is_nonblank,
                table.c[SKILLS_COLUMN].is_(None),
            )
        ).scalar_one()

    return {
        "total_rows": int(total_rows),
        "unique_record_ids": int(unique_record_ids),
        "nonblank_descriptions": int(nonblank_descriptions),
        "blank_descriptions": int(total_rows - nonblank_descriptions),
        "pending_extraction": int(pending_extraction),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load recovered job descriptions into the MySQL table used by "
            "src/skill_extraction.py."
        )
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=(
            "Path to recovered_job_postings.csv. "
            "Defaults to data/processed/recovered_job_postings.csv."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    return parser.parse_args()


def main() -> None:
    """Load the source table and validate extraction readiness."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive.")

    engine = build_engine()
    metadata = MetaData()
    table = define_job_postings_table(metadata)

    try:
        with engine.connect() as conn:
            conn.execute(select(1))

        metadata.create_all(engine)

        (
            source_rows,
            source_unique_ids,
            source_nonblank,
            source_blank,
        ) = load_source_rows(
            input_path=args.input_csv,
            engine=engine,
            table=table,
            chunk_size=args.chunk_size,
        )

        database_counts = validate_database(engine, table)

        print("\nSkill-source MySQL load complete.")
        print(
            f"Table: {DATABASE_CONFIG['database']}.{TABLE_NAME}"
        )
        print(f"Source rows: {source_rows:,}")
        print(f"Source unique record IDs: {source_unique_ids:,}")
        print(f"Source nonblank descriptions: {source_nonblank:,}")
        print(f"Source blank descriptions: {source_blank:,}")
        print(f"Database total rows: {database_counts['total_rows']:,}")
        print(
            "Database unique record IDs: "
            f"{database_counts['unique_record_ids']:,}"
        )
        print(
            "Database nonblank descriptions: "
            f"{database_counts['nonblank_descriptions']:,}"
        )
        print(
            "Database blank descriptions: "
            f"{database_counts['blank_descriptions']:,}"
        )
        print(
            "Pending extraction rows: "
            f"{database_counts['pending_extraction']:,}"
        )

        if source_rows != source_unique_ids:
            raise RuntimeError(
                "Source validation failed: record_id values are not unique."
            )

        if database_counts["total_rows"] != source_rows:
            raise RuntimeError(
                "Database validation failed: table row count does not match "
                "the recovered source file."
            )

        if database_counts["unique_record_ids"] != source_unique_ids:
            raise RuntimeError(
                "Database validation failed: unique record_id count mismatch."
            )

        if database_counts["nonblank_descriptions"] != source_nonblank:
            raise RuntimeError(
                "Database validation failed: nonblank description count mismatch."
            )

        if database_counts["blank_descriptions"] != source_blank:
            raise RuntimeError(
                "Database validation failed: blank description count mismatch."
            )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
