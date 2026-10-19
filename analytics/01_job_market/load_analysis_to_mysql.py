"""Load the analytics-ready CSV into a local MySQL staging table.

This utility avoids relying on GUI CSV-import inference. It loads every CSV
column as MySQL TEXT first, validates row counts, and leaves explicit type
conversion to the downstream SQL layer.

The password is requested interactively so it is not stored in shell history.
"""

from __future__ import annotations

import argparse
import getpass
from pathlib import Path

import pandas as pd
from sqlalchemy import URL, Text, create_engine, text


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 3306
DEFAULT_DATABASE = "job_market_portfolio"
DEFAULT_USER = "root"
DEFAULT_TABLE = "analysis_job_postings_staging"
DEFAULT_CHUNK_SIZE = 5000


def build_engine(
    host: str,
    port: int,
    database: str,
    user: str,
    password: str,
):
    """Create a local MySQL SQLAlchemy engine."""
    connection_url = URL.create(
        drivername="mysql+mysqlconnector",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
        query={"charset": "utf8mb4"},
    )

    return create_engine(
        connection_url,
        pool_pre_ping=True,
        echo=False,
    )


def load_csv_to_staging(
    input_path: Path,
    engine,
    table_name: str,
    chunk_size: int,
) -> tuple[int, int]:
    """Load all CSV columns as TEXT and validate row/ID counts."""
    first_chunk = True
    loaded_rows = 0

    for chunk in pd.read_csv(
        input_path,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
        chunksize=chunk_size,
    ):
        dtype_map = {
            column: Text()
            for column in chunk.columns
        }

        chunk.to_sql(
            table_name,
            con=engine,
            if_exists=(
                "replace"
                if first_chunk
                else "append"
            ),
            index=False,
            dtype=dtype_map,
            chunksize=1000,
            method="multi",
        )

        loaded_rows += len(chunk)
        first_chunk = False

        print(
            f"Loaded {loaded_rows:,} rows..."
        )

    if first_chunk:
        raise ValueError(
            "Input CSV is empty."
        )

    with engine.connect() as conn:
        total_rows = conn.execute(
            text(
                f"SELECT COUNT(*) "
                f"FROM `{table_name}`"
            )
        ).scalar_one()

        unique_record_ids = conn.execute(
            text(
                f"SELECT COUNT(DISTINCT record_id) "
                f"FROM `{table_name}`"
            )
        ).scalar_one()

    return int(total_rows), int(unique_record_ids)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Load analysis_job_postings.csv "
            "into a local MySQL staging table."
        )
    )

    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to analysis_job_postings.csv.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER,
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    return parser.parse_args()


def main() -> None:
    """Load and validate the local MySQL staging table."""
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: "
            f"{args.input_csv}"
        )

    if args.chunk_size <= 0:
        raise ValueError(
            "chunk-size must be positive."
        )

    password = getpass.getpass(
        f"MySQL password for "
        f"{args.user}@{args.host}: "
    )

    engine = build_engine(
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=password,
    )

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        total_rows, unique_record_ids = (
            load_csv_to_staging(
                input_path=args.input_csv,
                engine=engine,
                table_name=args.table,
                chunk_size=args.chunk_size,
            )
        )

        print(
            "\nMySQL staging load complete."
        )
        print(
            f"Table: "
            f"{args.database}."
            f"{args.table}"
        )
        print(
            f"Total rows: "
            f"{total_rows:,}"
        )
        print(
            f"Unique record IDs: "
            f"{unique_record_ids:,}"
        )

        if (
            total_rows
            != unique_record_ids
        ):
            raise RuntimeError(
                "Staging validation failed: "
                "record_id values are not unique."
            )

    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
