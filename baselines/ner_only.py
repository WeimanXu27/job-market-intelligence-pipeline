"""NER-only baseline for job-skill extraction.

This baseline extracts skills from job descriptions using only the
Nucha/Nucha_ITSkillNER_BERT named-entity recognition model. It represents the
local NER-only approach evaluated before the final hybrid NER + LLM pipeline.
"""

import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sqlalchemy import URL, create_engine, text
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


# Allow this baseline to reuse the public configuration in src/settings.py
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import (  # noqa: E402
    DATABASE_CONFIG,
    DATA_CONFIG,
    LOGGING_CONFIG,
    MODEL_CONFIG,
    PROCESSING_CONFIG,
)


TABLE_NAME = DATA_CONFIG["table_name"]
ID_COLUMN = DATA_CONFIG["id_column"]
DESCRIPTION_COLUMN = DATA_CONFIG["description_column"]
SKILLS_COLUMN = DATA_CONFIG["skills_column"]


# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
log_file = LOGGING_CONFIG.get(
    "log_file",
    "skill_extraction.log",
)
log_level = getattr(
    logging,
    LOGGING_CONFIG.get("log_level", "INFO"),
)

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class SkillsExtractor:
    """Extract job skills using only the local NER model."""

    def __init__(self):
        """Initialize database and NER dependencies."""
        self.db_engine = None
        self.ner_pipeline = None

        self.ner_model_name = MODEL_CONFIG["ner_model"]
        self.confidence_threshold = float(
            PROCESSING_CONFIG.get("confidence_threshold", 0.5)
        )
        self.record_limit = int(
            PROCESSING_CONFIG.get("record_limit", 0)
        )

        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "NER confidence threshold must be between 0 and 1."
            )

        if self.record_limit < 0:
            raise ValueError(
                "Record limit must be zero or a positive integer."
            )

        self._init_database()
        self._init_ner_model()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _init_database(self):
        """Initialize the database connection."""
        try:
            db_config = DATABASE_CONFIG

            connection_url = URL.create(
                drivername="mysql+mysqlconnector",
                username=db_config["username"],
                password=db_config["password"],
                host=db_config["host"],
                port=db_config["port"],
                database=db_config["database"],
                query={
                    "charset": db_config["charset"],
                },
            )

            self.db_engine = create_engine(
                connection_url,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_size=10,
                max_overflow=20,
                echo=False,
            )

            with self.db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Database connection successful.")

        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def _init_ner_model(self):
        """Initialize the Hugging Face NER pipeline."""
        try:
            logger.info(
                f"Loading NER model: {self.ner_model_name}"
            )

            device = 0 if torch.cuda.is_available() else -1
            logger.info(
                f"NER inference device: {'GPU' if device >= 0 else 'CPU'}"
            )

            tokenizer = AutoTokenizer.from_pretrained(
                self.ner_model_name
            )
            model = AutoModelForTokenClassification.from_pretrained(
                self.ner_model_name
            )

            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device,
            )

            logger.info("NER model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load NER model: {e}")
            raise

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_skills_with_ner(self, job_text: str) -> List[str]:
        """Extract skills from a job description using only NER."""
        if not self.ner_pipeline:
            raise RuntimeError("NER pipeline is not initialized.")

        clean_text = self._clean_text(job_text)

        if not clean_text:
            return []

        try:
            results = self.ner_pipeline(clean_text)

            skills = []
            for result in results:
                if result["score"] > self.confidence_threshold:
                    skill = result["word"].strip()

                    if skill and len(skill) > 1:
                        skills.append(skill)

            # Remove duplicates while preserving model output order.
            return list(dict.fromkeys(skills))

        except Exception as e:
            logger.error(f"NER skill extraction failed: {e}")
            raise

    def _clean_text(self, job_text: str) -> str:
        """Apply lightweight cleaning to job-description text."""
        if not job_text:
            return ""

        job_text = re.sub(
            r"<[^>]+>",
            "",
            str(job_text),
        )

        job_text = re.sub(
            r"\s+",
            " ",
            job_text,
        )

        return job_text.strip()

    # ------------------------------------------------------------------
    # Single-record processing
    # ------------------------------------------------------------------
    def process_single_record(
        self,
        record_id,
        description: str,
    ) -> Tuple[bool, List[str], str]:
        """Process one database record."""
        try:
            logger.info(f"Processing record {record_id}.")

            skills = self.extract_skills_with_ner(
                description
            )

            logger.info(
                f"Record {record_id}: extracted {len(skills)} skills."
            )

            return True, skills, ""

        except Exception as e:
            error_msg = (
                f"Failed to process record {record_id}: {e}"
            )

            logger.error(error_msg)
            logger.error(traceback.format_exc())

            return False, [], error_msg

    # ------------------------------------------------------------------
    # Database operations
    # ------------------------------------------------------------------
    def update_record_skills(
        self,
        record_id,
        skills: List[str],
    ) -> bool:
        """Store extracted skills as dashboard-ready numbered text."""
        try:
            numbered_skills = [
                f"{index}. {skill}"
                for index, skill in enumerate(skills, start=1)
            ]

            skills_text = "\n".join(numbered_skills)

            with self.db_engine.connect() as conn:
                update_sql = text(
                    f"UPDATE {TABLE_NAME} "
                    f"SET {SKILLS_COLUMN} = :skills "
                    f"WHERE {ID_COLUMN} = :record_id"
                )

                conn.execute(
                    update_sql,
                    {
                        "skills": skills_text,
                        "record_id": record_id,
                    },
                )

                conn.commit()

            logger.info(
                f"Record {record_id} skills updated successfully."
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to update skills for record "
                f"{record_id}: {e}"
            )

            return False

    def get_all_records(self) -> pd.DataFrame:
        """Fetch pending records from the public job-postings schema."""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(
                    text(f"DESCRIBE {TABLE_NAME}")
                )

                columns = [
                    row[0]
                    for row in result.fetchall()
                ]

                if ID_COLUMN not in columns:
                    raise RuntimeError(
                        f"Configured ID column '{ID_COLUMN}' "
                        f"was not found in table '{TABLE_NAME}'."
                    )

                if DESCRIPTION_COLUMN not in columns:
                    raise RuntimeError(
                        f"Configured description column "
                        f"'{DESCRIPTION_COLUMN}' was not found "
                        f"in table '{TABLE_NAME}'."
                    )

                if SKILLS_COLUMN not in columns:
                    conn.execute(
                        text(
                            f"ALTER TABLE {TABLE_NAME} "
                            f"ADD COLUMN {SKILLS_COLUMN} TEXT"
                        )
                    )

                    conn.commit()

                    logger.info(
                        f"Created {SKILLS_COLUMN} column."
                    )

                query_sql = (
                    f"SELECT {ID_COLUMN}, {DESCRIPTION_COLUMN} "
                    f"FROM {TABLE_NAME} "
                    f"WHERE {SKILLS_COLUMN} IS NULL "
                    f"AND {DESCRIPTION_COLUMN} IS NOT NULL "
                    f"AND TRIM({DESCRIPTION_COLUMN}) <> '' "
                    f"ORDER BY {ID_COLUMN}"
                )

                if self.record_limit > 0:
                    query_sql += (
                        f" LIMIT {self.record_limit}"
                    )

                    logger.info(
                        f"Record limit enabled: "
                        f"{self.record_limit}"
                    )

                else:
                    logger.info(
                        "Record limit disabled; processing "
                        "all pending records."
                    )

                df = pd.read_sql(
                    text(query_sql),
                    conn,
                )

                logger.info(
                    f"Records to process: {len(df)}"
                )

                return df

        except Exception as e:
            logger.error(f"Failed to fetch records: {e}")
            logger.error(traceback.format_exc())
            raise

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------
    def process_all_records(self) -> Dict:
        """Process all pending records with the NER-only baseline."""
        logger.info(
            "Starting NER-only baseline processing."
        )

        records_df = self.get_all_records()

        if records_df.empty:
            logger.info(
                "No pending records found."
            )

            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "failed_records": [],
            }

        stats = {
            "total": len(records_df),
            "success": 0,
            "failed": 0,
            "failed_records": [],
        }

        for _, row in records_df.iterrows():
            record_id = row[ID_COLUMN]
            description = row[DESCRIPTION_COLUMN]

            success, skills, error_msg = (
                self.process_single_record(
                    record_id,
                    description,
                )
            )

            if not success:
                stats["failed"] += 1

                stats["failed_records"].append(
                    {
                        "id": record_id,
                        "error": error_msg,
                    }
                )

                continue

            if self.update_record_skills(
                record_id,
                skills,
            ):
                stats["success"] += 1

                if skills:
                    logger.info(
                        f"Record {record_id} processing complete."
                    )
                else:
                    logger.warning(
                        f"Record {record_id} processed "
                        "successfully but no skills were extracted."
                    )

            else:
                stats["failed"] += 1

                stats["failed_records"].append(
                    {
                        "id": record_id,
                        "error": "Database update failed.",
                    }
                )

        logger.info(
            "NER-only baseline processing complete."
        )
        logger.info(
            f"Total records: {stats['total']}"
        )
        logger.info(
            f"Successful: {stats['success']}"
        )
        logger.info(
            f"Failed: {stats['failed']}"
        )

        return stats


def main():
    """Run the NER-only baseline."""
    try:
        extractor = SkillsExtractor()
        results = extractor.process_all_records()

        print("\n" + "=" * 50)
        print("NER-only baseline complete")
        print(f"Total records: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
