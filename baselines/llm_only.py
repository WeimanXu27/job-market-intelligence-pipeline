"""LLM-only baseline for job-skill extraction.

This baseline extracts skills directly from job descriptions with an OpenAI
language model. It represents the simpler LLM-only approach evaluated before
the final hybrid NER + LLM pipeline.
"""

import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from sqlalchemy import URL, create_engine, text


# Allow this baseline to reuse the public configuration in src/settings.py
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from settings import (  # noqa: E402
    OPENAI_API_KEY,
    DATABASE_CONFIG,
    DATA_CONFIG,
    MODEL_CONFIG,
    PROCESSING_CONFIG,
    LOGGING_CONFIG,
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


class SimpleSkillsExtractor:
    """Extract job skills using only a hosted LLM."""

    def __init__(self):
        """Initialize database and LLM dependencies."""
        self.db_engine = None
        self.llm_client = None

        self.openai_model = MODEL_CONFIG.get(
            "llm_model",
            "gpt-4o-mini",
        )

        self.record_limit = int(
            PROCESSING_CONFIG.get("record_limit", 0)
        )

        if self.record_limit < 0:
            raise ValueError(
                "Record limit must be zero or a positive integer."
            )

        self._init_database()
        self._init_llm_client()

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

    def _init_llm_client(self):
        """Initialize the OpenAI client."""
        if (
            not OPENAI_API_KEY
            or OPENAI_API_KEY == "your_openai_api_key_here"
        ):
            raise RuntimeError(
                "OPENAI_API_KEY is required for the LLM-only baseline."
            )

        try:
            self.llm_client = OpenAI(
                api_key=OPENAI_API_KEY
            )

            logger.info(
                f"LLM client initialized. Model: {self.openai_model}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_skills_with_llm(self, job_text: str) -> List[str]:
        """Extract skills from a job description using the LLM.

        Args:
            job_text: Raw job-description text.

        Returns:
            A list of extracted skill strings.

        Raises:
            RuntimeError: If the LLM request fails or returns an unusable
                response.
        """
        if not self.llm_client:
            raise RuntimeError("LLM client is not initialized.")

        clean_text = self._clean_text(job_text)

        if not clean_text:
            return []

        prompt = f"""Given the job description below, extract the relevant skills.

Include:
- Programming languages
- Frameworks and libraries
- Tools and technologies
- Technical skills
- Soft skills
- Certifications and qualifications

Job Description:
{clean_text}

Return only a valid JSON array of skill strings.

Example:
["Python", "SQL", "Communication"]"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You extract skills from job descriptions. "
                            "Return only a valid JSON array."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
            )

            content = response.choices[0].message.content or ""
            content = content.strip()

            if not content:
                raise RuntimeError(
                    "LLM returned an empty response."
                )

            try:
                parsed = json.loads(content)

                if not isinstance(parsed, list):
                    raise ValueError(
                        "Expected the LLM response to be a JSON list."
                    )

                return [
                    skill.strip()
                    for skill in parsed
                    if isinstance(skill, str) and skill.strip()
                ]

            except json.JSONDecodeError:
                skills = self._extract_skills_from_text(content)

                if skills:
                    return skills

                raise RuntimeError(
                    "LLM response could not be parsed into skills."
                )

        except Exception as e:
            logger.error(f"LLM skill extraction failed: {e}")
            raise

    def _extract_skills_from_text(self, response_text: str) -> List[str]:
        """Attempt simple recovery when the LLM does not return valid JSON."""
        skills = []

        for line in response_text.splitlines():
            line = line.strip()

            if not line:
                continue

            line = re.sub(
                r"^[\-\*\d\.\)\s]+",
                "",
                line,
            ).strip()

            if not line:
                continue

            if line.startswith("{") or line.startswith("["):
                continue

            skills.append(line)

        return skills

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

            skills = self.extract_skills_with_llm(
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
        """Process all pending records with the LLM-only baseline."""
        logger.info(
            "Starting LLM-only baseline processing."
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
            "LLM-only baseline processing complete."
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
    """Run the LLM-only baseline."""
    try:
        extractor = SimpleSkillsExtractor()
        results = extractor.process_all_records()

        print("\n" + "=" * 50)
        print("LLM-only baseline complete")
        print(f"Total records: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()