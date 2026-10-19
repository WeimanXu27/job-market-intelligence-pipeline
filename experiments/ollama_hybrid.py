"""NER + Ollama hybrid experiment for job-skill extraction.

This experiment combines the local NER model with a locally hosted Ollama
language model for contextual skill expansion. It documents an alternative
local-LLM approach evaluated before the final NER + OpenAI pipeline.
"""

import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import requests
import torch
from sqlalchemy import URL, create_engine, text
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline


# Allow this experiment to reuse the public configuration in src/settings.py
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

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
OLLAMA_MODEL = "mistral"


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
    """Extract job skills with NER and a local Ollama LLM."""

    def __init__(self):
        """Initialize database, NER, and Ollama dependencies."""
        self.db_engine = None
        self.ner_pipeline = None

        self.ner_model_name = MODEL_CONFIG["ner_model"]
        self.confidence_threshold = float(
            PROCESSING_CONFIG.get("confidence_threshold", 0.5)
        )
        self.record_limit = int(
            PROCESSING_CONFIG.get("record_limit", 0)
        )

        self.ollama_url = OLLAMA_GENERATE_URL
        self.ollama_model = OLLAMA_MODEL

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
        self._init_ollama_client()

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

    def _init_ollama_client(self):
        """Verify that the local Ollama service is reachable."""
        try:
            response = requests.get(
                OLLAMA_TAGS_URL,
                timeout=5,
            )
            response.raise_for_status()

            logger.info(
                f"Ollama connection successful. Model: {self.ollama_model}"
            )

        except requests.RequestException as e:
            raise RuntimeError(
                "Local Ollama service is unavailable. "
                "Start Ollama before running this hybrid experiment."
            ) from e

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    def extract_skills_with_ner(self, job_text: str) -> List[str]:
        """Extract skills from a job description using the NER model."""
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

            return list(dict.fromkeys(skills))

        except Exception as e:
            logger.error(f"NER skill extraction failed: {e}")
            raise

    def expand_skills_with_llm(
        self,
        job_text: str,
        detected_skills: List[str],
    ) -> List[str]:
        """Use the local Ollama model to extract additional skills."""
        clean_text = self._clean_text(job_text)

        if not clean_text:
            return []

        skills_str = ", ".join(detected_skills) if detected_skills else "None"
        prompt = f"""Given the job description below and the skills already detected, extract any additional relevant technical or soft skills.

Job Description:
{clean_text}

Detected Skills: {skills_str}

Return only a valid JSON array of additional skill strings.

Example:
["Python", "SQL", "Communication"]"""

        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
        }

        try:
            response = requests.post(
                self.ollama_url,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            content = (result.get("response") or "").strip()

            if not content:
                raise RuntimeError("Ollama returned an empty response.")

            try:
                parsed = json.loads(self._strip_code_fence(content))

                if not isinstance(parsed, list):
                    raise ValueError(
                        "Expected the Ollama response to be a JSON list."
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
                    "Ollama response could not be parsed into skills."
                )

        except requests.Timeout as e:
            logger.error("Ollama request timed out.")
            raise RuntimeError("Ollama request timed out.") from e
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise RuntimeError("Ollama request failed.") from e
        except ValueError as e:
            logger.error(f"Ollama response validation failed: {e}")
            raise

    def _strip_code_fence(self, response_text: str) -> str:
        """Remove a simple Markdown code fence from an Ollama response."""
        content = response_text.strip()

        if content.startswith("```"):
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
                if content.lower().startswith("json"):
                    content = content[4:].strip()

        elif content.startswith("`"):
            content = content.strip("`").strip()

        return content

    def _extract_skills_from_text(self, response_text: str) -> List[str]:
        """Attempt simple recovery when Ollama does not return valid JSON."""
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

            if line.lower().startswith(("skill:", "skills:", "additional skills:")):
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

    def merge_and_deduplicate_skills(
        self,
        ner_skills: List[str],
        llm_skills: List[str],
    ) -> List[str]:
        """Merge NER and Ollama skills with case-insensitive deduplication."""
        final_skills = []
        seen_normalized = set()

        for skill in ner_skills + llm_skills:
            if not isinstance(skill, str):
                continue

            clean_skill = skill.strip()
            if len(clean_skill) <= 1:
                continue

            normalized = clean_skill.lower()
            if normalized in seen_normalized:
                continue

            final_skills.append(clean_skill)
            seen_normalized.add(normalized)

        return final_skills

    # ------------------------------------------------------------------
    # Single-record processing
    # ------------------------------------------------------------------
    def process_single_record(
        self,
        record_id,
        description: str,
    ) -> Tuple[bool, List[str], str]:
        """Process one record with the NER + Ollama experiment."""
        try:
            logger.info(f"Processing record {record_id}.")

            ner_skills = self.extract_skills_with_ner(description)
            logger.info(
                f"Record {record_id}: NER extracted {len(ner_skills)} skills."
            )

            llm_skills = self.expand_skills_with_llm(
                description,
                ner_skills,
            )
            logger.info(
                f"Record {record_id}: Ollama returned "
                f"{len(llm_skills)} additional skills."
            )

            final_skills = self.merge_and_deduplicate_skills(
                ner_skills,
                llm_skills,
            )
            logger.info(
                f"Record {record_id}: {len(final_skills)} final skills."
            )

            return True, final_skills, ""

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
        """Process all pending records with the NER + Ollama experiment."""
        logger.info(
            "Starting NER + Ollama experiment processing."
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
            "NER + Ollama experiment processing complete."
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
    """Run the NER + Ollama experiment."""
    try:
        extractor = SkillsExtractor()
        results = extractor.process_all_records()

        print("\n" + "=" * 50)
        print("NER + Ollama experiment complete")
        print(f"Total records: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
