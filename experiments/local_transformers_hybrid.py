"""NER + local Transformers hybrid experiment for job-skill extraction.

This experiment combines the local NER model with a Mistral instruction model
loaded directly through Hugging Face Transformers. It documents a fully local
inference approach evaluated before the final NER + OpenAI pipeline.

Unlike the Ollama experiment, this version loads the language model inside the
Python process rather than calling a separate local model server.
"""

import json
import logging
import re
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sqlalchemy import URL, create_engine, text
from transformers import (
    AutoModelForCausalLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)


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

LOCAL_LLM_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
LOCAL_LLM_MAX_NEW_TOKENS = 512


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
    """Extract job skills with NER and a directly loaded local LLM."""

    def __init__(self):
        """Initialize database, NER, and local Transformers dependencies."""
        self.db_engine = None
        self.ner_pipeline = None
        self.llm_pipeline = None
        self.llm_tokenizer = None

        self.ner_model_name = MODEL_CONFIG["ner_model"]
        self.local_llm_model = LOCAL_LLM_MODEL
        self.confidence_threshold = float(
            PROCESSING_CONFIG.get("confidence_threshold", 0.5)
        )
        self.record_limit = int(
            PROCESSING_CONFIG.get("record_limit", 0)
        )
        self.use_gpu = torch.cuda.is_available()

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
        self._init_llm_model()

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

            device = 0 if self.use_gpu else -1
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

    def _init_llm_model(self):
        """Load Mistral directly through Hugging Face Transformers.

        The historical experiment first attempted a normal model load and then
        fell back to 8-bit quantization when the normal load failed.
        """
        logger.info(
            f"Loading local Transformers LLM: {self.local_llm_model}"
        )

        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                self.local_llm_model
            )
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(
                "Local Transformers tokenizer could not be loaded."
            ) from e

        try:
            model = self._load_llm_standard()
        except Exception:
            logger.warning(
                "Standard local LLM loading failed; "
                "attempting 8-bit quantization."
            )

            if self.use_gpu:
                torch.cuda.empty_cache()

            try:
                model = self._load_llm_quantized_8bit()
            except Exception as quantized_error:
                raise RuntimeError(
                    "Local Transformers LLM could not be loaded "
                    "in standard or 8-bit mode."
                ) from quantized_error

        self.llm_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.llm_tokenizer,
        )

        logger.info("Local Transformers LLM loaded successfully.")

    def _load_llm_standard(self):
        """Attempt the historical standard local-model loading path."""
        if self.use_gpu:
            logger.info("Attempting standard GPU loading.")
            device_map = "auto"
            torch_dtype = torch.float16
        else:
            logger.warning(
                "CUDA is unavailable; attempting standard CPU loading."
            )
            device_map = "cpu"
            torch_dtype = torch.float32

        return AutoModelForCausalLM.from_pretrained(
            self.local_llm_model,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

    def _load_llm_quantized_8bit(self):
        """Attempt the historical 8-bit fallback path."""
        logger.info("Attempting 8-bit quantized loading.")

        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

        device_map = "auto" if self.use_gpu else "cpu"
        torch_dtype = (
            torch.float16 if self.use_gpu else torch.float32
        )

        return AutoModelForCausalLM.from_pretrained(
            self.local_llm_model,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )

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
        """Use the directly loaded local Mistral model to find more skills."""
        if not self.llm_pipeline:
            raise RuntimeError(
                "Local Transformers LLM pipeline is not initialized."
            )

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

        try:
            response = self.llm_pipeline(
                prompt,
                max_new_tokens=LOCAL_LLM_MAX_NEW_TOKENS,
                do_sample=False,
                truncation=True,
                return_full_text=False,
            )

            content = (
                response[0].get("generated_text", "")
                if response
                else ""
            )
            content = content.strip()

            if not content:
                raise RuntimeError(
                    "Local Transformers LLM returned an empty response."
                )

            try:
                parsed = json.loads(
                    self._strip_code_fence(content)
                )

                if not isinstance(parsed, list):
                    raise ValueError(
                        "Expected the local LLM response to be a JSON list."
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
                    "Local LLM response could not be parsed into skills."
                )

        except Exception as e:
            logger.error(
                f"Local Transformers skill expansion failed: {e}"
            )
            raise

    def _strip_code_fence(self, response_text: str) -> str:
        """Remove a simple Markdown code fence from a local LLM response."""
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
        """Attempt simple recovery when the local LLM does not return JSON."""
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

            if line.lower().startswith(
                ("skill:", "skills:", "additional skills:")
            ):
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
        """Merge NER and local LLM skills with case-insensitive deduplication."""
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
        """Process one record with the NER + local Transformers experiment."""
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
                f"Record {record_id}: local Transformers LLM returned "
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
        """Process pending records with the local Transformers experiment."""
        logger.info(
            "Starting NER + local Transformers experiment processing."
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
            "NER + local Transformers experiment processing complete."
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
    """Run the NER + local Transformers experiment."""
    try:
        extractor = SkillsExtractor()
        results = extractor.process_all_records()

        print("\n" + "=" * 50)
        print("NER + local Transformers experiment complete")
        print(f"Total records: {results['total']}")
        print(f"Successful: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("=" * 50)

    except Exception as e:
        logger.error(f"Program execution failed: {e}")
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
