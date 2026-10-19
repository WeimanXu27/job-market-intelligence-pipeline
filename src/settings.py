"""Public-safe configuration for the skill extraction pipeline."""

import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


DATABASE_CONFIG = {
    "host": os.getenv("DATABASE_HOST", "localhost"),
    "port": int(os.getenv("DATABASE_PORT", "3306")),
    "database": os.getenv("DATABASE_NAME", "job_market_portfolio"),
    "username": os.getenv("DATABASE_USER", ""),
    "password": os.getenv("DATABASE_PASSWORD", ""),
    "charset": os.getenv("DATABASE_CHARSET", "utf8mb4"),
}


DATA_CONFIG = {
    "table_name": "job_postings",
    "id_column": "record_id",
    "description_column": "description",
    "skills_column": "extracted_skills",
}


MODEL_CONFIG = {
    "ner_model": os.getenv(
        "NER_MODEL",
        "Nucha/Nucha_ITSkillNER_BERT",
    ),
    "llm_model": os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini",
    ),
}


PROCESSING_CONFIG = {
    "confidence_threshold": float(
        os.getenv("NER_CONFIDENCE_THRESHOLD", "0.5")
    ),
    "llm_batch_size": int(
        os.getenv("LLM_BATCH_SIZE", "10")
    ),
    "record_limit": int(
        os.getenv("RECORD_LIMIT", "0")
    ),
    "max_prompt_chars": int(
        os.getenv("MAX_PROMPT_CHARS", "12000")
    ),
    "debug_llm_empty": (
        os.getenv("DEBUG_LLM_EMPTY", "false").lower() == "true"
    ),
    "debug_llm_empty_max": int(
        os.getenv("DEBUG_LLM_EMPTY_MAX", "10")
    ),
}


LOGGING_CONFIG = {
    "log_level": os.getenv("LOG_LEVEL", "INFO").upper(),
    "log_file": os.getenv(
        "LOG_FILE",
        "skill_extraction.log",
    ),
}