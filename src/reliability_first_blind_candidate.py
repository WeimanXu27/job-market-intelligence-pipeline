"""Hybrid job-skill extraction pipeline.

Uses Hugging Face NER and batched OpenAI LLM extraction to identify skill
phrases from job descriptions and write dashboard-ready results to a database.
"""

import json
import logging
import re
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from openai import OpenAI
from sqlalchemy import URL, create_engine, text
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline,
)

from settings import (
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

# Per-record evidence budget sent to the LLM after skill-focused filtering.
# The full batch is still protected by max_prompt_chars and recursive splitting.
MAX_CANDIDATE_TEXT_CHARS = 8000
MAX_EVIDENCE_CHUNK_CHARS = 3200
MAX_EVIDENCE_SEGMENT_CHARS = 1600

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
log_file = LOGGING_CONFIG.get(
    "log_file",
    "skill_extraction.log",
)
log_level = getattr(logging, LOGGING_CONFIG.get("log_level", "INFO"))

logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Log file: {log_file}")


class SkillsExtractor:
    """Extract job skills using NER candidate generation and batched LLM validation."""

    def __init__(self, batch_size: Optional[int] = None):
        """
        Initialize the extractor.

        Args:
            batch_size: Optional override for the logical LLM batch size.
                If omitted, the value from PROCESSING_CONFIG is used.
        """
        self.db_engine = None
        self.ner_pipeline = None
        self.llm_client = None
        self.openai_model = MODEL_CONFIG.get(
            "llm_model",
            "gpt-4o-mini",
        )
        self.batch_size = (
            batch_size
            if batch_size is not None
            else int(PROCESSING_CONFIG.get("llm_batch_size", 10))
        )

        if self.batch_size <= 0:
            raise ValueError(
                "LLM batch size must be a positive integer."
            )
        
        self.ner_confidence_threshold = float(
            PROCESSING_CONFIG.get("confidence_threshold", 0.5)
        )

        if not 0.0 <= self.ner_confidence_threshold <= 1.0:
            raise ValueError(
                "NER confidence threshold must be between 0.0 and 1.0."
            )

        self.max_prompt_chars = int(
            PROCESSING_CONFIG.get("max_prompt_chars", 12000)
        )

        if self.max_prompt_chars <= 0:
            raise ValueError(
                "Maximum prompt length must be a positive integer."
            )
        
        self.debug_llm_empty = bool(
            PROCESSING_CONFIG.get("debug_llm_empty", False)
        )

        self.debug_llm_empty_max = int(
            PROCESSING_CONFIG.get("debug_llm_empty_max", 5)
        )

        if self.debug_llm_empty_max < 0:
            raise ValueError(
                "LLM empty-debug limit must be zero or a positive integer."
            )
        
        self.record_limit = int(
            PROCESSING_CONFIG.get("record_limit", 0)
        )

        if self.record_limit < 0:
            raise ValueError(
                "Record limit must be zero or a positive integer."
            )

        self._init_database()
        self._init_ner_model()
        self._init_llm_client()

    # ----------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------
    def _init_database(self):
        """Initialize database connection."""
        try:
            print("  Connecting to database...")
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
            print("  ✓ Database connection successful")
            logger.info("Database connection successful")

        except Exception as e:
            print(f"  ✗ Database connection failed: {e}")
            logger.error(f"Database connection failed: {e}")
            raise

    def _init_ner_model(self):
        """Initialize Hugging Face NER pipeline."""
        try:
            model_name = MODEL_CONFIG['ner_model']
            print("  Loading NER model...")
            print(f"    Model: {model_name}")
            logger.info(f"Loading NER model: {model_name}")

            device = 0 if torch.cuda.is_available() else -1
            device_str = 'GPU' if device >= 0 else 'CPU'
            print(f"    Device: {device_str}")

            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            self.ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=device
            )
            print("  ✓ NER model loaded successfully")
            logger.info("NER model loaded successfully")

        except Exception as e:
            print(f"  ✗ Failed to load NER model: {e}")
            logger.error(f"Failed to load NER model: {e}")
            raise

    def _init_llm_client(self):
        """Initialize the OpenAI client used for contextual skill extraction."""
        print("  Initializing LLM client...")

        if not OPENAI_API_KEY:
            print("  ⚠️ OPENAI_API_KEY not found. LLM disabled; using NER only.")
            logger.warning(
                "OPENAI_API_KEY not found. LLM disabled; using NER only."
            )
            self.llm_client = None
            return

        try:

            self.llm_client = OpenAI(api_key=OPENAI_API_KEY)

            print(f"  ✓ LLM client initialized. Model: {self.openai_model}")
            logger.info(
                f"LLM client initialized. Model: {self.openai_model}"
            )

        except Exception as e:
            print(f"  ✗ Failed to initialize LLM client: {e}")
            logger.error(f"Failed to initialize LLM client: {e}")
            self.llm_client = None

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _safe_id(self, val: Any) -> str:
        """Safely stringify record IDs for logging."""
        try:
            if isinstance(val, bytes):
                return val.hex()
            return str(val)
        except Exception:
            return "unknown"

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning for NER."""
        if not text:
            return ""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Keep common punctuation and technical symbols such as /, +, and #
        text = re.sub(r'[^\w\s\-.,()/+#]', '', text)
        return text.strip()

    def _strip_noise_words(self, skill: str) -> str:
        """Remove common noise prefixes from skill strings."""
        noise_words = ["skill", "skills:", "skill:", "additional", "additional:"]
        s = skill.strip()
        s_low = s.lower()
        for w in noise_words:
            if s_low.startswith(w):
                s = s[len(w):].strip(" -:;")
                break
        return s

    def _skill_dedup_key(self, skill: str) -> str:
        """Build a conservative normalization key for near-duplicate skills."""
        s = re.sub(r"\s+", " ", skill.strip().lower())
        s = s.replace("&", "and")

        # Remove generic wording that often creates duplicate labels without
        # changing the underlying skill concept.
        for suffix in (" skills", " skill", " proficiency"):
            if s.endswith(suffix):
                s = s[:-len(suffix)].strip()
                break

        aliases = {
            "ml": "machine learning",
            "ms office": "microsoft office",
            "microsoft office suite": "microsoft office",
        }
        return aliases.get(s, s)

    def _deduplicate_skill_phrases(self, skills: List[str]) -> List[str]:
        """Remove conservative near-duplicates while preserving stable order."""
        canonical_display = {
            "machine learning": "Machine Learning",
            "microsoft office": "Microsoft Office",
        }

        grouped: Dict[str, List[str]] = {}
        order: List[str] = []

        for skill in skills:
            key = self._skill_dedup_key(skill)
            if not key:
                continue
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(skill)

        final: List[str] = []
        for key in order:
            if key in canonical_display:
                final.append(canonical_display[key])
                continue

            candidates = grouped[key]

            # Prefer the most concise wording when variants differ only by
            # generic suffixes such as "skills" or "proficiency".
            best = min(
                candidates,
                key=lambda value: (
                    len(value.split()),
                    len(value),
                    value.lower(),
                ),
            )
            final.append(best)

        return final

    def _select_more_complete_grounded_pass(
        self,
        primary_skills: List[str],
        confirmation_skills: List[str],
    ) -> List[str]:
        """Select the more complete validated pass without union inflation.

        Patch-29 v3 merged two independently grounded passes. That improved
        recall but created systematic duplicate-like inflation when both passes
        expressed the same source concepts with different wording.

        Patch-29 v4/v5 keep the independent confirmation but treat it as a
        best-of-two reliability mechanism:
        - validate and deduplicate each pass independently;
        - keep the pass with more grounded unique concepts;
        - on a tie, keep the original batch pass for stability.

        This does NOT impose a minimum skill count and does NOT use
        occupation-specific rules.
        """
        primary = self._deduplicate_skill_phrases(
            primary_skills or []
        )
        confirmation = self._deduplicate_skill_phrases(
            confirmation_skills or []
        )

        if not primary:
            return confirmation

        if not confirmation:
            return primary

        if len(confirmation) > len(primary):
            return confirmation

        return primary

    def _repair_json_to_dict(self, content: str) -> Dict[str, Any]:
        """
        Attempt to repair and parse a JSON dictionary from LLM output.

        Expected format:
        {"id1": ["skillA", ...], "id2": [...]}
        """
        content = content.strip()
        logger.debug(
            "Received LLM output with %s characters.",
            len(content),
        )

        # 1) Try direct JSON parsing
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 2) Extract the outermost JSON object
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            sub = content[start:end + 1]
            try:
                data = json.loads(sub)
                if isinstance(data, dict):
                    return data
            except Exception:
                content = sub

        # 3) Apply simple repairs for single quotes and trailing commas
        repaired = content.replace("'", '"')
        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)

        try:
            data = json.loads(repaired)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 4) Salvage fragments that resemble "key": [...]
        pattern = r'"[^"]+"\s*:\s*\[[^\]]*\]'
        matches = re.findall(pattern, repaired)
        if matches:
            obj_str = "{%s}" % ",".join(matches)
            try:
                data = json.loads(obj_str)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        logger.error("JSON repair failed. Returning an empty dictionary.")
        return {}

    def _validate_llm_skill_map(
        self, 
        skill_map: Dict[str, Any], 
        expected_ids: List[str],
    ) -> Dict[str, List[str]]:
        """Validate and lightly clean LLM skill output for expected record IDs."""
        if not isinstance(skill_map, dict):
            return {rid: [] for rid in expected_ids}

        # Filter abstract category labels rather than concrete skills.
        category_noise = {
            "programming language", "programming languages",
            "cloud", "database", "databases",
            "framework", "frameworks",
            "tools", "tool",
            "technology", "technologies",
            "software",
            "data", "analytics", "analysis",
            "ability", "opportunities", "ambitious",
            "relationship", "development", "proficiency",
            "reliability", "efficiency", "customer satisfaction",
        }

        out: Dict[str, List[str]] = {}
        for rid in expected_ids:
            v = skill_map.get(rid, [])
            if not isinstance(v, list):
                out[rid] = []
                continue

            cleaned: List[str] = []
            for item in v:
                if not isinstance(item, str):
                    continue

                s = re.sub(
                    r"^\s*\d+[.)]\s*",
                    "",
                    item,
                ).strip()
                s = re.sub(r"\s+", " ", s)

                if not s:
                    continue

                # Reject malformed tokenizer/subword fragments if the model
                # echoes a noisy NER hint.
                if "##" in s:
                    continue

                s_low = s.lower().strip()
                if s_low in category_noise:
                    continue

                # Conservative, domain-agnostic exclusions for physical or
                # eligibility conditions that are not skills/credentials.
                non_skill_patterns = [
                    r"^(?:auto|vehicle) insurance(?: coverage)?$",
                    r"^(?:reliable )?(?:personal )?(?:automobile|vehicle) transportation$",
                    r"^transportation reliability$",
                    r"^(?:lifting|lift)(?: products| boxes| objects)?(?: up to| from)?\s*\d+.*(?:lb|lbs|pounds).*$",
                    r"^physical stamina$",
                ]

                if any(
                    re.fullmatch(
                        pattern,
                        s_low,
                    )
                    for pattern in non_skill_patterns
                ):
                    continue

                cleaned.append(s)

            out[rid] = self._deduplicate_skill_phrases(cleaned)

        return out

    def _add_numbering(self, skills: List[str]) -> List[str]:
        """
        Add numbering prefix (1. 2. 3. ...) to each skill.
        Example: ["Python", "SQL"] -> ["1. Python", "2. SQL"]
        """
        numbered = []
        for idx, skill in enumerate(skills, start=1):
            numbered.append(f"{idx}. {skill}")
        return numbered

    # ----------------------------------------------------------------------
    # NER + sentence filtering
    # ----------------------------------------------------------------------
    def extract_skills_with_ner(self, text: str) -> List[str]:
        """Extract skill candidates using the NER model."""
        try:
            if not self.ner_pipeline:
                return []
            clean_text = self._clean_text(text)
            if not clean_text:
                return []

            results = self.ner_pipeline(clean_text)
            skills = []

            for r in results:
                if r['score'] > self.ner_confidence_threshold:
                    w = r['word'].strip()

                    # Drop very short or non-alphanumeric tokens.
                    if not w or len(w) <= 1:
                        continue
                    
                    if not re.search(r"[A-Za-z0-9]", w):
                        continue
                    
                    skills.append(w)

            return list(dict.fromkeys(skills))

        except Exception as e:
            logger.error(f"NER extraction failed: {e}")
            logger.error(traceback.format_exc())
            return []

    def extract_skill_candidate_sentences(self, text: str, ner_skills: List[str]) -> str:
        """Build broad, structurally labeled evidence for LLM skill extraction.

        This stage is intentionally domain-agnostic. It does not use industry-
        specific keyword lists to decide which sentences survive.

        Strategy:
        - Remove the scraped posting header before the first "Job description"
          marker so title/company/location metadata does not anchor extraction.
        - Preserve nearly all job-relevant body text.
        - Use generic section structure to exclude benefits, compensation,
          company-profile, EEO, and physical-demand boilerplate.
        - Label retained evidence as GENERAL, RESPONSIBILITY, or QUALIFICATION
          so the LLM can distinguish work activities from requirements.
        - Preserve only the applicable "<Region> Applicants:" block when the
          posting jurisdiction can be inferred from the original opening text.
        - For long descriptions, balance qualification and responsibility
          evidence before adding distributed coverage from the remaining body.
        """
        if not text:
            return ""

        normalized_text = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        # Keep the original opening context for jurisdiction inference, but
        # remove scraper/page header metadata from the evidence given to the LLM.
        body_text = normalized_text
        job_description_match = re.search(
            r"\bjob description\b",
            normalized_text,
            flags=re.IGNORECASE,
        )

        # Treat "Job description" as scraped page chrome only when it appears
        # near the beginning of the posting. The phrase can also occur later
        # inside legal/disclaimer text ("this job description..."). Slicing at
        # a late occurrence can discard the actual role content.
        if (
            job_description_match is not None
            and job_description_match.start() <= 500
        ):
            body_text = normalized_text[
                job_description_match.end():
            ]

        # Normalize common list bullets before section parsing. Job-board
        # descriptions frequently flatten bullets into characters such as "·".
        # Treat them as structural separators so benefits and requirements do
        # not collapse into one giant segment.
        segments = re.split(
            r'[.!?]\s+|\n+|[\*\-\•\·\▪\◦]\s+',
            body_text,
        )
        segments = [
            re.sub(r"\s+", " ", segment).strip()
            for segment in segments
            if segment and segment.strip()
        ]

        if not segments:
            return body_text[:MAX_CANDIDATE_TEXT_CHARS].strip()

        qualification_pattern = re.compile(
            r"\b("
            r"qualifications?|requirements?|required|preferred|"
            r"minimum qualifications?|minimum requirements?|"
            r"education|experience|knowledge|skills?|"
            r"proficien(?:t|cy)|certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?|degree"
            r")\b",
            flags=re.IGNORECASE,
        )

        responsibility_pattern = re.compile(
            r"\b("
            r"responsibilit(?:y|ies)|duties|functions|"
            r"accountabilit(?:y|ies)|job overview|role overview|"
            r"position summary|role summary|what .{0,24} do"
            r")\b",
            flags=re.IGNORECASE,
        )

        excluded_pattern = re.compile(
            r"\b("
            r"benefits?|perks?|compensation|salary|pay range|"
            r"what we offer|about us|about the company|company description|"
            r"equal opportunity|eeo|physical demands?|physical requirements?|"
            r"working conditions?|report about|report working at"
            r")\b",
            flags=re.IGNORECASE,
        )

        credential_pattern = re.compile(
            r"\b("
            r"certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?|"
            r"degree|diploma|ged|associate|bachelor|master|ph\.?d|"
            r"cpr|bls|pals"
            r")\b",
            flags=re.IGNORECASE,
        )

        candidate_qualification_signal_pattern = re.compile(
            r"\b("
            r"(?:at\s+least\s+|minimum\s+of\s+)?\d+\+?\s*(?:years?|yrs?)"
            r"(?:\s+of)?\s+experience|"
            r"experience\s+(?:in|with|using|working)|"
            r"required|preferred|must|should|"
            r"ability\s+to|"
            r"proficien(?:t|cy)|"
            r"knowledge\s+of|"
            r"familiarity\s+with|"
            r"degree|diploma|ged|"
            r"certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?"
            r")\b",
            flags=re.IGNORECASE,
        )

        def _heading_info(
            value: str,
        ) -> Tuple[bool, Optional[str]]:
            """Identify generic section boundaries without company-specific labels."""
            stripped = value.strip()
            low = stripped.lower()

            colon_position = stripped.find(":")
            prefix = (
                stripped[:colon_position].strip()
                if 0 <= colon_position <= 120
                else stripped
            )

            prefix_words = prefix.split()

            looks_like_heading = bool(
                (
                    0 <= colon_position <= 120
                    and 1 <= len(prefix_words) <= 12
                )
                or (
                    len(stripped) <= 100
                    and len(prefix_words) <= 10
                    and stripped == stripped.strip(" -:")
                )
            )

            strong_patterns = [
                (
                    "qualification",
                    qualification_pattern,
                ),
                (
                    "responsibility",
                    responsibility_pattern,
                ),
                (
                    "excluded",
                    excluded_pattern,
                ),
            ]

            if looks_like_heading:
                prefix_low = prefix.lower()

                if excluded_pattern.search(prefix_low):
                    return True, "excluded"
                if qualification_pattern.search(prefix_low):
                    return True, "qualification"
                if responsibility_pattern.search(prefix_low):
                    return True, "responsibility"

                # Unknown headings terminate any sticky section state.
                return True, None

            for section_name, pattern in strong_patterns:
                match = pattern.search(low[:180])
                if match is None:
                    continue

                matched_text = match.group(0).lower()
                weak_inline_terms = {
                    "required",
                    "requirements",
                    "experience",
                    "education",
                    "skills",
                    "salary",
                    "compensation",
                    "benefits",
                }

                if matched_text in weak_inline_terms:
                    continue

                return True, section_name

            return False, None

        def _label(
            segment: str,
            section: Optional[str],
        ) -> str:
            """Attach a generic structural label used by the LLM prompts."""
            if section == "qualification":
                return f"[QUALIFICATION] {segment}"
            if section == "responsibility":
                return f"[RESPONSIBILITY] {segment}"
            return f"[GENERAL] {segment}"

        jurisdiction_labels = list(dict.fromkeys(
            re.findall(
                r"\b([A-Z][A-Za-z]+)\s+Applicants\s*:",
                normalized_text,
            )
        ))

        opening_context = normalized_text[:1500].lower()

        current_jurisdiction = next(
            (
                label
                for label in jurisdiction_labels
                if re.search(
                    rf"\b{re.escape(label.lower())}\b",
                    opening_context,
                )
            ),
            None,
        )

        retained: List[str] = []
        qualification_evidence: List[str] = []
        responsibility_evidence: List[str] = []

        active_section: Optional[str] = None
        jurisdiction_allowed = True

        for segment in segments:
            low = segment.lower()

            jurisdiction_match = re.match(
                r"^\s*([a-z][a-z]+)\s+applicants\s*:",
                low,
            )

            if jurisdiction_match and jurisdiction_labels:
                region_label = jurisdiction_match.group(1)

                jurisdiction_allowed = (
                    current_jurisdiction is None
                    or region_label == current_jurisdiction.lower()
                )

                if not jurisdiction_allowed:
                    continue

                active_section = "qualification"
                labeled = _label(
                    segment,
                    active_section,
                )
                retained.append(labeled)
                qualification_evidence.append(labeled)
                continue

            if not jurisdiction_allowed:
                continue

            is_heading, heading_section = _heading_info(
                segment
            )

            if is_heading:
                active_section = heading_section

                if heading_section == "excluded":
                    continue

            if active_section == "excluded":
                # Do not allow a prior benefits/company section to hide a later
                # flattened Requirements/Responsibilities marker.
                explicit_candidate_marker = re.search(
                    r"\b("
                    r"requirements?|qualifications?|"
                    r"responsibilities?|duties"
                    r")\s*:",
                    low,
                    flags=re.IGNORECASE,
                )

                if explicit_candidate_marker is None:
                    continue

                marker = (
                    explicit_candidate_marker.group(1)
                    .lower()
                )

                active_section = (
                    "qualification"
                    if marker.startswith(
                        (
                            "requirement",
                            "qualification",
                        )
                    )
                    else "responsibility"
                )

            inferred_section = active_section

            if inferred_section is None:
                if (
                    credential_pattern.search(low)
                    or candidate_qualification_signal_pattern.search(
                        low
                    )
                ):
                    inferred_section = "qualification"
                elif responsibility_pattern.search(low):
                    inferred_section = "responsibility"

            labeled = _label(
                segment,
                inferred_section,
            )
            retained.append(labeled)

            if inferred_section == "qualification":
                qualification_evidence.append(labeled)

            if inferred_section == "responsibility":
                responsibility_evidence.append(labeled)

        retained = list(dict.fromkeys(retained))
        qualification_evidence = list(
            dict.fromkeys(
                qualification_evidence
            )
        )
        responsibility_evidence = list(
            dict.fromkeys(
                responsibility_evidence
            )
        )

        if not retained:
            return ""

        retained_text = "\n".join(retained)

        if len(retained_text) <= MAX_CANDIDATE_TEXT_CHARS:
            return retained_text.strip()

        selected: List[str] = []

        def _append_unique(
            candidates: List[str],
        ) -> None:
            for candidate in candidates:
                if candidate not in selected:
                    selected.append(candidate)

        # Opening body context, without the scraped title/company header.
        _append_unique(
            retained[:6]
        )

        # Balance qualifications and responsibilities.
        max_priority_len = max(
            len(qualification_evidence),
            len(responsibility_evidence),
        )

        for idx in range(max_priority_len):
            if idx < len(qualification_evidence):
                _append_unique(
                    [qualification_evidence[idx]]
                )

            if idx < len(responsibility_evidence):
                _append_unique(
                    [responsibility_evidence[idx]]
                )

        # Distributed anchors preserve coverage across long descriptions.
        anchor_count = min(
            12,
            len(retained),
        )

        if anchor_count > 1:
            anchor_indices = sorted({
                round(
                    idx
                    * (len(retained) - 1)
                    / (anchor_count - 1)
                )
                for idx in range(anchor_count)
            })
        else:
            anchor_indices = [0]

        _append_unique(
            [
                retained[idx]
                for idx in anchor_indices
            ]
        )

        # Fill the remaining evidence budget in original order.
        _append_unique(
            retained
        )

        budgeted: List[str] = []
        current_length = 0

        for segment in selected:
            separator_length = (
                1
                if budgeted
                else 0
            )

            next_length = (
                current_length
                + separator_length
                + len(segment)
            )

            if next_length <= MAX_CANDIDATE_TEXT_CHARS:
                budgeted.append(segment)
                current_length = next_length
                continue

            remaining = (
                MAX_CANDIDATE_TEXT_CHARS
                - current_length
                - separator_length
            )

            if remaining >= 120:
                budgeted.append(
                    segment[:remaining].rstrip()
                )

            break

        return "\n".join(budgeted).strip()


    # ----------------------------------------------------------------------
    # LLM batch processing
    # ----------------------------------------------------------------------
    def _build_llm_prompt(self, records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[str], int]:
        """
        Build chat messages for batched LLM skill extraction.

        Args:
            records: Records containing an ID, filtered text, and NER skill hints.

        Returns:
            A tuple of chat messages, input record IDs, and prompt length in characters.
        """
        SYSTEM_PROMPT = (
            "You are an information extraction engine. Your job is to extract SKILL PHRASES from job descriptions.\n\n"
            "Definition:\n"
            "A “skill phrase” is a concrete, human-readable capability, technology, method, tool, platform, standard, "
            "framework, credential, domain practice, or workflow requirement that a candidate is expected to know or do.\n\n"
            "You MUST:\n"
            "- Output ONLY concrete skill phrases (tools, methods, technologies, workflows that imply capability).\n"
            "- Never output abstract categories (e.g., “programming language”, “cloud”, “databases”) as skills.\n"
            "- Only include skills supported by the provided text (or strongly implied by a requirement statement).\n"
            "- Prefer canonical, short noun phrases (no sentences).\n"
            "- Deduplicate case-insensitively and remove near-duplicates.\n\n"
            "You MUST NOT:\n"
            "- Invent skills not evidenced in the text.\n"
            "- Include company names, locations, benefits, compensation, EEO statements, schedules, travel, or visa.\n"
            "- Include vague traits unless explicitly required as a soft skill.\n\n"
            "Taxonomy (categories you may extract from, but DO NOT output category names):\n"
            "1) Technical: tools/platforms, programming languages, libraries/frameworks, cloud/services, databases/warehouses, "
            "operating systems, networking, security, MLOps/DevOps, data/ML methods, analytics/BI tools, APIs/integrations.\n"
            "2) Domain & workflow: domain knowledge, business/functional processes, compliance/regulations, governance/quality practices.\n"
            "3) Certifications & credentials: named certifications, licenses, degrees or formal qualifications.\n"
            "4) Soft skills: communication, stakeholder management, leadership, collaboration, problem-solving, writing/presentation "
            "(only if explicitly required).\n\n"
            "Coverage rule:\n"
            "- Aim to extract a reasonably complete set of skills mentioned in requirements/responsibilities.\n"
            "- If the text is sparse, extract what is present; do not hallucinate.\n\n"
            "Output contract:\n"
            "Return STRICT JSON only, with keys exactly matching the provided IDs and values as arrays of strings.\n"
        )

        USER_PREFIX = (
            "Extract skills for each record.\n\n"
            "For each item:\n"
            "- Use item.text as primary evidence.\n"
            "- Use item.ner as helpful hints (candidate terms), but do not blindly copy noise.\n"
            "- Extract concrete skill phrases only (no category names).\n"
            "- Keep each skill 1–6 words when possible.\n"
            "- Prefer the form people put on resumes (noun phrases).\n"
            "- Do not target a minimum skill count. Precision is more important than filling a quota; return only skills clearly supported by the evidence.\n"
            "- Do not include duplicates or near-duplicates. If an acronym and full form mean the same skill, return one canonical form.\n"
            "- Prefer the concrete skill name over generic wrappers such as 'skills' or 'proficiency'.\n"
            "- Never output tokenizer fragments such as terms beginning with '##'.\n"
            "- Do not output job titles, employer names, benefits, outcomes, business goals, or generic nouns as skills.\n"
            "- A role title or occupational label by itself is not a skill.\n"
            "- Employer service offerings, business lines, company capabilities, employee support programs, mentoring programs, and training benefits are not candidate skills unless the text explicitly states that the candidate must perform or know them.\n"
            "- If requirements differ by state, region, location, or business unit, extract only the requirements that clearly apply to this posting; ignore alternatives explicitly labeled for other locations.\n"
            "- Evidence lines may be labeled [RESPONSIBILITY], [QUALIFICATION], or [GENERAL]. Use the labels as structure, not as output text.\n"
            "- Scan every [RESPONSIBILITY] line for repeatable work capabilities, workflows, methods, tools, systems, or domain practices. Do not ignore responsibility evidence just because qualification evidence is also present.\n"
            "- Capture concrete methods, workflows, certifications, tools, and named technical practices from both responsibilities and qualifications.\n"
            "- Explicitly required licenses, certifications, degrees, and named credentials take priority over generic traits or soft skills.\n"
            "- Do not treat employer-provided education programs, tuition benefits, optional development offerings, or benefit eligibility language as candidate skills or credentials.\n\n"
            "Return STRICT JSON:\n"
            "{\n"
            "  \"<id1>\": [\"skill phrase\", ...],\n"
            "  \"<id2>\": [\"skill phrase\", ...]\n"
            "}\n\n"
            "Items:\n"
        )

        input_payload = []
        input_ids = []
        for r in records:
            rid = str(r["id"])
            input_ids.append(rid)

            ner = r.get("ner", [])
            if not isinstance(ner, list):
                ner = []
            # Cap NER hints to control prompt size and reduce noisy anchoring.
            ner = [str(x).strip() for x in ner if str(x).strip()]
            ner = ner[:40]

            input_payload.append({
                "id": rid,
                "text": (r.get("text", "") or "")[:MAX_CANDIDATE_TEXT_CHARS],
                "ner": ner,
            })

        user_content = USER_PREFIX + json.dumps(input_payload, ensure_ascii=False)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        prompt_len = len(SYSTEM_PROMPT) + len(user_content)
        return messages, input_ids, prompt_len

    def batch_llm_extract_skills(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """
        Extract skills from job text using batched LLM inference.

        Args:
            records: Records containing IDs, filtered job text, and NER skill hints.

        Returns:
            A mapping from record IDs to extracted skill lists.
        """
    
        if not records:
            return {}

        max_len = self.max_prompt_chars

        messages, input_ids, prompt_len = self._build_llm_prompt(records)

        # Split oversized batches recursively so every record is still processed.
        if prompt_len > max_len and len(records) > 1:
            midpoint = max(1, len(records) // 2)

            left_records = records[:midpoint]
            right_records = records[midpoint:]

            logger.info(
                "LLM prompt exceeds limit (%s > %s characters); "
                "splitting batch into %s and %s records.",
                prompt_len,
                max_len,
                len(left_records),
                len(right_records),
            )

            left_results = self.batch_llm_extract_skills(left_records)
            right_results = self.batch_llm_extract_skills(right_records)

            return {
                **left_results,
                **right_results,
            }

        if prompt_len > max_len:
            logger.warning(
                "Single-record prompt exceeds the configured limit "
                "(%s > %s characters); processing the truncated input.",
                prompt_len,
                max_len,
            )

        logger.info(
            f"LLM processing records: {len(input_ids)} "
            f"(batch size: {len(records)})"
        )
        logger.debug(f"LLM prompt length: {prompt_len} characters")

        final_out: Dict[str, List[str]] = {
            rid: [] for rid in input_ids
        }

        response_text = ""

        try:
            if self.llm_client is None:
                logger.warning(
                    "LLM client is not initialized. Skipping LLM call."
                )
                return final_out

            resp = self.llm_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0,
            )

            response_text = resp.choices[0].message.content or ""

            usage = getattr(resp, "usage", None)
            if usage is not None:
                logger.info(
                    "OpenAI usage - prompt_tokens: %s, "
                    "completion_tokens: %s, total_tokens: %s",
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    getattr(usage, "total_tokens", None),
                )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            logger.error(traceback.format_exc())
            return final_out

        if not response_text.strip():
            logger.warning("LLM returned empty content.")
            return final_out
        
        skill_map_raw = self._repair_json_to_dict(response_text)

        validated = self._validate_llm_skill_map(skill_map_raw, input_ids)

        for rid in input_ids:
            final_out[rid] = validated.get(rid, [])

        if self.debug_llm_empty:
            empty_count = 0
            rec_map = {
                str(r.get("id")): r
                for r in records
                if r.get("id") is not None
            }

            for rid in input_ids:
                if final_out[rid]:
                    continue

                empty_count += 1
                if empty_count > self.debug_llm_empty_max:
                    break

                r = rec_map.get(rid, {})

                ner_list = r.get("ner") or []
                if not isinstance(ner_list, list):
                    ner_list = []

                raw_v = (
                    skill_map_raw.get(rid)
                    if isinstance(skill_map_raw, dict)
                    else None
                )
                raw_item_count = len(raw_v) if isinstance(raw_v, list) else 0

                logger.warning(
                    f"[LLM_EMPTY_DEBUG] id={rid} | "
                    f"ner_skill_count={len(ner_list)}"
                )
                logger.warning(
                    f"[LLM_EMPTY_DEBUG] id={rid} | "
                    f"raw_item_count={raw_item_count}"
                )
                logger.warning(
                    f"[LLM_EMPTY_DEBUG] id={rid} | validated_skill_count=0"
                )

        return final_out


    def _build_llm_refinement_prompt(
        self,
        records: List[Dict[str, Any]],
        initial_skill_map: Dict[str, List[str]],
    ) -> Tuple[List[Dict[str, str]], List[str], int]:
        """Build a bounded semantic-validation prompt.

        The refinement pass is intentionally NOT allowed to invent or recover
        new skills. It can only keep or drop candidates produced by the first
        extraction pass. Candidates are addressed by integer index so the model
        cannot rewrite them into new unsupported phrases.
        """
        SYSTEM_PROMPT = (
            "You are a strict semantic validator for job-skill extraction.\\n\\n"
            "For each record you receive job evidence plus an indexed list of "
            "CANDIDATE skills produced by an earlier extraction pass.\\n\\n"
            "Your only task is to decide which candidate indices are valid. "
            "You MUST NOT add, rewrite, paraphrase, or recover new skills.\\n\\n"
            "KEEP a candidate only when the evidence clearly supports it as a "
            "candidate capability, technology, method, tool, platform, standard, "
            "workflow, domain practice, license, certification, degree, formal "
            "qualification, or explicitly required soft skill.\\n\\n"
            "DROP candidates that are merely:\\n"
            "- job titles or occupational labels;\\n"
            "- employer products, services, business lines, departments, markets, "
            "or company capabilities mentioned only as context;\\n"
            "- employee benefits, mentoring/support programs, optional training, "
            "professional-development offerings, or company-provided education;\\n"
            "- compensation, pay formulas, schedules, travel conditions, insurance "
            "coverage, physical requirements, or background-check information;\\n"
            "- outcomes, goals, vague traits, slogans, generic nouns, or unsupported "
            "inferences;\\n"
            "- duplicate or near-duplicate candidates where one clearly expresses "
            "the same skill as another.\\n\\n"
            "Important distinction:\\n"
            "- A bare occupation name is not a skill.\\n"
            "- A license/certification/degree that includes an occupation name is "
            "valid when the evidence explicitly requires that credential.\\n\\n"
            "Evidence lines may be labeled [RESPONSIBILITY], [QUALIFICATION], or "
            "[GENERAL]. Use those labels only as context.\\n\\n"
            "Return STRICT JSON only. Each key must be an input record ID and each "
            "value must be an array of 1-based candidate indices to KEEP.\\n"
        )

        USER_PREFIX = (
            "Validate the indexed candidate skills for each record.\\n\\n"
            "Rules:\\n"
            "1) Return indices only from that record's candidate list.\\n"
            "2) Do not add new skills.\\n"
            "3) If no candidate is valid, return an empty array.\\n"
            "4) Review responsibilities and qualifications, but judge only the "
            "candidates already supplied.\\n\\n"
            "Return STRICT JSON:\\n"
            "{\\n"
            '  "<id1>": [1, 3, 5],\\n'
            '  "<id2>": []\\n'
            "}\\n\\n"
            "Items:\\n"
        )

        input_payload = []
        input_ids: List[str] = []

        for record in records:
            rid = str(record["id"])
            input_ids.append(rid)

            draft = initial_skill_map.get(rid, [])
            if not isinstance(draft, list):
                draft = []

            draft = [
                str(skill).strip()
                for skill in draft
                if str(skill).strip()
            ]

            indexed_candidates = [
                {
                    "index": idx,
                    "skill": skill,
                }
                for idx, skill in enumerate(
                    draft,
                    start=1,
                )
            ]

            input_payload.append({
                "id": rid,
                "text": (
                    record.get("text", "") or ""
                )[:MAX_CANDIDATE_TEXT_CHARS],
                "candidates": indexed_candidates,
            })

        user_content = (
            USER_PREFIX
            + json.dumps(
                input_payload,
                ensure_ascii=False,
            )
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_len = (
            len(SYSTEM_PROMPT)
            + len(user_content)
        )

        return (
            messages,
            input_ids,
            prompt_len,
        )

    @staticmethod
    def _select_draft_by_indices(
        raw_value: Any,
        draft: List[str],
    ) -> Optional[List[str]]:
        """Map a refinement response back to an exact subset of the draft.

        Returns:
            A selected draft list when the response is structurally valid.
            None when the response is malformed and should trigger fallback.
        """
        if not isinstance(raw_value, list):
            return None

        selected: List[str] = []
        seen_indices = set()

        for raw_index in raw_value:
            if isinstance(raw_index, bool):
                return None

            if isinstance(raw_index, int):
                index = raw_index
            elif (
                isinstance(raw_index, str)
                and raw_index.strip().isdigit()
            ):
                index = int(
                    raw_index.strip()
                )
            else:
                return None

            if (
                index < 1
                or index > len(draft)
            ):
                return None

            if index in seen_indices:
                continue

            seen_indices.add(index)
            selected.append(
                draft[index - 1]
            )

        return selected

    def _retry_empty_refinement(
        self,
        record: Dict[str, Any],
        draft: List[str],
    ) -> List[str]:
        """Retry one unexpectedly empty validation result in isolation.

        The retry remains bounded: it may only choose from the existing draft.
        It is used to protect against batch-level over-deletion without silently
        accepting an invalid first-pass list.
        """
        rid = str(record["id"])

        messages, _, _ = (
            self._build_llm_refinement_prompt(
                [record],
                {
                    rid: draft,
                },
            )
        )

        if self.llm_client is None:
            return []

        logger.warning(
            "Refinement removed every candidate for record %s "
            "despite a non-trivial draft (%s candidates); "
            "retrying the validation in isolation.",
            rid,
            len(draft),
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0,
            )

            response_text = (
                resp.choices[0].message.content
                or ""
            )

        except Exception as e:
            logger.error(
                "Single-record refinement retry failed "
                "for %s: %s",
                rid,
                e,
            )
            return []

        if not response_text.strip():
            return []

        repaired = self._repair_json_to_dict(
            response_text
        )

        raw_value = (
            repaired.get(rid)
            if isinstance(
                repaired,
                dict,
            )
            else None
        )

        selected = self._select_draft_by_indices(
            raw_value,
            draft,
        )

        if selected is None:
            return []

        return selected

    def batch_llm_refine_skills(
        self,
        records: List[Dict[str, Any]],
        initial_skill_map: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """Validate first-pass skills without allowing additive refinement."""
        if not records:
            return {}

        messages, input_ids, prompt_len = (
            self._build_llm_refinement_prompt(
                records,
                initial_skill_map,
            )
        )

        if (
            prompt_len > self.max_prompt_chars
            and len(records) > 1
        ):
            midpoint = max(
                1,
                len(records) // 2,
            )

            left_records = records[:midpoint]
            right_records = records[midpoint:]

            logger.info(
                "LLM validation prompt exceeds limit "
                "(%s > %s characters); splitting batch "
                "into %s and %s records.",
                prompt_len,
                self.max_prompt_chars,
                len(left_records),
                len(right_records),
            )

            left_results = self.batch_llm_refine_skills(
                left_records,
                initial_skill_map,
            )
            right_results = self.batch_llm_refine_skills(
                right_records,
                initial_skill_map,
            )

            return {
                **left_results,
                **right_results,
            }

        fallback = {
            rid: list(
                initial_skill_map.get(
                    rid,
                    [],
                )
                or []
            )
            for rid in input_ids
        }

        if self.llm_client is None:
            return fallback

        logger.info(
            "LLM validating records: %s "
            "(batch size: %s)",
            len(input_ids),
            len(records),
        )

        try:
            resp = self.llm_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0,
            )

            response_text = (
                resp.choices[0].message.content
                or ""
            )

            usage = getattr(
                resp,
                "usage",
                None,
            )

            if usage is not None:
                logger.info(
                    "OpenAI validation usage - prompt_tokens: %s, "
                    "completion_tokens: %s, total_tokens: %s",
                    getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "completion_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "total_tokens",
                        None,
                    ),
                )

        except Exception as e:
            logger.error(
                f"LLM validation call failed: {e}"
            )
            logger.error(
                traceback.format_exc()
            )
            return fallback

        if not response_text.strip():
            logger.warning(
                "LLM validation returned empty content; "
                "keeping first-pass results."
            )
            return fallback

        repaired = self._repair_json_to_dict(
            response_text
        )

        record_lookup = {
            str(record["id"]): record
            for record in records
        }

        final_out: Dict[str, List[str]] = {}

        for rid in input_ids:
            draft = fallback[rid]

            raw_value = (
                repaired.get(rid)
                if isinstance(
                    repaired,
                    dict,
                )
                else None
            )

            selected = self._select_draft_by_indices(
                raw_value,
                draft,
            )

            if selected is None:
                logger.warning(
                    "Malformed or missing validation result "
                    "for record %s; keeping first-pass skills.",
                    rid,
                )
                final_out[rid] = draft
                continue

            # Empty output can be correct for sparse/non-job descriptions.
            # For a non-trivial draft, verify the deletion in a single-record
            # retry to reduce batch-level over-deletion risk.
            if (
                not selected
                and len(draft) >= 3
            ):
                retry_selected = (
                    self._retry_empty_refinement(
                        record_lookup[rid],
                        draft,
                    )
                )

                final_out[rid] = retry_selected
            else:
                final_out[rid] = selected

            logger.info(
                "Record %s validation: draft=%s, final=%s.",
                rid,
                len(draft),
                len(final_out[rid]),
            )

        return final_out


    @staticmethod
    def _build_role_content_preview(
        text: str,
        max_chars: int = 6000,
    ) -> str:
        """Build a distributed preview for the role-content gate.

        The gate only decides whether a posting contains substantive candidate
        duties/qualifications. It is not used to extract skills.
        """
        cleaned = re.sub(
            r"\s+",
            " ",
            str(text or ""),
        ).strip()

        if len(cleaned) <= max_chars:
            return cleaned

        half = max_chars // 2

        return (
            cleaned[:half]
            + "\n[... middle omitted for screening ...]\n"
            + cleaned[-half:]
        )

    def _build_role_content_screen_prompt(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], List[str], int]:
        """Build a conservative screening prompt for non-job-content records."""
        system_prompt = (
            "You are a conservative job-posting content screener.\\n\\n"
            "For each record, decide whether the text contains substantive "
            "candidate-facing ROLE CONTENT from which job skills can reasonably "
            "be extracted.\\n\\n"
            "Return true when the text contains at least one concrete duty, "
            "responsibility, required/preferred qualification, tool/technology, "
            "license/certification, education requirement, or role-specific "
            "experience requirement.\\n\\n"
            "Return false ONLY when the text is clearly not substantive role "
            "content, such as a privacy/legal notice, application disclosure, "
            "background-check disclosure, company marketing/profile only, benefits "
            "or compensation only, staffing/assignment logistics only, or a role "
            "title with no meaningful duties or qualifications.\\n\\n"
            "A company name or job title by itself is not role content. Generic "
            "claims about the employer are not role content.\\n\\n"
            "If the record mixes boilerplate with real duties or qualifications, "
            "return true. If uncertain, return true. False is reserved for clearly "
            "non-substantive postings.\\n\\n"
            "Return STRICT JSON only, with keys exactly matching the supplied IDs "
            "and boolean values true or false."
        )

        payload = []
        input_ids: List[str] = []

        for record in records:
            rid = str(record["id"])
            input_ids.append(rid)

            payload.append({
                "id": rid,
                "text": self._build_role_content_preview(
                    record.get(
                        "description",
                        "",
                    )
                ),
            })

        user_content = (
            "Screen every record. Return only a JSON object mapping each ID "
            "to true or false.\\n\\nItems:\\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_len = (
            len(system_prompt)
            + len(user_content)
        )

        return (
            messages,
            input_ids,
            prompt_len,
        )

    def batch_role_content_screen(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, bool]:
        """Screen records for substantive role content.

        Missing, malformed, or failed screening results default to True. This
        makes the gate conservative against false-negative data loss.
        """
        if not records:
            return {}

        messages, input_ids, prompt_len = (
            self._build_role_content_screen_prompt(
                records
            )
        )

        if (
            prompt_len > self.max_prompt_chars
            and len(records) > 1
        ):
            midpoint = max(
                1,
                len(records) // 2,
            )

            left = self.batch_role_content_screen(
                records[:midpoint]
            )
            right = self.batch_role_content_screen(
                records[midpoint:]
            )

            return {
                **left,
                **right,
            }

        fallback = {
            rid: True
            for rid in input_ids
        }

        if self.llm_client is None:
            return fallback

        try:
            response = (
                self.llm_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.0,
                )
            )

            response_text = (
                response.choices[0].message.content
                or ""
            )

            usage = getattr(
                response,
                "usage",
                None,
            )

            if usage is not None:
                logger.info(
                    "OpenAI role-content screening usage - "
                    "prompt_tokens: %s, completion_tokens: %s, "
                    "total_tokens: %s",
                    getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "completion_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "total_tokens",
                        None,
                    ),
                )

        except Exception as exc:
            logger.error(
                "Role-content screening failed: %s",
                exc,
            )
            return fallback

        if not response_text.strip():
            return fallback

        repaired = self._repair_json_to_dict(
            response_text
        )

        if not isinstance(
            repaired,
            dict,
        ):
            return fallback

        out = dict(fallback)

        for rid in input_ids:
            raw_value = repaired.get(
                rid,
            )

            if isinstance(
                raw_value,
                bool,
            ):
                out[rid] = raw_value
                continue

            if isinstance(
                raw_value,
                str,
            ):
                normalized = (
                    raw_value.strip().lower()
                )

                if normalized == "true":
                    out[rid] = True
                elif normalized == "false":
                    out[rid] = False

        return out


    def build_skill_evidence_chunks(
        self,
        text: str,
        ner_skills: List[str],
    ) -> List[str]:
        """Build full-description, domain-agnostic evidence chunks.

        Unlike extract_skill_candidate_sentences(), this method does not truncate
        a long posting to one evidence window. It keeps all retained job-relevant
        text and divides it into bounded chunks for LLM extraction.

        The cleaner remains structural rather than industry-specific:
        - strip a scraped "Job description" header only when it appears near the
          start of the posting;
        - exclude obvious benefits, compensation, company-profile, EEO, and
          physical-demand boilerplate;
        - preserve applicable regional requirement blocks;
        - attach GENERAL / RESPONSIBILITY / QUALIFICATION labels.
        """
        if not text:
            return []

        normalized_text = (
            text.replace("\r\n", "\n")
            .replace("\r", "\n")
        )

        body_text = normalized_text

        job_description_match = re.search(
            r"\bjob description\b",
            normalized_text,
            flags=re.IGNORECASE,
        )

        if (
            job_description_match is not None
            and job_description_match.start() <= 500
        ):
            body_text = normalized_text[
                job_description_match.end():
            ]

        # Restore section boundaries that are often flattened by job-board
        # HTML before sentence/bullet splitting. This keeps employer-offering
        # text separate from later candidate requirements.
        # Restore distinctive multi-word headings case-insensitively.
        inline_boundary_pattern = re.compile(
            r"(?i)(?<!\n)\s+(?=("
            r"what(?:'s| is) in it for you|"
            r"what does this mean for you|"
            r"what we (?:offer|provide|expect)|"
            r"what you can expect|"
            r"why (?:work|join) (?:with )?us|"
            r"employee benefits?|our benefits?|perks?|"
            r"pay transparency|"
            r"systems? you(?:['’]ll|\s+will)? use"
            r")\s*:)"
        )

        body_text = inline_boundary_pattern.sub(
            "\n",
            body_text,
        )

        # Restore ambiguous one-word headings only when they look like actual
        # capitalized job-board headings. This avoids splitting phrases such as
        # "customer requirements:" and "user requirements".
        capitalized_heading_pattern = re.compile(
            r"(?<!\n)\s+(?=("
            r"(?:REQUIREMENTS?|Requirements?)|"
            r"(?:QUALIFICATIONS?|Qualifications?)|"
            r"(?:REQUIRED\s+SKILLS\s+(?:AND|&)\s+EXPERIENCE|Required\s+Skills\s+(?:and|&)\s+Experience)|"
            r"(?:REQUIRED\s+EXPERIENCE\s+(?:AND|&)\s+SKILLS|Required\s+Experience\s+(?:and|&)\s+Skills)|"
            r"(?:RESPONSIBILITIES?|Responsibilities?)|"
            r"(?:DUTIES|Duties)"
            r")\s*:)"
        )

        body_text = capitalized_heading_pattern.sub(
            "\n",
            body_text,
        )

        # Some job boards flatten section headings without a colon into the
        # surrounding text. Restore those generic boundaries, including curly
        # apostrophes, before bullet/sentence splitting.
        flattened_heading_pattern = re.compile(
            r"(?i)(?<!\n)\s+("
            r"what\s+you(?:['’]ll|\s+will)?\s+do|"
            r"what\s+you(?:['’]ll|\s+will)?\s+get\s+to\s+do|"
            r"what\s+you(?:['’]ll|\s+will)?\s+get|"
            r"what\s+you(?:['’]ll|\s+will)?\s+need|"
            r"what\s+you\s+have|"
            r"what\s+will\s+set\s+you\s+apart|"
            r"about\s+the\s+role|"
            r"you(?:['’]re|\s+are)\s+excited\s+about\s+this\s+opportunity\s+because\s+you\s+will|"
            r"we(?:['’]re|\s+are)\s+excited\s+about\s+you\s+because|"
            r"what\s+we(?:['’]re|\s+are)\s+looking\s+for|"
            r"key\s+responsibilities?|"
            r"what\s+we\s+(?:offer|provide)|"
            r"our\s+benefits?|"
            r"company\s+description|"
            r"about\s+us|"
            r"about\s+the\s+(?:company|organization|employer|team)|"
            r"we\s+value\s+diversity|"
            r"pay\s+transparency|"
            r"systems?\s+you(?:['’]ll|\s+will)?\s+use|"
            r"position\s+overview|"
            r"job\s+summary"
            r")(?=\s)"
        )

        body_text = flattened_heading_pattern.sub(
            lambda match: (
                "\n"
                + match.group(1)
                + ":\n"
            ),
            body_text,
        )

        capitalized_flat_heading_pattern = re.compile(
            r"(?<!\n)\s+("
            r"(?:REQUIREMENTS?|Requirements?)(?!\s+(?:for|of|to)\b)|"
            r"(?:QUALIFICATIONS?|Qualifications?)|"
            r"(?:RESPONSIBILITIES?|Responsibilities?)|"
            r"(?:DUTIES|Duties)"
            r")(?=\s)"
        )

        body_text = capitalized_flat_heading_pattern.sub(
            lambda match: (
                "\n"
                + match.group(1)
                + ":\n"
            ),
            body_text,
        )

        # Flattened boards can append an employer-provided paid training section
        # directly to candidate qualifications. Restore that boundary so a later
        # employer-offering filter cannot suppress unrelated candidate skills.
        paid_training_heading_pattern = re.compile(
            r"(?i)(?<!\n)\s+("
            r"(?:\d+\s*[-–]\s*week\s+)?paid\s+training\s+program"
            r")\s*:\s*"
        )

        body_text = paid_training_heading_pattern.sub(
            lambda match: (
                "\n"
                + match.group(1)
                + ":\n"
            ),
            body_text,
        )

        # Normalize a terse but common responsibility heading used by
        # operational/coordination postings.
        system_owner_heading_pattern = re.compile(
            r"(?i)(?<!\n)\s+(system\s+owner\s+for)\s*:\s*"
        )

        body_text = system_owner_heading_pattern.sub(
            lambda match: (
                "\n"
                + match.group(1)
                + ":\n"
            ),
            body_text,
        )

        # Normalize common list bullets before section parsing.
        segments = re.split(
            r'[.!?]\s+|\n+|[\*\-\•\·\▪\◦]\s+',
            body_text,
        )

        segments = [
            re.sub(r"\s+", " ", segment).strip()
            for segment in segments
            if segment and segment.strip()
        ]

        if not segments:
            return []

        qualification_pattern = re.compile(
            r"\b("
            r"qualifications?|requirements?|required|preferred|"
            r"minimum qualifications?|minimum requirements?|"
            r"education|experience|knowledge|skills?|"
            r"proficien(?:t|cy)|certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?|degree"
            r")\b",
            flags=re.IGNORECASE,
        )

        responsibility_pattern = re.compile(
            r"\b("
            r"responsibilit(?:y|ies)|duties|functions|"
            r"accountabilit(?:y|ies)|job overview|role overview|"
            r"position summary|role summary|what .{0,24} do"
            r")\b",
            flags=re.IGNORECASE,
        )

        excluded_pattern = re.compile(
            r"\b("
            r"benefits?|perks?|compensation|salary|pay range|"
            r"what we offer|about us|about the company|company description|"
            r"equal opportunity|eeo|physical demands?|physical requirements?|"
            r"working conditions?|report about|report working at"
            r")\b",
            flags=re.IGNORECASE,
        )

        credential_pattern = re.compile(
            r"\b("
            r"certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?|"
            r"degree|diploma|ged|associate|bachelor|master|ph\.?d|"
            r"cpr|bls|pals"
            r")\b",
            flags=re.IGNORECASE,
        )

        candidate_qualification_signal_pattern = re.compile(
            r"\b("
            r"(?:at\s+least\s+|minimum\s+of\s+)?\d+\+?\s*(?:years?|yrs?)"
            r"(?:\s+of)?\s+experience|"
            r"experience\s+(?:in|with|using|working)|"
            r"background\s+(?:in|with)|"
            r"required|preferred|must|should|"
            r"ability\s+to|"
            r"proficien(?:t|cy)|"
            r"knowledge\s+of|"
            r"familiarity\s+with|"
            r"(?:strong|deep)\s+understanding\s+of|"
            r"degree|diploma|ged|"
            r"certif(?:ied|ication|ications)?|"
            r"licen[cs](?:e|ed|ure|ing)?|credentials?"
            r")\b",
            flags=re.IGNORECASE,
        )

        candidate_responsibility_signal_pattern = re.compile(
            r"(?:"
            r"\b(?:is|are)\s+responsible\s+for\b|"
            r"\byou(?:['’]ll|\s+will)\s*:?[\s-]*"
            r"(?:serve|perform|manage|lead|coach|mentor|implement|create|"
            r"partner|support|deliver|conduct|coordinate|research|schedule|"
            r"provide|assist|develop|maintain|track|prepare|inspect|operate|"
            r"analyze|review|build|sell|follow\s+up|organize|order|work)\b|"
            r"^\s*(?:design|develop|write|architect|implement|build|maintain|"
            r"conduct|deliver|own|handle|manage|lead|support|operate|inspect|"
            r"assemble|install|test|analyze|review|coordinate|prepare|perform|"
            r"assist|monitor|troubleshoot|configure|automate|collaborate)\b|"
            r"\b(?:the|this)\s+[a-z][a-z0-9 /&'’.-]{1,70}\s+will\s+"
            r"(?:serve|perform|manage|lead|coordinate|provide|assist|develop|"
            r"maintain|track|prepare|inspect|operate|analyze|review|build)\b"
            r")",
            flags=re.IGNORECASE,
        )

        def _heading_info(
            value: str,
        ) -> Tuple[bool, Optional[str]]:
            stripped = value.strip()
            low = stripped.lower()

            colon_position = stripped.find(":")
            prefix = (
                stripped[:colon_position].strip()
                if 0 <= colon_position <= 120
                else stripped
            )

            prefix_words = prefix.split()

            has_colon_heading = bool(
                0 <= colon_position <= 120
                and 1 <= len(prefix_words) <= 12
            )

            prefix_low = prefix.lower()

            # Common candidate-section and employer-offering headings are
            # structural patterns that occur across industries. They are not
            # occupation-specific skill rules.
            generic_basic_heading = re.match(
                r"^\s*(requirements?|qualifications?|responsibilities?|duties)\b",
                prefix_low,
            )

            if generic_basic_heading:
                marker = generic_basic_heading.group(1).lower()

                if marker.startswith(
                    (
                        "requirement",
                        "qualification",
                    )
                ):
                    return True, "qualification"

                return True, "responsibility"

            generic_responsibility_heading = re.match(
                r"^\s*("
                r"what\s+you(?:['’]ll|\s+will)?\s+(?:do|be doing|get\s+to\s+do)|"
                r"about\s+the\s+role|"
                r"you(?:['’]re|\s+are)\s+excited\s+about\s+this\s+opportunity\s+because\s+you\s+will|"
                r"what you can expect|"
                r"here(?:['’]s|\s+is)\s+more\s+of\s+what\s+the\s+position\s+entails|"
                r"when\s+you\s+join\s+our\s+team\s+as\s+.{1,80}?\s+you\s+will|"
                r"how\s+you\s+will\s+make\s+a\s+difference|"
                r"you(?:['’]ll|\s+will)\s+make\s+a\s+difference\s+by|"
                r"system\s+owner\s+for|"
                r"day[- ]to[- ]day|"
                r"key responsibilities?|"
                r"essential functions?|"
                r"position overview|"
                r"job summary"
                r")\b",
                prefix_low,
            )

            generic_qualification_heading = re.match(
                r"^\s*("
                r"what\s+you(?:['’]ll|\s+will)?\s+need|"
                r"what\s+you\s+have|"
                r"what\s+will\s+set\s+you\s+apart|"
                r"we(?:['’]re|\s+are)\s+excited\s+about\s+you\s+because|"
                r"what\s+we(?:['’]re|\s+are)\s+looking\s+for|"
                r"required\s+skills\s+(?:and|&)\s+experience|"
                r"required\s+experience\s+(?:and|&)\s+skills|"
                r"skills\s+(?:and|&)\s+experience|"
                r"systems?\s+you(?:['’]ll|\s+will)?\s+use"
                r")\b",
                prefix_low,
            )

            generic_excluded_heading = re.match(
                r"^\s*("
                r"what\s+you(?:['’]ll|\s+will)?\s+get|"
                r"what we (?:offer|provide)|"
                r"why (?:work|join) (?:with )?us|"
                r"what(?:['’]s| is) in it for you|"
                r"what does this mean for you|"
                r"employee benefits?|"
                r"our benefits?|"
                r"perks?|"
                r"company description|"
                r"about us|"
                r"about the (?:company|organization|employer|team)|"
                r"we value diversity|"
                r"pay transparency|"
                r"(?:\d+\s*[-–]\s*week\s+)?paid\s+training\s+program"
                r")\b",
                prefix_low,
            )

            if generic_excluded_heading:
                return True, "excluded"

            if generic_qualification_heading:
                return True, "qualification"

            if generic_responsibility_heading:
                return True, "responsibility"

            if has_colon_heading:
                # Candidate-facing section cues take precedence when a heading
                # also contains an ambiguous domain word such as "benefits".
                if qualification_pattern.search(prefix_low):
                    return True, "qualification"
                if responsibility_pattern.search(prefix_low):
                    return True, "responsibility"
                if excluded_pattern.search(prefix_low):
                    return True, "excluded"

                # An unknown colon heading still ends the previous sticky state,
                # but an ordinary short bullet no longer does.
                return True, None

            # Heading text without a colon is accepted only when it contains a
            # recognized section concept. This preserves headings such as
            # "Requirements" or "Key Responsibilities" while avoiding the old
            # behavior where every short bullet reset section state.
            if (
                len(stripped) <= 100
                and len(prefix_words) <= 10
            ):
                if excluded_pattern.fullmatch(prefix_low):
                    return True, "excluded"
                if qualification_pattern.fullmatch(prefix_low):
                    return True, "qualification"
                if responsibility_pattern.fullmatch(prefix_low):
                    return True, "responsibility"

            explicit_inline_section = re.search(
                r"\b("
                r"requirements?|qualifications?|"
                r"responsibilities?|duties|"
                r"benefits?|compensation"
                r")\s*:",
                low[:400],
                flags=re.IGNORECASE,
            )

            if explicit_inline_section is not None:
                marker = (
                    explicit_inline_section.group(1)
                    .lower()
                )

                if marker.startswith(
                    (
                        "requirement",
                        "qualification",
                    )
                ):
                    return True, "qualification"

                if marker.startswith(
                    (
                        "responsibil",
                        "dut",
                    )
                ):
                    return True, "responsibility"

                return True, "excluded"

            strong_patterns = [
                ("responsibility", responsibility_pattern),
                ("excluded", excluded_pattern),
            ]

            weak_inline_terms = {
                "required",
                "requirements",
                "experience",
                "education",
                "skills",
                "salary",
                "compensation",
                "benefits",
            }

            for section_name, pattern in strong_patterns:
                match = pattern.search(low[:180])
                if match is None:
                    continue

                marker = match.group(0).lower()

                if marker in weak_inline_terms:
                    continue

                if (
                    section_name == "excluded"
                    and marker == "benefit"
                    and candidate_qualification_signal_pattern.search(
                        low[:180]
                    )
                ):
                    continue

                return True, section_name

            return False, None

        def _label(
            segment: str,
            section: Optional[str],
        ) -> str:
            if section == "qualification":
                return f"[QUALIFICATION] {segment}"
            if section == "responsibility":
                return f"[RESPONSIBILITY] {segment}"
            return f"[GENERAL] {segment}"

        jurisdiction_labels = list(dict.fromkeys(
            re.findall(
                r"\b([A-Z][A-Za-z]+)\s+Applicants\s*:",
                normalized_text,
            )
        ))

        opening_context = normalized_text[:1500].lower()

        current_jurisdiction = next(
            (
                label
                for label in jurisdiction_labels
                if re.search(
                    rf"\b{re.escape(label.lower())}\b",
                    opening_context,
                )
            ),
            None,
        )

        retained: List[str] = []
        active_section: Optional[str] = None
        jurisdiction_allowed = True

        for segment in segments:
            low = segment.lower()

            jurisdiction_match = re.match(
                r"^\s*([a-z][a-z]+)\s+applicants\s*:",
                low,
            )

            if jurisdiction_match and jurisdiction_labels:
                region_label = jurisdiction_match.group(1)

                jurisdiction_allowed = (
                    current_jurisdiction is None
                    or region_label == current_jurisdiction.lower()
                )

                if not jurisdiction_allowed:
                    continue

                active_section = "qualification"
                retained.append(
                    _label(
                        segment,
                        active_section,
                    )
                )
                continue

            if not jurisdiction_allowed:
                continue

            is_heading, heading_section = _heading_info(
                segment
            )

            if is_heading:
                active_section = heading_section

                if heading_section == "excluded":
                    continue

            if active_section == "excluded":
                # Patch 29 candidate: job-board flattening can place real
                # candidate duties/qualifications directly after an excluded
                # company/benefits block without a clean heading. Do not let
                # the excluded state suppress concrete role content forever.
                if candidate_responsibility_signal_pattern.search(low):
                    active_section = "responsibility"
                elif (
                    credential_pattern.search(low)
                    or candidate_qualification_signal_pattern.search(low)
                ):
                    active_section = "qualification"
                else:
                    continue

            inferred_section = active_section

            if inferred_section is None:
                if (
                    responsibility_pattern.search(low)
                    or candidate_responsibility_signal_pattern.search(
                        low
                    )
                ):
                    inferred_section = "responsibility"
                elif (
                    credential_pattern.search(low)
                    or candidate_qualification_signal_pattern.search(
                        low
                    )
                ):
                    inferred_section = "qualification"

            retained.append(
                _label(
                    segment,
                    inferred_section,
                )
            )

        retained = list(dict.fromkeys(retained))

        if not retained:
            return []

        # Flattened job-board bullets can create one very long labeled segment.
        # Split it BEFORE assigning evidence IDs so every piece remains citable.
        expanded_retained: List[str] = []

        for segment in retained:
            if len(segment) <= MAX_EVIDENCE_SEGMENT_CHARS:
                expanded_retained.append(segment)
                continue

            label_match = re.match(
                r"^(\[(?:GENERAL|RESPONSIBILITY|QUALIFICATION)\])\s*(.*)$",
                segment,
                flags=re.IGNORECASE | re.DOTALL,
            )

            if label_match is None:
                label_prefix = "[GENERAL]"
                content = segment
            else:
                label_prefix = label_match.group(1).upper()
                content = label_match.group(2).strip()

            words = content.split()
            piece_words: List[str] = []
            piece_length = len(label_prefix) + 1

            for word in words:
                additional = (
                    len(word)
                    + (1 if piece_words else 0)
                )

                if (
                    piece_words
                    and piece_length + additional
                    > MAX_EVIDENCE_SEGMENT_CHARS
                ):
                    expanded_retained.append(
                        f"{label_prefix} "
                        + " ".join(piece_words)
                    )
                    piece_words = []
                    piece_length = len(label_prefix) + 1
                    additional = len(word)

                piece_words.append(word)
                piece_length += additional

            if piece_words:
                expanded_retained.append(
                    f"{label_prefix} "
                    + " ".join(piece_words)
                )

        retained = expanded_retained

        # Give every retained evidence segment a stable local ID. The LLM will
        # cite these IDs instead of reproducing exact quotes, which avoids
        # brittle failures caused by punctuation or whitespace paraphrasing.
        retained = [
            f"[S{index:04d}] {segment}"
            for index, segment in enumerate(
                retained,
                start=1,
            )
        ]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for segment in retained:
            separator_length = (
                1
                if current_chunk
                else 0
            )

            next_length = (
                current_length
                + separator_length
                + len(segment)
            )

            if (
                current_chunk
                and next_length > MAX_EVIDENCE_CHUNK_CHARS
            ):
                chunks.append(
                    "\n".join(
                        current_chunk
                    ).strip()
                )
                current_chunk = []
                current_length = 0
                separator_length = 0

            # Extremely long individual segments are split conservatively.
            if len(segment) > MAX_EVIDENCE_CHUNK_CHARS:
                if current_chunk:
                    chunks.append(
                        "\n".join(
                            current_chunk
                        ).strip()
                    )
                    current_chunk = []
                    current_length = 0

                start = 0
                while start < len(segment):
                    piece = segment[
                        start:
                        start + MAX_EVIDENCE_CHUNK_CHARS
                    ].strip()

                    if piece:
                        chunks.append(piece)

                    start += MAX_EVIDENCE_CHUNK_CHARS

                continue

            current_chunk.append(segment)
            current_length = (
                current_length
                + separator_length
                + len(segment)
            )

        if current_chunk:
            chunks.append(
                "\n".join(
                    current_chunk
                ).strip()
            )

        return [
            chunk
            for chunk in chunks
            if chunk
        ]

    @staticmethod
    def _is_generic_non_skill_phrase(
        skill: str,
    ) -> bool:
        """Reject a small, domain-agnostic set of non-skill requirements.

        These rules represent taxonomy policy rather than occupation-specific
        heuristics. They exclude working conditions, benefits, screening items,
        and physical requirements that should not appear in Skills Intelligence.
        """
        value = re.sub(
            r"\s+",
            " ",
            str(skill or ""),
        ).strip().lower()

        if not value:
            return True

        exact_noise = {
            "customer satisfaction",
            "medical clearance",
            "proof of insurance",
            "auto insurance coverage",
            "vehicle insurance",
            "liability coverage",
            "reliable transportation",
            "reliable vehicle",
            "transportation reliability",
            "full vaccination against covid-19",
            "covid-19 vaccination",
            "working full-time",
            "working part-time",
            "overnight travel",
            "traveling as necessary",
            "independent travel",
            "flexible availability",
            "physical work",
            "physical work capability",
            "meeting physical requirements",
            "professional development",
            "career development",
            "misconduct disclosure",
            "authorization to release information",
            "police reference check",
            "age requirement compliance",
        }

        if value in exact_noise:
            return True

        patterns = [
            # Physical/job-condition requirements are outside the Skills
            # Intelligence taxonomy, even when phrased as "ability to ...".
            r"^(?:lift|lifting|carry|carrying)\b",
            r"\b(?:lift|lifting|move|moving)\b.*\b(?:pounds?|lbs?|objects?|items?)\b",
            r"\b(?:auto|vehicle|liability|malpractice)\s+(?:insurance|coverage)\b",
            r"\bproof\s+of\s+insurance\b",
            r"\breliable\s+(?:transportation|vehicle)\b",
            r"\b(?:background|criminal|credit|drug|police)\s+(?:check|screening|history|clearance|reference\s+check)\b",
            r"\bvulnerable\s+sector\s+(?:screening|check)\b",
            r"\breference\s+(?:management|check|checks|screening)\b",
            r"\bmedical\s+clearance\b",
            r"\b(?:employment|professional)\s+history\b",
            r"\bphysical\s+(?:ability|requirements?|demands?|stamina|lifting\s+capability)\b",
            r"\boutdoor\s+work\s+exposure\b",
            r"\b(?:flexible|open)\s+(?:work\s+)?(?:schedule|scheduling|availability)\b",
            r"\b(?:flexibility|availability)\s+(?:in|for|with)\s+(?:work\s+)?(?:hours|shifts?|schedule|scheduling)\b",
            r"\b(?:u\.?s\.?\s+)?work\s+authorization\b",
            r"\b(?:physical\s+)?mobility\s+(?:for|during)\s+(?:work|scheduled\s+shifts?)\b",
            r"\bwork(?:ing)?\s+(?:weekends?|nights?|holidays?|over\s+\d+\s+hours)\b",
            r"^(?:travel|traveling)\b",
            r"\b(?:vaccination|vaccine)\b",
            r"\bpassion\s+for\b",
            r"\b(?:cme|education|tuition)\s+allowance\b",
            r"\b(?:professional|career)\s+development(?:\s+workshops?)?$",
            r"\bmisconduct\s+disclosure\b",
            r"\bauthorization\s+to\s+release\s+information\b",
            r"\bage\s+requirement\b",
        ]

        return any(
            re.search(
                pattern,
                value,
            )
            for pattern in patterns
        )

    @staticmethod
    def _segment_lookup(
        chunk_text: str,
    ) -> Dict[str, str]:
        """Map globally unique evidence IDs to their complete segment text."""
        lookup: Dict[str, str] = {}

        for raw_line in str(
            chunk_text or ""
        ).splitlines():
            line = raw_line.strip()

            match = re.match(
                r"^\[([A-Za-z0-9_]+__S\d{4})\]\s*(.*)$",
                line,
            )

            if match is None:
                continue

            lookup[
                match.group(1).upper()
            ] = match.group(2).strip()

        return lookup

    def _meaningful_role_evidence_ids(
        self,
        chunk_text: str,
    ) -> set[str]:
        """Return substantive responsibility/qualification evidence IDs.

        This is used only as a retry trigger. It does not create skills.
        A segment must already have been classified as RESPONSIBILITY or
        QUALIFICATION, must contain substantive text, and must not be
        employer-offering or application-process boilerplate.
        """
        lookup = self._segment_lookup(
            chunk_text
        )

        role_ids: set[str] = set()

        for evidence_id, segment in lookup.items():
            if re.match(
                r"^\[(?:RESPONSIBILITY|QUALIFICATION)\]",
                segment,
                flags=re.IGNORECASE,
            ) is None:
                continue

            content = re.sub(
                r"^\[(?:RESPONSIBILITY|QUALIFICATION)\]\s*",
                "",
                segment,
                flags=re.IGNORECASE,
            ).strip()

            # Ignore heading fragments such as "Qualifications:".
            if len(
                re.findall(
                    r"\b[\w+#./-]+\b",
                    content,
                )
            ) < 3:
                continue

            if self._is_employer_offering_segment(
                segment
            ):
                continue

            if self._is_application_process_segment(
                segment
            ):
                continue

            role_ids.add(
                evidence_id.upper()
            )

        return role_ids

    def _meaningful_role_evidence_word_count(
        self,
        chunk_text: str,
    ) -> int:
        """Count words in substantive responsibility/qualification evidence.

        This is used only to decide whether a second independent extraction
        pass is warranted. It does not create skills and does not impose any
        minimum skill count.
        """
        lookup = self._segment_lookup(
            chunk_text
        )
        role_ids = self._meaningful_role_evidence_ids(
            chunk_text
        )

        total_words = 0

        for evidence_id in role_ids:
            segment = lookup.get(
                evidence_id,
                "",
            )
            content = re.sub(
                r"^\[(?:RESPONSIBILITY|QUALIFICATION)\]\s*",
                "",
                str(segment or ""),
                flags=re.IGNORECASE,
            ).strip()

            total_words += len(
                re.findall(
                    r"\b[\w+#./-]+\b",
                    content,
                )
            )

        return total_words

    @staticmethod
    def _has_generic_candidate_role_signal(
        chunk_text: str,
    ) -> bool:
        """Return True when text contains generic candidate-role markers.

        This is only an unresolved-zero retry guard. It is intentionally
        occupation-agnostic and never creates a skill by itself.
        """
        value = str(
            chunk_text or ""
        ).lower()

        if not value.strip():
            return False

        patterns = [
            r"\bresponsibilit(?:y|ies)\b",
            r"\bduties\b",
            r"\bjob\s+summary\b",
            r"\bjob\s+description\b",
            r"\brequirements?\b",
            r"\bqualifications?\b",
            r"\bessential\s+functions?\b",
            r"\bthe\s+role\b",
            r"\byour\s+role\b",
            r"\byou(?:['’]ll|\s+will)\b",
        ]

        return any(
            re.search(
                pattern,
                value,
                flags=re.IGNORECASE,
            )
            for pattern in patterns
        )

    @staticmethod
    def _is_employer_offering_segment(
        segment_text: str,
    ) -> bool:
        """Identify employer-provided training/benefit context.

        Candidate responsibilities such as training or coaching other people are
        preserved. The filter targets what the employer gives the candidate.
        """
        value = re.sub(
            r"\s+",
            " ",
            str(
                segment_text or ""
            ),
        ).strip().lower()

        if not value:
            return False

        responsibility_training_patterns = [
            r"\[responsibility\].*\b(?:train|training|coach|coaching|mentor|mentoring)\b.*\b(?:staff|employees?|team|associates?|workers?|students?|direct reports?|others?)\b",
            r"\[responsibility\].*\b(?:develop|deliver|conduct|lead|facilitate)\b.*\btraining\b",
            r"\b(?:your\s+role|you(?:['’]ll|\s+will)|candidates?\s+will)\b.*\b"
            r"(?:develop|deliver|conduct|lead|facilitate|provide)\b.*\btraining\b",
        ]

        if any(
            re.search(
                pattern,
                value,
            )
            for pattern in responsibility_training_patterns
        ):
            return False

        offering_patterns = [
            r"\bwe\s+(?:offer|provide|give|support)\b.*\b(?:training|coaching|mentoring|mentorship|development|education)\b",
            r"\byou(?:'ll|\s+will)\s+(?:receive|get|have|gain|be\s+provided)\b.*\b(?:training|coaching|mentoring|mentorship|development|education)\b",
            r"\b(?:paid|comprehensive|ongoing|extensive|formal|entry-level)\s+(?:sales\s+|marketing\s+|leadership\s+|business\s+development\s+)?training\b",
            r"\btraining\s+(?:is\s+)?provided\b",
            r"\b(?:career|professional)\s+development\s+(?:opportunities|programs?|support|benefits?)\b",
            r"\bmentorship\s+(?:program|opportunities|support)\b",
            r"\b(?:employee|team member)\s+(?:training|development|education)\s+(?:program|benefit|opportunity)\b",
            r"\bour\s+(?:training|development|coaching|mentorship)\s+(?:program|programs|process|opportunities)\b",
            r"\b(?:training|coaching|mentoring|mentorship|development)\s+(?:program|programs|opportunities|provided|available|included)\b",
            r"\bno\s+(?:prior\s+)?experience\b.*\btraining\b",
            r"\bpaid\s+training\s+provided\b",
            r"\bclinical\s+ladders?\b",
            r"\btuition\s+assistance\b",
            r"\bcareer\s+(?:advancement|path|paths|progression)\b",
            r"\bclear\s+(?:career\s+)?path\b",
            r"\bpath\s+to\s+becoming\b",
            r"\btraining\s+to\s+become\b",
            r"\bdevelopment\s+opportunities\b",
            r"\bopportunit(?:y|ies)\s+to\s+(?:grow|develop|advance)\b",
            r"\bchance\s+to\s+(?:grow|develop|advance)\b",
            r"\bdevelop\s+into\s+(?:supervisory|leadership)\b",
            r"\bwhat\s+we\s+offer\b",
        ]

        if any(
            re.search(
                pattern,
                value,
            )
            for pattern in offering_patterns
        ):
            return True

        # List-style employer perks often have no explicit subject.
        # These phrases are offering-oriented regardless of imperfect section
        # labels caused by flattened job-board formatting.
        general_offering_patterns = [
            r"^\[[a-z]+\]\s*training\s+in\b",
            r"^\[general\]\s*hands-on\s+training\b",
            r"^\[[a-z]+\]\s*(?:executive|career|professional)\s+(?:level\s+)?coaching\b",
            r"^\[[a-z]+\]\s*(?:bonuses?|incentives?)\b",
            r"^\[[a-z]+\]\s*travel\s+opportunit(?:y|ies)\b",
            r"^\[[a-z]+\]\s*(?:personal|professional|career)\s+development\b",
            r"^\[[a-z]+\]\s*mentorship\b",
        ]

        if any(
            re.search(
                pattern,
                value,
            )
            for pattern in general_offering_patterns
        ):
            return True

        return False

    @staticmethod
    def _is_application_process_segment(
        segment_text: str,
    ) -> bool:
        """Identify hiring-process submission instructions rather than skills."""
        value = re.sub(
            r"\s+",
            " ",
            str(segment_text or ""),
        ).strip().lower()

        if not value:
            return False

        patterns = [
            r"\bapplicants?\b.*\b(?:submit|upload|provide)\b",
            r"\b(?:application|selection)\s+process\b",
            r"\bbe prepared to\b.*\b(?:upload|submit|provide)\b",
            r"\bcurriculum vitae\b",
            r"\bcover letter\b",
            r"\bstatement of (?:personal )?faith\b",
            r"\bprovide names? and contacts? for\b.*\breferences?\b",
            r"\bofficial transcripts?\b.*\b(?:required|submit|provide|contract)\b",
            r"\bunofficial transcripts?\b",
            r"\bapplication materials?\b",
        ]

        return any(
            re.search(
                pattern,
                value,
            )
            for pattern in patterns
        )

    @staticmethod
    def _is_general_segment(
        segment_text: str,
    ) -> bool:
        """Return True when evidence is general context rather than a candidate section."""
        value = str(
            segment_text or ""
        )

        return (
            "[GENERAL]" in value
            and "[RESPONSIBILITY]" not in value
            and "[QUALIFICATION]" not in value
        )


    def _build_grounded_chunk_prompt(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], List[str], int]:
        """Build a segment-ID-grounded chunk extraction prompt."""
        system_prompt = (
            "You are an evidence-grounded job-skill extraction engine.\n\n"
            "Each input item is one chunk from a job posting. Every evidence line "
            "begins with a globally unique segment ID such as [job_012345__chunk_001__S0007].\n\n"
            "Extract only candidate capabilities that are explicitly supported by "
            "one or more segment IDs in that same item.\n\n"
            "For every extracted item return:\n"
            "- skill: one concise resume-style noun phrase;\n"
            "- evidence_ids: one or more exact globally unique segment IDs from item.text;\n"
            "- skill_type: one of skill, workflow, credential, soft_skill. Use 'skill' for concrete tools, technologies, platforms, programming languages, frameworks, databases, equipment, methods, and standards. Do not invent other skill_type labels such as 'tool' or 'technology'.\n\n"
            "Prefer normalized noun phrases rather than copying full task sentences. "
            "For example, convert 'manage inventory levels' to 'inventory management' "
            "and 'develop sourcing strategies' to 'sourcing strategy development'.\n\n"
            "Extract concrete technologies, methods, tools, standards, workflows, "
            "domain practices, licenses, certifications, degrees, formal "
            "qualifications, repeatable work capabilities, and explicitly required "
            "soft skills.\n\n"
            "Do NOT extract:\n"
            "- job titles or occupational labels by themselves;\n"
            "- employer products, service offerings, business lines, departments, "
            "markets, or company capabilities mentioned only as context;\n"
            "- benefits, mentoring/support programs, optional training, tuition or "
            "professional-development offerings;\n"
            "- application instructions or submission materials such as resumes, "
            "cover letters, transcripts, references, or statements requested only "
            "for the hiring process;\n"
            "- compensation, pay formulas, schedules, travel requirements, vehicle "
            "or insurance requirements, background checks, vaccination requirements, "
            "or physical requirements such as lifting weight;\n"
            "- vague traits, business outcomes, slogans, or generic nouns.\n\n"
            "A credential containing an occupation name is valid when the evidence "
            "explicitly requires that credential.\n\n"
            "Scan every evidence line before finishing the record, including "
            "[GENERAL] lines. Section labels are hints, not hard gates. A [GENERAL] "
            "line may contain a real candidate responsibility when the job board uses "
            "an unfamiliar heading. Extract from [GENERAL] only when the text itself "
            "clearly describes something the candidate must know, use, perform, "
            "manage, build, inspect, operate, analyze, repair, sell, teach, document, "
            "or otherwise do as part of the role. Do not treat employer descriptions, "
            "benefits, offerings, outcomes, or marketing language as candidate skills.\n\n"
            "Coverage rule: assess each candidate-facing segment independently. If a "
            "segment contains a concrete repeatable workflow, named tool, technology, "
            "equipment type, method, standard, credential, license, degree, or "
            "role-specific technical experience, preserve that concrete detail. Do "
            "not replace several materially distinct named capabilities with one vague "
            "summary such as 'technical skills', 'quality control', or 'design software'.\n\n"
            "Workflow coverage rule: do not reduce a role to credentials, tools, or "
            "soft skills when candidate-facing evidence also states substantive work "
            "the candidate is expected to perform. Convert explicit repeatable duties "
            "into concise workflow noun phrases even when the work is nontechnical, "
            "manual, caregiving, sales, teaching, coordination, or administrative. "
            "Do not infer duties that are not stated, and do not extract employer "
            "marketing, benefits, or ordinary working conditions.\n\n"
            "Literal enumeration rule: when candidate-facing evidence explicitly lists "
            "named tools, technologies, platforms, languages, frameworks, databases, "
            "equipment, standards, certifications, or methods, extract EACH materially "
            "distinct listed item separately. Do not collapse an explicit list into a "
            "broader label such as 'inspection tools', 'migration tools', 'programming "
            "languages', or 'technical stack'. This applies to both capitalized software "
            "names and lower-case equipment names. If the evidence says 'Tools include "
            "AlphaTool, BetaTool, and GammaTool', return AlphaTool, BetaTool, and GammaTool "
            "as separate skills. If the evidence says 'Experience with FrameworkA, "
            "LanguageB, DatabaseC', return each named item separately. Keep a broader "
            "workflow only when it adds distinct information. For each concrete listed "
            "tool, technology, platform, language, framework, database, equipment item, "
            "method, or standard, set skill_type to 'skill'.\n\n"
            "Employer-provided training, coaching, mentoring, incentives, or "
            "professional-development opportunities are not candidate skills. Training "
            "or coaching is valid only when the candidate is expected to train or coach "
            "other people as part of the job.\n\n"
            "Do not target a minimum count. Combine true duplicates and near-duplicates, "
            "but keep materially distinct tools, workflows, standards, and credentials.\n\n"
            "Every evidence ID you return MUST exactly match an ID in item.text. Do not invent "
            "segment IDs or skills.\n\n"
            "Return STRICT JSON only. Keys must exactly match the supplied chunk IDs. "
            "Values must be arrays of objects with keys skill, evidence_ids, and "
            "skill_type.\n"
        )

        user_prefix = (
            "Extract segment-grounded skills for every chunk.\n\n"
            "Return STRICT JSON in this shape:\n"
            "{\n"
            '  "<chunk_id>": [\n'
            "    {\n"
            '      "skill": "inventory management",\n'
            '      "evidence_ids": ["job_012345__chunk_001__S0007"],\n'
            '      "skill_type": "workflow"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Items:\n"
        )

        payload = []
        input_ids: List[str] = []

        for record in records:
            chunk_id = str(record["id"])
            input_ids.append(chunk_id)

            ner = record.get(
                "ner",
                [],
            )

            if not isinstance(
                ner,
                list,
            ):
                ner = []

            ner = [
                str(skill).strip()
                for skill in ner
                if str(skill).strip()
            ][:30]

            payload.append({
                "id": chunk_id,
                "text": (
                    record.get(
                        "text",
                        "",
                    )
                    or ""
                ),
                "ner_hints": ner,
            })

        user_content = (
            user_prefix
            + json.dumps(
                payload,
                ensure_ascii=False,
            )
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        prompt_len = (
            len(system_prompt)
            + len(user_content)
        )

        return (
            messages,
            input_ids,
            prompt_len,
        )

    def _validate_grounded_chunk_map(
        self,
        raw_map: Any,
        records: List[Dict[str, Any]],
    ) -> Tuple[
        Dict[str, List[str]],
        set[str],
    ]:
        """Validate segment IDs and return cleaned skill lists by chunk.

        Returns:
            validated skill lists by chunk ID;
            chunk IDs whose response was structurally invalid and should be retried.
        """
        record_map = {
            str(record["id"]): record
            for record in records
        }

        validated: Dict[str, List[str]] = {
            chunk_id: []
            for chunk_id in record_map
        }

        retry_ids: set[str] = set()

        if not isinstance(
            raw_map,
            dict,
        ):
            return (
                validated,
                set(record_map),
            )

        for chunk_id, record in record_map.items():
            if chunk_id not in raw_map:
                retry_ids.add(
                    chunk_id
                )
                continue

            raw_items = raw_map.get(
                chunk_id,
            )

            if not isinstance(
                raw_items,
                list,
            ):
                retry_ids.add(
                    chunk_id
                )
                continue

            record_text = str(
                record.get(
                    "text",
                    "",
                )
                or ""
            )

            valid_segment_ids = {
                evidence_id.upper()
                for evidence_id in re.findall(
                    r"\[([A-Za-z0-9_]+__S\d{4})\]",
                    record_text,
                )
            }

            segment_lookup = (
                self._segment_lookup(
                    record_text
                )
            )

            grounded_skills: List[str] = []
            accepted_evidence_ids: set[str] = set()
            saw_nonempty_item = False
            saw_structural_grounding_failure = False

            for item in raw_items:
                if not isinstance(
                    item,
                    dict,
                ):
                    saw_structural_grounding_failure = True
                    continue

                saw_nonempty_item = True

                skill = str(
                    item.get(
                        "skill",
                        "",
                    )
                    or ""
                ).strip()

                evidence_ids = item.get(
                    "evidence_ids",
                    [],
                )

                skill_type = str(
                    item.get(
                        "skill_type",
                        "",
                    )
                    or ""
                ).strip().lower()

                skill_type_aliases = {
                    "tool": "skill",
                    "technology": "skill",
                    "platform": "skill",
                    "software": "skill",
                    "framework": "skill",
                    "programming_language": "skill",
                    "language": "skill",
                    "database": "skill",
                    "equipment": "skill",
                    "method": "skill",
                    "standard": "skill",
                    "technical_skill": "skill",
                    "technical skill": "skill",
                    "process": "workflow",
                    "task": "workflow",
                    "certification": "credential",
                    "license": "credential",
                    "degree": "credential",
                    "soft skill": "soft_skill",
                    "soft-skill": "soft_skill",
                    "softskill": "soft_skill",
                }

                skill_type = skill_type_aliases.get(
                    skill_type,
                    skill_type,
                )

                allowed_skill_types = {
                    "skill",
                    "workflow",
                    "credential",
                    "soft_skill",
                }

                missing_skill_type = not skill_type

                # Patch-29 v5: skill_type is useful metadata, but it must not
                # become a single-point-of-failure for an otherwise grounded
                # extraction. In exposed v4 diagnostics, the model sometimes
                # returned a valid skill and exact evidence IDs while omitting
                # only skill_type; v4 then discarded the whole item.
                #
                # Keep malformed non-empty labels strict, but defer a missing
                # label until after the cited evidence is inspected. Missing
                # type is recoverable only when every cited segment is explicit
                # RESPONSIBILITY / QUALIFICATION evidence. This avoids allowing
                # untyped GENERAL/company-context items to bypass the soft-skill
                # safeguard. Recovered items still trigger isolated confirmation
                # below through saw_structural_grounding_failure.
                if (
                    not skill
                    or not isinstance(
                        evidence_ids,
                        list,
                    )
                    or not evidence_ids
                    or (
                        skill_type
                        and skill_type not in allowed_skill_types
                    )
                ):
                    saw_structural_grounding_failure = True
                    continue

                cleaned_evidence_ids = []

                for evidence_id in evidence_ids:
                    normalized_id = str(
                        evidence_id
                        or ""
                    ).strip()

                    normalized_id = (
                        normalized_id
                        .strip("[]")
                        .upper()
                    )

                    if normalized_id:
                        cleaned_evidence_ids.append(
                            normalized_id
                        )

                expected_prefix = (
                    f"{chunk_id}__".upper()
                )

                if (
                    not cleaned_evidence_ids
                    or any(
                        (
                            not evidence_id.startswith(
                                expected_prefix
                            )
                            or evidence_id
                            not in valid_segment_ids
                        )
                        for evidence_id
                        in cleaned_evidence_ids
                    )
                ):
                    saw_structural_grounding_failure = True
                    logger.warning(
                        "Dropping cross-chunk or invalid evidence "
                        "for chunk %s: %s",
                        chunk_id,
                        skill,
                    )
                    continue

                cited_segments = [
                    segment_lookup.get(
                        evidence_id,
                        "",
                    )
                    for evidence_id
                    in cleaned_evidence_ids
                ]

                if missing_skill_type:
                    candidate_facing_segments = (
                        bool(cited_segments)
                        and all(
                            re.match(
                                r"^\[(?:RESPONSIBILITY|QUALIFICATION)\]",
                                segment,
                                flags=re.IGNORECASE,
                            )
                            is not None
                            for segment in cited_segments
                        )
                    )

                    if not candidate_facing_segments:
                        saw_structural_grounding_failure = True
                        logger.warning(
                            "Dropping untyped grounded item outside explicit "
                            "candidate-facing evidence for chunk %s: %s",
                            chunk_id,
                            skill,
                        )
                        continue

                    # The final product stores only the skill phrase, not the
                    # type. Use a neutral accepted type so the grounded phrase
                    # survives validation, while still forcing an independent
                    # confirmation because the response schema was incomplete.
                    skill_type = "skill"
                    saw_structural_grounding_failure = True
                    logger.info(
                        "Recovering grounded item with missing skill_type for "
                        "chunk %s: %s",
                        chunk_id,
                        skill,
                    )

                if (
                    cited_segments
                    and all(
                        self._is_employer_offering_segment(
                            segment
                        )
                        for segment
                        in cited_segments
                    )
                ):
                    logger.info(
                        "Dropping employer-offering skill "
                        "for chunk %s: %s",
                        chunk_id,
                        skill,
                    )
                    continue

                if (
                    cited_segments
                    and all(
                        self._is_application_process_segment(
                            segment
                        )
                        for segment
                        in cited_segments
                    )
                ):
                    logger.info(
                        "Dropping application-process item "
                        "for chunk %s: %s",
                        chunk_id,
                        skill,
                    )
                    continue

                # Generic soft-skill language in an opening/company paragraph is
                # too weak for Skills Intelligence. Soft skills remain valid when
                # grounded in explicit responsibility/qualification evidence.
                if (
                    skill_type == "soft_skill"
                    and cited_segments
                    and all(
                        self._is_general_segment(
                            segment
                        )
                        for segment
                        in cited_segments
                    )
                ):
                    continue

                if self._is_generic_non_skill_phrase(
                    skill
                ):
                    continue

                grounded_skills.append(
                    skill
                )
                accepted_evidence_ids.update(
                    cleaned_evidence_ids
                )

            meaningful_role_ids = (
                self._meaningful_role_evidence_ids(
                    record_text
                )
            )

            role_evidence_coverage = (
                len(
                    meaningful_role_ids
                    & accepted_evidence_ids
                )
                / len(
                    meaningful_role_ids
                )
                if meaningful_role_ids
                else 1.0
            )

            meaningful_role_word_count = (
                self._meaningful_role_evidence_word_count(
                    record_text
                )
            )

            # Rich substantive role evidence receives one independent
            # confirmation pass even when the first batch result looks
            # structurally valid. The two grounded outputs are merged later.
            #
            # This is a source-richness trigger, NOT a minimum skill-count rule.
            # It is intended to reduce stochastic partial under-recall when one
            # long role segment contains several distinct capabilities.
            rich_role_confirmation = (
                len(
                    meaningful_role_ids
                ) >= 4
                or meaningful_role_word_count >= 100
            )

            # A non-empty response can still be materially incomplete. When a
            # chunk contains many explicit responsibility/qualification
            # segments but accepted skills cite only a small fraction of them,
            # request one isolated confirmation. This is a coverage-based
            # trigger, not a minimum-skill-count rule.
            sparse_role_coverage = (
                len(
                    meaningful_role_ids
                ) >= 6
                and role_evidence_coverage < 0.40
            )

            # If the model attempted items but every item failed structural
            # validation, retry the chunk once in isolation. Also confirm an
            # explicit [] once in isolation because a batched empty decision can
            # otherwise erase a valid role. If the isolated retry is also empty,
            # the empty result remains legitimate.
            if (
                saw_structural_grounding_failure
                or sparse_role_coverage
                or rich_role_confirmation
                or (
                    raw_items
                    and saw_nonempty_item
                    and not grounded_skills
                )
                or (
                    raw_items == []
                    and record_text.strip()
                )
            ):
                retry_ids.add(
                    chunk_id
                )

            cleaned_map = self._validate_llm_skill_map(
                {
                    chunk_id: grounded_skills,
                },
                [
                    chunk_id,
                ],
            )

            validated[chunk_id] = (
                cleaned_map.get(
                    chunk_id,
                    [],
                )
            )

        return (
            validated,
            retry_ids,
        )

    def _retry_grounded_chunk(
        self,
        record: Dict[str, Any],
    ) -> List[str]:
        """Retry one malformed/ungrounded chunk once in isolation."""
        chunk_id = str(
            record["id"]
        )

        messages, _, _ = (
            self._build_grounded_chunk_prompt(
                [
                    record,
                ]
            )
        )

        if self.llm_client is None:
            return []

        logger.warning(
            "Retrying grounded extraction for chunk %s "
            "in isolation.",
            chunk_id,
        )

        try:
            response = (
                self.llm_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.0,
                )
            )

            response_text = (
                response.choices[0].message.content
                or ""
            )

        except Exception as exc:
            logger.error(
                "Grounded chunk retry failed for %s: %s",
                chunk_id,
                exc,
            )
            return []

        if not response_text.strip():
            return []

        raw_map = self._repair_json_to_dict(
            response_text
        )

        validated, _ = (
            self._validate_grounded_chunk_map(
                raw_map,
                [
                    record,
                ],
            )
        )

        return validated.get(
            chunk_id,
            [],
        )

    def batch_llm_extract_grounded_chunks(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, List[str]]:
        """Extract segment-grounded skills from description chunks."""
        if not records:
            return {}

        messages, input_ids, prompt_len = (
            self._build_grounded_chunk_prompt(
                records
            )
        )

        if (
            prompt_len > self.max_prompt_chars
            and len(records) > 1
        ):
            midpoint = max(
                1,
                len(records) // 2,
            )

            left_records = records[
                :midpoint
            ]
            right_records = records[
                midpoint:
            ]

            logger.info(
                "Segment-grounded prompt exceeds limit "
                "(%s > %s characters); splitting batch "
                "into %s and %s chunks.",
                prompt_len,
                self.max_prompt_chars,
                len(left_records),
                len(right_records),
            )

            left_results = (
                self.batch_llm_extract_grounded_chunks(
                    left_records
                )
            )
            right_results = (
                self.batch_llm_extract_grounded_chunks(
                    right_records
                )
            )

            return {
                **left_results,
                **right_results,
            }

        empty_out = {
            chunk_id: []
            for chunk_id in input_ids
        }

        if self.llm_client is None:
            return empty_out

        logger.info(
            "LLM segment-grounded extraction chunks: %s",
            len(records),
        )

        try:
            response = (
                self.llm_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.0,
                )
            )

            response_text = (
                response.choices[0].message.content
                or ""
            )

            usage = getattr(
                response,
                "usage",
                None,
            )

            if usage is not None:
                logger.info(
                    "OpenAI segment-grounded usage - "
                    "prompt_tokens: %s, completion_tokens: %s, "
                    "total_tokens: %s",
                    getattr(
                        usage,
                        "prompt_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "completion_tokens",
                        None,
                    ),
                    getattr(
                        usage,
                        "total_tokens",
                        None,
                    ),
                )

        except Exception as exc:
            logger.error(
                "Segment-grounded LLM call failed: %s",
                exc,
            )
            logger.error(
                traceback.format_exc()
            )
            return empty_out

        if not response_text.strip():
            return empty_out

        raw_map = self._repair_json_to_dict(
            response_text
        )

        validated, retry_ids = (
            self._validate_grounded_chunk_map(
                raw_map,
                records,
            )
        )

        if retry_ids:
            record_lookup = {
                str(record["id"]): record
                for record in records
            }

            for chunk_id in sorted(
                retry_ids
            ):
                record = record_lookup.get(
                    chunk_id
                )

                if record is None:
                    continue

                retry_skills = (
                    self._retry_grounded_chunk(
                        record
                    )
                )

                # Use the isolated call as an independent confirmation, but
                # do not union the two passes. Both outputs are grounded; keep
                # the more complete pass to reduce stochastic under-recall
                # without creating duplicate-like inflation.
                validated[chunk_id] = (
                    self._select_more_complete_grounded_pass(
                        validated.get(
                            chunk_id,
                            [],
                        ),
                        retry_skills,
                    )
                )

                # Accepting a true zero is important, but one batch empty plus
                # one isolated empty can still be stochastic. For a chunk with
                # substantive role evidence, require one additional isolated
                # confirmation before leaving the result empty. This extra call
                # occurs only on unresolved zero chunks.
                if (
                    not validated[chunk_id]
                    and (
                        self._meaningful_role_evidence_ids(
                            str(
                                record.get(
                                    "text",
                                    "",
                                )
                                or ""
                            )
                        )
                        or self._has_generic_candidate_role_signal(
                            str(
                                record.get(
                                    "text",
                                    "",
                                )
                                or ""
                            )
                        )
                    )
                ):
                    second_retry_skills = (
                        self._retry_grounded_chunk(
                            record
                        )
                    )

                    validated[chunk_id] = (
                        self._select_more_complete_grounded_pass(
                            validated[chunk_id],
                            second_retry_skills,
                        )
                    )

        return validated


    @staticmethod
    def _extract_literal_list_skills_from_evidence(
        chunk_text: str,
    ) -> List[str]:
        """Recover explicit named items from candidate-facing evidence.

        This safeguard is intentionally narrow. It only inspects
        RESPONSIBILITY / QUALIFICATION evidence and only extracts items from
        explicit list constructions such as:

        - "tools including A, B, and C"
        - "frameworks such as A, B"
        - "(e.g., A, B)"

        The method does not infer new skills. Every returned phrase must appear
        literally in the grounded evidence text.
        """
        if not chunk_text:
            return []

        family_pattern = (
            r"(?:\b(?:"
            r"tools?|systems?|technolog(?:y|ies)|"
            r"programming languages?|languages?|frameworks?|"
            r"databases?|platforms?|software|solutions?|apps?|"
            r"equipment|instruments?|apparatus|"
            r"identity providers?|migration tools?"
            r")\b)"
        )

        example_prefix_pattern = re.compile(
            r"(?:"
            + family_pattern
            + r"|experience\s+with|knowledge\s+of|"
            r"familiarity\s+with|proficiency\s+with"
            r").{0,60}$",
            flags=re.IGNORECASE,
        )

        continuation_stop_pattern = re.compile(
            r"\s+(?=("
            r"Produce|Perform|Conduct|Maintain|Develop|Read|Ensure|"
            r"Participate|Collaborate|Provide|Execute|Inspect|Use|"
            r"Support|Manage|Work|Comply|Sustain|Track|Prepare|"
            r"Monitor|Analyze|Review|Calibrate|"
            r"Expertise\s+(?:in|with|developing|using|building|working)|"
            r"Experience\s+(?:in|with|using|working|developing|designing|building)|"
            r"Proven\s+(?:history|track\s+record|ability|experience)|"
            r"Demonstrated\s+(?:ability|application|experience|knowledge|proficiency)|"
            r"Proficiency\s+(?:in|with)|"
            r"Knowledge\s+of|"
            r"Familiarity\s+with|"
            r"Ability\s+to|"
            r"Excellent\s+(?:(?:[A-Za-z][A-Za-z-]*\s+){0,3})"
            r"(?:skills?|knowledge|proficiency|experience)|"
            r"Highly\s+(?:organized|proficient|skilled)|"
            r"Team\s+player|"
            r"(?:is|are)\s+(?:a\s+)?plus|"
            r"to\s+(?:create|build|generate|enable|allow)\b|"
            r"Strong\s+(?:communication|interpersonal|understanding|knowledge)|"
            r"Collaborative\s+experience|"
            r"Prior\s+(?:work\s+)?experience|"
            r"Well\s+versed\s+in|"
            r"Fluent\s+in|"
            r"Candidates?\s+(?:must|should)|"
            r"Applicants?\s+(?:must|should)|"
            r"Required|Preferred|"
            r"including\s+[a-z][a-z-]*ing"
            r")\b)"
        )

        generic_items = {
            "etc",
            "etc.",
            "other",
            "others",
            "and more",
            "other tools",
            "related tools",
            "other technologies",
            "related technologies",
            "other key technologies",
            "company standards",
            "more",
            "equipment",
            "tools",
            "systems",
            "platforms",
        }

        def split_literal_items(
            raw_list: str,
        ) -> List[str]:
            value = re.sub(
                r"\s+",
                " ",
                raw_list or "",
            ).strip(" \t\r\n.;:")

            if not value:
                return []

            # Flattened scraped text can continue directly into the next
            # responsibility sentence. Stop before a clear action verb.
            value = continuation_stop_pattern.split(
                value,
                maxsplit=1,
            )[0].strip(" ,;")

            value = re.sub(
                r"\b(?:and|or)\s+other\b.*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip(" ,;")

            value = re.sub(
                r"^(?:but\s+)?not\s+limited\s+to\s*:?\s*",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip(" ,;")

            value = re.split(
                r";\s*(?=(?:may\s+(?:operate|use|perform|require)|"
                r"performs?|operates?|provides?|supports?|manages?|"
                r"develops?|maintains?|assists?)\b)",
                value,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip(" ,;")

            pieces = re.split(
                r"\s*,\s*|\s*;\s*|\s+(?:and|or|&)\s+",
                value,
                flags=re.IGNORECASE,
            )

            results: List[str] = []

            for piece in pieces:
                candidate = piece.strip(
                    " \t\r\n.;:()[]{}"
                )

                candidate = re.sub(
                    r"^(?:and|or)\s+",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"^(?:e\.g\.|for example|such as)\s*[:,]?\s*",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"^(?:but\s+)?not\s+limited\s+to\s*:?\s*",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                if not candidate:
                    continue

                if candidate.lower().startswith("the "):
                    remainder = candidate[4:].strip()

                    if len(remainder.split()) >= 2:
                        candidate = remainder
                    else:
                        continue

                usage_prefix = re.match(
                    r"^(?:using|via|effective\s+use\s+of)\s+(.+)$",
                    candidate,
                    flags=re.IGNORECASE,
                )

                if usage_prefix is not None:
                    remainder = usage_prefix.group(1).strip()

                    if (
                        len(remainder.split()) <= 4
                        and re.search(
                            r"[A-Z0-9/#.+-]",
                            remainder,
                        )
                    ):
                        candidate = remainder

                if re.match(
                    r"^(?:is|are|was|were|from|to|for|with|without|"
                    r"by|as|who|which|that|this|these|those|"
                    r"may\s+(?:operate|use|perform|require)|"
                    r"performs?|operates?|provides?|supports?|"
                    r"manages?|develops?|maintains?|assists?)\b",
                    candidate,
                    flags=re.IGNORECASE,
                ):
                    continue

                if (
                    candidate[:1].islower()
                    and re.match(
                        r"^(?:create|maintain|develop|perform|provide|"
                        r"support|manage|operate|use|work|ensure|assist|"
                        r"monitor|analyze|review|prepare|conduct|execute|"
                        r"inspect)\b",
                        candidate,
                        flags=re.IGNORECASE,
                    )
                ):
                    continue

                candidate = re.sub(
                    r"\s+to\s+(?:perform|support|provide|ensure|develop|"
                    r"maintain|operate|use|work|create|build|generate|"
                    r"enable|allow)\b.*$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"\s+(?:is|are)\s+(?:a\s+)?plus\b.*$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"\s+must\s+be\b.*$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"\s+OTHER$",
                    "",
                    candidate,
                ).strip()

                candidate = re.sub(
                    r"\s+(?:mostly\s+)?used\s+(?:on|for|in|to|with)\b.*$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                candidate = re.sub(
                    r"\s*\([^)]*\b(?:experience|preferred|required|asset|plus)\b[^)]*\)?$",
                    "",
                    candidate,
                    flags=re.IGNORECASE,
                ).strip()

                if not candidate:
                    continue

                if candidate.count("(") > candidate.count(")"):
                    parenthetical_tail = candidate.rsplit("(", 1)[1].strip()

                    if (
                        parenthetical_tail
                        and len(parenthetical_tail.split()) <= 4
                        and re.search(
                            r"[A-Z0-9/#.+-]",
                            parenthetical_tail,
                        )
                    ):
                        candidate = parenthetical_tail
                    else:
                        continue

                if re.search(
                    family_pattern
                    + r".{0,8}?\b(?:include|includes|including|such as)\b",
                    candidate,
                    flags=re.IGNORECASE,
                ):
                    continue

                if candidate.lower() in generic_items:
                    continue

                if (
                    len(candidate) > 80
                    or len(candidate.split()) > 8
                ):
                    continue

                if re.fullmatch(
                    r"[\d\W_]+",
                    candidate,
                ):
                    continue

                results.append(
                    candidate
                )

            return results

        literal_items: List[str] = []

        line_pattern = re.compile(
            r"^\[[^\]]+\]\s+"
            r"\[(RESPONSIBILITY|QUALIFICATION)\]\s+"
            r"(.*)$"
        )

        direct_list_start_pattern = re.compile(
            family_pattern
            + r".{0,8}?\b"
            r"(?:include|includes|including|such as)\b"
            r"\s*[:,-]?\s*",
            flags=re.IGNORECASE,
        )

        # Appositional product naming, e.g.
        # "IT support ticketing system, Jira Service Management".
        appositional_name_pattern = re.compile(
            family_pattern
            + r"\s*,\s*"
            r"([A-Z][A-Za-z0-9+.#/&()'’ -]{1,60})"
            r"(?:[.;]|$)"
        )

        example_pattern = re.compile(
            r"\((?:e\.g\.|for example)\s*[:,]?\s*([^)]+)\)",
            flags=re.IGNORECASE,
        )

        for raw_line in chunk_text.splitlines():
            match = line_pattern.match(
                raw_line.strip()
            )

            if not match:
                continue

            segment_text = match.group(2).strip()

            for list_match in direct_list_start_pattern.finditer(
                segment_text
            ):
                literal_items.extend(
                    split_literal_items(
                        segment_text[
                            list_match.end():
                        ]
                    )
                )

            for apposition_match in appositional_name_pattern.finditer(
                segment_text
            ):
                literal_items.extend(
                    split_literal_items(
                        apposition_match.group(1)
                    )
                )

            for example_match in example_pattern.finditer(
                segment_text
            ):
                prefix = segment_text[
                    :example_match.start()
                ]

                if not example_prefix_pattern.search(
                    prefix
                ):
                    continue

                literal_items.extend(
                    split_literal_items(
                        example_match.group(1)
                    )
                )

        final_items: List[str] = []
        seen = set()

        for item in literal_items:
            normalized = re.sub(
                r"\s+",
                " ",
                item,
            ).strip()

            if not normalized:
                continue

            # Final grounding check: safeguard outputs must be literal spans.
            if normalized.lower() not in chunk_text.lower():
                continue

            key = normalized.lower()

            if key in seen:
                continue

            seen.add(
                key
            )
            final_items.append(
                normalized
            )

        return final_items


    # ----------------------------------------------------------------------
    # Merge and clean
    # ----------------------------------------------------------------------
    def merge_and_deduplicate_skills(
        self, ner_skills: List[str], llm_skills: List[str]
    ) -> List[str]:
        """Merge available skill lists and conservatively remove duplicates."""
        all_skills = (ner_skills or []) + (llm_skills or [])
        cleaned: List[str] = []

        for skill in all_skills:
            if not skill:
                continue

            candidate = self._strip_noise_words(skill)
            candidate = re.sub(r"\s+", " ", candidate).strip()

            if not candidate or len(candidate) <= 1:
                continue
            if "##" in candidate:
                continue

            cleaned.append(candidate)

        return self._deduplicate_skill_phrases(cleaned)

    # ----------------------------------------------------------------------
    # Database I/O
    # ----------------------------------------------------------------------
    def update_record_skills(self, record_id, skills: List[str], id_field: str = 'id') -> bool:
        """
        Update extracted skills for a single database record.

        Skills are stored as plain text with one numbered skill per line rather than JSON
        to support straightforward dashboard display.
        """
        try:
            # Join skills into multi-line text, no JSON, no extra quotes
            skills_text = "\n".join(skills)

            with self.db_engine.connect() as conn:
                # Use parameterized values for the record update.
                update_sql = text(
                    f"UPDATE {TABLE_NAME} "
                    f"SET {SKILLS_COLUMN} = :skills "
                    f"WHERE {id_field} = :record_id"
                )

                params = {
                    "skills": skills_text,
                    "record_id": record_id,
                }

                conn.execute(update_sql, params)
                conn.commit()

            logger.info(f"Record {self._safe_id(record_id)} skills updated successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to update skills for record {self._safe_id(record_id)}: {e}")
            logger.error(traceback.format_exc())
            return False

    def get_all_records(self) -> Tuple[pd.DataFrame, str]:
        """Fetch database records that still require skill extraction.

        Returns:
            A DataFrame containing standardized ID and description columns,
            along with the source ID column name."""
        try:
            with self.db_engine.connect() as conn:
                desc_res = conn.execute(
                    text(f"DESCRIBE {TABLE_NAME}")
                )
                columns = [row[0] for row in desc_res.fetchall()]
                logger.info(f"Table columns: {columns}")

                if ID_COLUMN not in columns:
                    raise RuntimeError(
                        f"Configured ID column '{ID_COLUMN}' "
                        f"was not found in table '{TABLE_NAME}'."
                    )

                id_field = ID_COLUMN

                logger.info(f"Record ID column: {id_field}")

                if DESCRIPTION_COLUMN not in columns:
                    raise RuntimeError(
                        f"Configured description column '{DESCRIPTION_COLUMN}' "
                        f"was not found in table '{TABLE_NAME}'."
                    )

                desc_field = DESCRIPTION_COLUMN
                
                logger.info(f"Description column: {desc_field}")

                skills_field = SKILLS_COLUMN

                if skills_field not in columns:
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

                where_clause = (
                    f"({skills_field} IS NULL "
                    f"AND {desc_field} IS NOT NULL "
                    f"AND TRIM({desc_field}) <> ''"
                )
                base_sql = (
                    f"SELECT {id_field}, {desc_field} "
                    f"FROM {TABLE_NAME} "
                    f"WHERE {where_clause}"
                )

                if self.record_limit > 0:
                    base_sql += f" LIMIT {self.record_limit}"
                    logger.info(
                        f"Record limit enabled: {self.record_limit}"
                    )
                else:
                    logger.info(
                        "Record limit disabled; processing all pending records."
                    )

                logger.info(f"Query SQL: {base_sql}")
                res = conn.execute(text(base_sql))
                rows = res.fetchall()

                if not rows:
                    logger.info("No records found that require processing.")
                    return pd.DataFrame(columns=['id', 'description']), id_field

                df = pd.DataFrame(rows, columns=[id_field, 'description'])

                df["id"] = df[id_field]

                df['description'] = df['description'].astype(str)
                logger.info(f"Records to process: {len(df)}")
                return df[['id', 'description']], id_field

        except Exception as e:
            logger.error(f"Failed to fetch records: {e}")
            logger.error(traceback.format_exc())
            raise

    # ----------------------------------------------------------------------
    # Main processing
    # ----------------------------------------------------------------------
    def process_all_records(self) -> Dict[str, Any]:
        """Process pending records with conservative screening + grounded chunks."""
        logger.info(
            "Starting grounded processing with deterministic literal-list safeguard."
        )
        logger.info(
            f"LLM logical batch size: {self.batch_size}"
        )

        df, id_field = self.get_all_records()

        if df.empty:
            logger.warning(
                "No records found to process."
            )
            return {
                "total": 0,
                "success": 0,
                "failed": 0,
                "failed_records": [],
            }

        stats = {
            "total": len(df),
            "success": 0,
            "failed": 0,
            "failed_records": [],
        }

        raw_records: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            record_id = row["id"]
            record_id_str = (
                record_id.hex()
                if isinstance(
                    record_id,
                    bytes,
                )
                else str(record_id)
            )

            raw_records.append({
                "id": record_id_str,
                "db_id": record_id,
                "description": row["description"],
            })

        logger.info(
            "Step 1: Screening records for substantive role content..."
        )

        screen_results: Dict[str, bool] = {}

        screen_batches = [
            raw_records[
                index:
                index + self.batch_size
            ]
            for index in range(
                0,
                len(raw_records),
                self.batch_size,
            )
        ]

        for screen_batch in screen_batches:
            batch_screen = (
                self.batch_role_content_screen(
                    screen_batch
                )
            )
            screen_results.update(
                batch_screen
            )

        # A false screen result can discard an entire valid posting. Confirm
        # every negative decision in isolation to reduce batch-context false
        # negatives. Only two explicit false decisions may suppress extraction.
        for record in raw_records:
            record_id_str = str(
                record["id"]
            )

            if screen_results.get(
                record_id_str,
                True,
            ):
                continue

            confirmation = (
                self.batch_role_content_screen(
                    [
                        record,
                    ]
                )
            )

            if confirmation.get(
                record_id_str,
                True,
            ):
                logger.warning(
                    "Role-content screen false in batch but "
                    "true in isolated confirmation for %s; "
                    "retaining the record.",
                    record_id_str,
                )
                screen_results[
                    record_id_str
                ] = True

        approved_records: List[Dict[str, Any]] = []

        for record in raw_records:
            record_id_str = str(
                record["id"]
            )
            record_id = record["db_id"]

            has_role_content = (
                screen_results.get(
                    record_id_str,
                    True,
                )
            )

            if not has_role_content:
                logger.warning(
                    "Record %s screened as non-substantive role content, "
                    "but Patch 29 treats the LLM screen as advisory and "
                    "continues through deterministic evidence building.",
                    self._safe_id(
                        record_id
                    ),
                )

            # Patch 29 candidate: the LLM role-content gate is advisory rather
            # than destructive. Full-corpus QA showed that a false negative
            # here can erase an otherwise valid posting. Downstream evidence
            # filtering and grounded extraction still return [] for postings
            # with no substantive candidate-facing content.
            approved_records.append(
                record
            )

        logger.info(
            "Role-content screening retained %s/%s records.",
            len(approved_records),
            len(raw_records),
        )

        logger.info(
            "Step 2: Running NER extraction for retained records..."
        )

        ner_results: Dict[str, Dict[str, Any]] = {}

        for record in approved_records:
            record_id_str = str(
                record["id"]
            )
            record_id = record["db_id"]
            description = record["description"]

            try:
                ner_skills = self.extract_skills_with_ner(
                    description
                )

                ner_results[record_id_str] = {
                    "id": record_id,
                    "ner_skills": ner_skills,
                    "description": description,
                }

                logger.info(
                    f"Record {self._safe_id(record_id)}: "
                    f"NER extracted {len(ner_skills)} skills."
                )

            except Exception as exc:
                message = (
                    f"NER extraction failed: {exc}"
                )

                logger.error(
                    f"Record {self._safe_id(record_id)} "
                    f"{message}"
                )
                logger.error(
                    traceback.format_exc()
                )

                stats["failed"] += 1
                stats["failed_records"].append({
                    "id": record_id,
                    "error": message,
                })

        logger.info(
            "Step 3: Building full-description evidence chunks..."
        )

        chunk_records: List[Dict[str, Any]] = []
        chunk_parent: Dict[str, str] = {}
        chunk_literal_items: Dict[str, List[str]] = {}

        for record_id_str, data in ner_results.items():
            chunks = self.build_skill_evidence_chunks(
                data["description"],
                data["ner_skills"],
            )

            logger.info(
                "Record %s: %s evidence chunks.",
                self._safe_id(
                    data["id"]
                ),
                len(chunks),
            )

            for chunk_index, chunk_text in enumerate(
                chunks,
                start=1,
            ):
                chunk_id = (
                    f"{record_id_str}__chunk_"
                    f"{chunk_index:03d}"
                )

                # Local IDs such as [S0001] repeat in every record. When many
                # chunks are sent in one LLM batch, repeated IDs make accidental
                # cross-record evidence attribution possible. Scope every
                # evidence ID to its chunk before the text reaches the model.
                chunk_text = re.sub(
                    r"\[(S\d{4})\]",
                    lambda match: (
                        f"[{chunk_id}__{match.group(1)}]"
                    ),
                    chunk_text,
                )

                chunk_low = (
                    chunk_text.lower()
                )

                filtered_ner = [
                    skill
                    for skill in data["ner_skills"]
                    if (
                        isinstance(
                            skill,
                            str,
                        )
                        and skill.strip()
                        and skill.lower().strip()
                        in chunk_low
                    )
                ]

                chunk_records.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "ner": filtered_ner,
                })

                chunk_parent[chunk_id] = (
                    record_id_str
                )

                literal_items = (
                    self._extract_literal_list_skills_from_evidence(
                        chunk_text
                    )
                )

                chunk_literal_items[
                    chunk_id
                ] = literal_items

                if literal_items:
                    logger.info(
                        "Record %s chunk %s: recovered %s "
                        "explicit literal-list items.",
                        self._safe_id(
                            data["id"]
                        ),
                        chunk_index,
                        len(
                            literal_items
                        ),
                    )

        logger.info(
            "Step 4: Running segment-grounded LLM extraction "
            "across %s chunks...",
            len(chunk_records),
        )

        chunk_results: Dict[str, List[str]] = {}

        if (
            self.llm_client is not None
            and chunk_records
        ):
            batches = [
                chunk_records[
                    index:
                    index + self.batch_size
                ]
                for index in range(
                    0,
                    len(chunk_records),
                    self.batch_size,
                )
            ]

            logger.info(
                "Total grounded LLM batches: %s, "
                "maximum logical batch size: %s",
                len(batches),
                self.batch_size,
            )

            print(
                f"\n[Segment-Grounded LLM] "
                f"Total chunk batches: {len(batches)}\n"
            )

            for batch_index, batch in enumerate(
                batches,
                start=1,
            ):
                print(
                    f"[Chunk Batch {batch_index}/"
                    f"{len(batches)}] "
                    f"Processing {len(batch)} chunks..."
                )

                try:
                    batch_results = (
                        self.batch_llm_extract_grounded_chunks(
                            batch
                        )
                    )

                    chunk_results.update(
                        batch_results
                    )

                    nonempty = sum(
                        1
                        for skills
                        in batch_results.values()
                        if skills
                    )

                    print(
                        f"[Chunk Batch {batch_index}/"
                        f"{len(batches)}] "
                        f"✓ Done (non-empty chunks: "
                        f"{nonempty}/{len(batch)})"
                    )

                except Exception as exc:
                    logger.error(
                        "Chunk batch %s failed: %s",
                        batch_index,
                        exc,
                    )
                    logger.error(
                        traceback.format_exc()
                    )

                    for item in batch:
                        chunk_results.setdefault(
                            item["id"],
                            [],
                        )

        record_llm_results: Dict[str, List[str]] = {
            record_id_str: []
            for record_id_str in ner_results
        }

        for chunk_id, skills in chunk_results.items():
            parent_id = chunk_parent.get(
                chunk_id
            )

            if parent_id is None:
                continue

            record_llm_results[
                parent_id
            ].extend(
                skills
            )

        for chunk_id, literal_items in chunk_literal_items.items():
            parent_id = chunk_parent.get(
                chunk_id
            )

            if parent_id is None:
                continue

            record_llm_results[
                parent_id
            ].extend(
                literal_items
            )

        for record_id_str in record_llm_results:
            record_llm_results[
                record_id_str
            ] = self.merge_and_deduplicate_skills(
                [],
                record_llm_results[
                    record_id_str
                ],
            )

        logger.info(
            "Step 5: Formatting and updating the database..."
        )

        for record_id_str, data in ner_results.items():
            record_id = data["id"]
            ner_skills = data["ner_skills"]
            llm_skills = record_llm_results.get(
                record_id_str,
                [],
            )

            try:
                if self.llm_client is not None:
                    final_skills = (
                        self.merge_and_deduplicate_skills(
                            [],
                            llm_skills,
                        )
                    )
                else:
                    final_skills = (
                        self.merge_and_deduplicate_skills(
                            ner_skills,
                            [],
                        )
                    )

                if not final_skills:
                    logger.warning(
                        f"Record {self._safe_id(record_id)}: "
                        "no skills were extracted; storing an empty result."
                    )

                numbered_skills = self._add_numbering(
                    final_skills
                )

                logger.info(
                    f"Record {self._safe_id(record_id)}: "
                    f"{len(numbered_skills)} final skills "
                    f"(NER: {len(ner_skills)}, "
                    f"segment-grounded LLM: {len(llm_skills)})."
                )

                if self.update_record_skills(
                    record_id,
                    numbered_skills,
                    id_field,
                ):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["failed_records"].append({
                        "id": record_id,
                        "error": (
                            "Database update failed"
                        ),
                    })

            except Exception as exc:
                message = (
                    f"Merge/update failed: {exc}"
                )

                logger.error(
                    f"Record {self._safe_id(record_id)}: "
                    f"{message}"
                )
                logger.error(
                    traceback.format_exc()
                )

                stats["failed"] += 1
                stats["failed_records"].append({
                    "id": record_id,
                    "error": message,
                })

        logger.info(
            "Screened chunked segment-grounded processing completed."
        )
        logger.info(
            "Total: %s, Success: %s, Failed: %s",
            stats["total"],
            stats["success"],
            stats["failed"],
        )

        return stats



def main():
    print("=" * 50)
    print("Job Skill Extraction Pipeline (NER + Batched LLM)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        extractor = SkillsExtractor()
        results = extractor.process_all_records()

        print("\nProcessing completed!")
        print(f"Total records: {results['total']}")
        print(f"Successfully processed: {results['success']}")
        print(f"Failed: {results['failed']}")
        print("=" * 50)

    except Exception as e:
        print(f"\nERROR: Script execution failed: {e}")
        logger.error(f"Script execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
