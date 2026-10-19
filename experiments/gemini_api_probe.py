"""Gemini hosted-provider probe for job-skill extraction.

This standalone experiment documents a brief provider-swap prototype used to
test Gemini for the same general skill-expansion task as the hosted LLM path.
It is not a separate production pipeline or a formal benchmark.

The historical prototype used Gemini 1.5 Flash. The public reconstruction uses
a configurable current Gemini model so the probe remains runnable.
"""

import json
import os
from typing import List

from dotenv import load_dotenv
from google import genai


DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

SAMPLE_JOB_DESCRIPTION = """
We are looking for a Senior Software Engineer with expertise in:
- Python programming and data structures
- React and modern JavaScript frameworks
- SQL databases (MySQL, PostgreSQL)
- REST API design and development
- AWS cloud services

The ideal candidate should have strong problem-solving skills and excellent
communication abilities. Experience with Docker and Kubernetes is a plus.
""".strip()

SAMPLE_DETECTED_SKILLS = ["Python", "React", "SQL"]


def _strip_code_fence(response_text: str) -> str:
    """Remove a simple Markdown code fence from a model response."""
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


def build_skill_prompt(
    job_description: str,
    detected_skills: List[str],
) -> str:
    """Build the same style of skill-expansion prompt used in the prototype."""
    skills_text = ", ".join(detected_skills) if detected_skills else "None"

    return f"""Given the job description below and the list of skills already detected, extract any additional relevant technical or soft skills.

Job Description:
{job_description}

Detected Skills: {skills_text}

Return only a valid JSON array of additional skill strings.

Example:
["Docker", "Kubernetes", "Communication"]"""


def parse_skill_response(response_text: str) -> List[str]:
    """Parse and validate a JSON list returned by Gemini."""
    content = _strip_code_fence(response_text)

    if not content:
        raise RuntimeError("Gemini returned an empty response.")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini response was not valid JSON."
        ) from exc

    if not isinstance(parsed, list):
        raise RuntimeError(
            "Expected the Gemini response to be a JSON list."
        )

    return [
        skill.strip()
        for skill in parsed
        if isinstance(skill, str) and skill.strip()
    ]


def run_skill_extraction_probe() -> List[str]:
    """Run one synthetic skill-extraction request against Gemini."""
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is required to run the Gemini provider probe."
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        DEFAULT_GEMINI_MODEL,
    )

    client = genai.Client(api_key=api_key)

    prompt = build_skill_prompt(
        SAMPLE_JOB_DESCRIPTION,
        SAMPLE_DETECTED_SKILLS,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    response_text = response.text or ""
    skills = parse_skill_response(response_text)

    print(f"Gemini provider probe succeeded with model: {model_name}")
    print(f"Validated additional skills: {len(skills)}")

    for index, skill in enumerate(skills, start=1):
        print(f"{index}. {skill}")

    return skills


def main():
    """Run the standalone Gemini provider probe."""
    try:
        run_skill_extraction_probe()
    except Exception as exc:
        print(f"Gemini provider probe failed: {exc}")
        raise


if __name__ == "__main__":
    main()
