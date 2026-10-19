# Skill Extraction Reliability Evaluation

## Overview

Reliable skill extraction is essential for labor-market analysis. Inaccurate or inconsistent extraction can distort reported skill demand, particularly when job descriptions contain a mixture of responsibilities, qualifications, application instructions, and employer information.

This document explains the reliability-evaluation methodology associated with the Job Market Intelligence Pipeline, including candidate testing, blind-holdout controls, failure analysis, and the decision to retain the established production extractor.

The objective was not simply to extract more skills, but to determine whether an alternative extraction approach could deliver more reliable results without introducing new systematic errors.

## 1. Production Baseline

The established production extractor is:

[`src/skill_extraction.py`](../src/skill_extraction.py)

This extractor produced the skill features used in the project's published analytical outputs and Tableau Skills Intelligence dashboard.

The production implementation is preserved without changes to its source content. Its SHA-256 fingerprint is:

`e07f0fcf11e4d5ce0c9028590bb550e668ae722feeb058739e7b51f38fb05dc4`

Experimental candidates were evaluated separately and were not used to replace the production results.

Extracted skills are treated as **model-derived analytical features**, not ground-truth human annotations.

## 2. Evaluation Methodology

The reliability-evaluation process separated development and diagnostic data from untouched blind-holdout samples.

### Exposed Regression Evaluation

Previously exposed records were used to investigate extraction failures, test candidate revisions, and identify recurring error mechanisms.

The established evaluation pool contained 935 exposed records.

These records supported iterative development, but improvements on them were not considered sufficient evidence of generalization.

### Frozen Blind Evaluation

Candidate source files and evaluation datasets were frozen before blind inference.

Two separate blind evaluations were conducted:

| Evaluation | Holdout size | Purpose |
|---|---:|---|
| Initial blind evaluation | 288 records | Evaluate the first frozen replacement candidate |
| Replacement blind evaluation | 288 records | Test a revised candidate on a separate untouched sample |

Both evaluations used recorded source and holdout SHA-256 fingerprints.

The two blind sets were disjoint. A third fresh 288-record holdout was also prepared separately and remained unused in the retained evaluation workflow.

The purpose of these controls was to reduce the risk of repeatedly optimizing against the same evaluation examples.

### Failure Analysis

Evaluation considered more than aggregate skill counts.

Relevant failure categories included:

- False-zero extraction from descriptions containing meaningful role evidence
- Severe partial under-extraction
- Application or document-submission instructions incorrectly extracted as skills
- Pre-employment conditions and employer-context leakage
- Grounding and validation failures
- Repeated error mechanisms across unrelated postings

A zero-skill output or a change in extracted skill count was not automatically treated as an error. Source context and diagnostic evidence were necessary to interpret the result.

## 3. Representative Candidate Designs

The public repository retains four representative experimental implementations rather than every intermediate development version.

| Source file | Engineering focus |
|---|---|
| [`src/reliability_first_blind_candidate.py`](../src/reliability_first_blind_candidate.py) | First formally evaluated replacement candidate |
| [`src/reliability_replacement_blind_candidate.py`](../src/reliability_replacement_blind_candidate.py) | Revised candidate for replacement blind evaluation |
| [`src/reliability_rule_hardened_candidate.py`](../src/reliability_rule_hardened_candidate.py) | Rule-based precision hardening |
| [`src/reliability_semantic_gate_candidate.py`](../src/reliability_semantic_gate_candidate.py) | Independent semantic review of extracted proposals |

These represent distinct evaluation or architectural checkpoints. They are not four production releases.

### Initial and Replacement Blind Candidates

The first candidate was frozen after exposed regression testing and evaluated against the initial untouched blind set.

Post-evaluation diagnostics identified weaknesses that were not adequately resolved by the earlier exposed tests.

A revised candidate was subsequently tested against a separately frozen replacement blind set. This evaluation also revealed important remaining issues, including extraction omissions and leakage of non-skill information.

The inference summaries recorded 48 zero-skill outputs in the first blind evaluation and 47 in the replacement blind evaluation. These figures describe output counts, not the number of confirmed extraction errors.

Neither candidate demonstrated sufficiently reliable generalization to justify replacing the production baseline.

### Rule-Based Precision Hardening

Subsequent experiments investigated specific precision failures using rule and phrase-context refinements.

This approach was useful for diagnosing known failure patterns. However, improving exposed examples could not, by itself, establish reliability on previously unseen job descriptions.

The retained rule-hardening candidate represents this engineering approach.

### Independent Semantic Gate

Another architectural approach introduced independent semantic evaluation between grounded skill extraction and final validation.

The retained design follows:

**Grounded extraction → Independent semantic gate → Deterministic validation**

Rather than trusting job-description section labels to determine whether an extracted item is a skill, the semantic gate evaluates every structurally grounded proposal.

Proposals are classified into categories such as:

- Role capability or experience
- Required credential
- Application or proof process
- Employment condition
- Employer context
- Unknown

This separates substantive job requirements from instructions about applying for a job, employment eligibility, and information about the employer.

The design addresses a specific limitation discovered in an earlier independent-adjudication experiment: incorrectly labeled responsibility sections could allow non-skill information to bypass semantic review.

## 4. Targeted Diagnostic Findings

Selected architecture experiments were compared using ten previously exposed blocker records.

| Diagnostic measure | Rule-hardened candidate | Independent semantic-gate candidate |
|---|---:|---:|
| Exposed blocker records | 10 | 10 |
| Records containing exact targeted noise | 0 | 1 |
| Overlap with fresh blind holdout | 0 | 0 |

In the semantic-gate experiment, all 442 structurally grounded proposals were routed for semantic evaluation, with no parser-section bypass.

By comparison, the preceding independent-adjudicator design had allowed 183 proposals to bypass its secondary review on the same ten-record diagnostic scope.

This supported the architectural rationale for removing section-label-based bypasses.

However, these findings have important limitations:

- The ten records were already exposed.
- The diagnostic results did not establish performance across the full evaluation population.
- A targeted noise counter does not measure overall precision, recall, or extraction usefulness.
- The rule-hardened and semantic-gate candidates represented different engineering approaches, not a completed head-to-head blind comparison.

Therefore, improved behavior on a targeted diagnostic set was not treated as sufficient evidence for production promotion.

## 5. Production Decision

**The experimental replacement was not promoted to production because it did not demonstrate sufficiently stable generalization over the frozen baseline.**

This decision followed a distinction between improvements on known examples and demonstrated reliability on independent evaluation data.

The retained production extractor remained the source for downstream skill analytics.

The evaluation work provides evidence of several engineering practices:

- Separating development and evaluation datasets
- Freezing candidates before blind testing
- Using source fingerprints to protect evaluation integrity
- Diagnosing recurring failure mechanisms
- Comparing rule-based and architecture-level interventions
- Avoiding further tuning when the evidence did not justify promotion

The decision not to replace a working production pipeline was itself a reliability and data-quality judgment.

## 6. Code and Evidence Organization

The public repository separates production workflows from reliability evaluation.

**Production code**

- [`src/skill_extraction.py`](../src/skill_extraction.py)
- [`analytics/02_skill_production/`](../analytics/02_skill_production/)

**Evaluation code**

- [`analytics/03_skill_evaluation/`](../analytics/03_skill_evaluation/)
- Four representative candidate sources in [`src/`](../src/)

The evaluation directory contains scripts for sample construction, blind-holdout freezing, exposed regression, blind inference, failure diagnosis, and targeted architecture checks.

The repository preserves the representative code and methodology without publishing every intermediate candidate, private record-level result, or operational artifact.

Original freeze manifests and detailed evaluation outputs remain in the private project archive. Public source filenames were simplified for readability; the underlying retained candidate source contents were not changed.

## 7. Limitations and Reproducibility

The public repository supports inspection of the implementation and evaluation design, but it does not include the private job descriptions, complete evaluation datasets, or historical database state required to reproduce all experiment results.

The retained source files and documented integrity checks provide an auditable technical record within those limits.

For environment requirements, private-data boundaries, and workflow dependencies, see [Reproducibility and Data Availability](reproducibility.md).