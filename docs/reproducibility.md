# Reproducibility and Data Availability

## Project Overview

This repository presents the **Job Market Intelligence Pipeline**, originating from 2025 internship at AllSumJobs.

The project focuses on collecting, processing, and analyzing job-posting data to understand labor-market trends, compensation patterns, geographic opportunities, and employer skill requirements.

The repository organizes the project's Python code, SQL queries, NLP skill extraction workflows, evaluation methodology, and analytical outputs into a public-facing structure.

To prepare the project for public release, sensitive information, private datasets, operational logs, and generated intermediate files have been excluded.

The repository is intended to make the project's methodology and engineering decisions accessible for review. It is not an exact copy of the original working environment or a self-contained, one-command reproduction of the full production pipeline.

## Environment and Dependencies

The repository is organized for a Python 3.11 environment.

Primary components include:

- Python for data processing, NLP, quality assurance, and analytics
- MySQL / AWS RDS for relational data storage and querying
- Pandas and NumPy for data preparation and analytical transformations
- Hosted and local language-model tooling for skill extraction and evaluation
- Tableau for interactive data visualization

Python dependencies are documented in the root `requirements.txt`.

The dependency versions were recorded from a local environment in October 2026 and do not necessarily represent the exact package versions used in the original internship environment.

Database access and applicable model workflows require credentials or configuration values that are not distributed in this repository.

The `.env.example` file documents configuration variables without exposing private credentials.

## Data Availability

### Included in the Public Repository

- Python source code for the main processing and analytical workflows
- SQL scripts for database operations and labor-market analysis
- NLP skill extraction and selected reliability-evaluation code
- Selected human-reviewed reference datasets in `data/reference/`
- Data-quality checks and analytical transformation logic
- Documentation describing the project's methodology and data dependencies

### Excluded from the Public Repository

- Original scraped job-posting data and full job descriptions
- Private MySQL / AWS RDS database contents
- Complete production and intermediate processed datasets
- Record-level experiment outputs and private evaluation manifests
- Operational logs, checkpoints, and temporary working files
- Database credentials, API keys, and local configuration
- Tableau workbook source files

These exclusions protect private project data and keep the public repository focused on source code, methodology, and analytical results.

The completed interactive Tableau dashboards are available separately through Tableau Public.

## Project Workflow

### 1. Job Market Data Preparation

Directory: `analytics/01_job_market/`

This component contains scripts for:

- Recovering historical job-posting records
- Auditing data quality and completeness
- Preparing analysis-ready datasets
- Performing exploratory job-market analysis
- Loading analytical datasets into MySQL

Reproducing this stage requires the original job-posting data, which is not publicly distributed.

### 2. NLP Skill Extraction and Production Quality Assurance

Directory: `analytics/02_skill_production/`

This component documents the skill extraction workflow, including:

- Preparing skill extraction input data
- Validating production prerequisites
- Running NLP-based skill extraction
- Investigating rate-limit and processing failures
- Applying controlled recovery procedures
- Auditing final production outputs

The production extractor is preserved in `src/skill_extraction.py`.

The production workflow depends on the corresponding database state, model access, and historical integrity checks.

### 3. Skill Extraction Reliability Evaluation

Directory: `analytics/03_skill_evaluation/`

This component contains selected scripts covering:

- Evaluation sample construction
- Development and validation datasets
- Blind-holdout preparation and freezing
- Candidate regression testing
- Blind evaluation
- Failure diagnosis
- Targeted reliability and architecture checks

The repository preserves representative experimental candidates and the associated evaluation methodology rather than every intermediate implementation.

The evaluated replacement candidates were not promoted because they did not demonstrate sufficiently stable generalization over the frozen production baseline.

The published skill analytics therefore remain based on the established production extractor.

### 4. Skills Intelligence and Analytical Data Preparation

Directory: `analytics/04_skills_intelligence/`

This component transforms NLP-derived skill information into analytical datasets used for labor-market intelligence.

Its responsibilities include:

- Skill vocabulary normalization
- Capability and qualification classification
- Context-based filtering and policy review
- Human-reviewed reference mappings
- California county standardization
- Preparation of Tableau-compatible analytical datasets

These transformations support the Skills Intelligence dashboard and depend on intermediate production outputs that are not included in the public repository.

## Data Integrity and Validation

The project includes validation procedures for data completeness, processing consistency, and analytical reliability.

Examples include:

- Record-count reconciliation
- Production source integrity checks using SHA-256
- Processing failure and recovery audits
- Evaluation-set separation and leakage prevention
- Human-reviewed normalization and filtering policies
- Preservation of the appropriate analytical population and denominator

The NLP-derived skill records are treated as analytical features rather than gold-standard human annotations.

Certain validation scripts reference historical record counts, dataset schemas, or source hashes. These checks are specific to the corresponding project artifacts and may not apply to arbitrary replacement data.

## Reproducibility Limitations

The repository provides the code and methodology necessary to inspect the major components of the project.

However, a full reproduction of the original analytical results requires:

- Access to the historical job-posting datasets
- The corresponding MySQL database and schema
- Appropriate API credentials and model access
- Required intermediate outputs
- Compatible execution environments and configurations

Because these resources are not fully public, cloning the repository alone is not sufficient to reproduce every production result.

The source code, analytical methodology, data-quality procedures, and published Tableau dashboards provide the primary means of reviewing the project's implementation and outcomes.

## Published Results

The project's final analytical outputs are presented through the **Job Market Intelligence** Tableau Public dashboard.

The dashboard contains four sections:

1. Market Overview
2. Compensation Landscape
3. Geographic Opportunity
4. Skills Intelligence

The repository README provides the dashboard link and an overview of the analytical findings.