-- Build the typed analytics table from the all-TEXT staging table.
-- Dialect: MySQL 8+
--
-- Input:
--   job_market_portfolio.analysis_job_postings_staging
--
-- Output:
--   job_market_portfolio.analysis_job_postings
--
-- The staging table is intentionally all TEXT so CSV ingestion is lossless.
-- This script applies explicit data types and boolean/numeric conversions.

USE job_market_portfolio;

DROP TABLE IF EXISTS analysis_job_postings;

CREATE TABLE analysis_job_postings (
    record_id VARCHAR(32) NOT NULL,
    posting_date DATE NOT NULL,
    posting_month CHAR(7) NOT NULL,

    company VARCHAR(512) NULL,
    job_title TEXT NULL,
    search_query_title VARCHAR(255) NULL,
    search_onet_code VARCHAR(32) NULL,

    location_raw VARCHAR(512) NULL,
    search_market VARCHAR(255) NULL,
    job_city VARCHAR(255) NULL,
    job_county VARCHAR(255) NULL,

    salary_job_type_raw DECIMAL(14, 4) NULL,
    hourly_salary_raw DECIMAL(12, 4) NULL,
    min_salary_raw DECIMAL(14, 4) NULL,
    max_salary_raw DECIMAL(14, 4) NULL,
    salary_unit VARCHAR(16) NULL,

    structured_salary_midpoint_raw DECIMAL(14, 4) NULL,
    structured_hourly_equivalent DECIMAL(12, 4) NULL,
    structured_salary_consistency VARCHAR(32) NOT NULL,
    salary_quality_flag VARCHAR(32) NOT NULL,
    salary_usable_for_analysis BOOLEAN NOT NULL,

    hourly_salary_clean DECIMAL(12, 4) NULL,
    annual_salary_equivalent DECIMAL(14, 4) NULL,

    is_potential_duplicate BOOLEAN NOT NULL,
    url TEXT NULL,

    PRIMARY KEY (record_id),

    INDEX idx_posting_month (posting_month),
    INDEX idx_search_query_title (search_query_title),
    INDEX idx_search_onet_code (search_onet_code),
    INDEX idx_search_market (search_market),
    INDEX idx_job_county (job_county),
    INDEX idx_salary_usable (salary_usable_for_analysis)
)
CHARACTER SET utf8mb4
COLLATE utf8mb4_bin;


INSERT INTO analysis_job_postings (
    record_id,
    posting_date,
    posting_month,
    company,
    job_title,
    search_query_title,
    search_onet_code,
    location_raw,
    search_market,
    job_city,
    job_county,
    salary_job_type_raw,
    hourly_salary_raw,
    min_salary_raw,
    max_salary_raw,
    salary_unit,
    structured_salary_midpoint_raw,
    structured_hourly_equivalent,
    structured_salary_consistency,
    salary_quality_flag,
    salary_usable_for_analysis,
    hourly_salary_clean,
    annual_salary_equivalent,
    is_potential_duplicate,
    url
)
SELECT
    TRIM(record_id),

    STR_TO_DATE(
        NULLIF(TRIM(posting_date), ''),
        '%Y-%m-%d'
    ),

    NULLIF(TRIM(posting_month), ''),

    NULLIF(TRIM(company), ''),
    NULLIF(TRIM(job_title), ''),
    NULLIF(TRIM(search_query_title), ''),
    NULLIF(TRIM(search_onet_code), ''),

    NULLIF(TRIM(location_raw), ''),
    NULLIF(TRIM(search_market), ''),
    NULLIF(TRIM(job_city), ''),
    NULLIF(TRIM(job_county), ''),

    CAST(
        NULLIF(TRIM(salary_job_type_raw), '')
        AS DECIMAL(14, 4)
    ),

    CAST(
        NULLIF(TRIM(hourly_salary_raw), '')
        AS DECIMAL(12, 4)
    ),

    CAST(
        NULLIF(TRIM(min_salary_raw), '')
        AS DECIMAL(14, 4)
    ),

    CAST(
        NULLIF(TRIM(max_salary_raw), '')
        AS DECIMAL(14, 4)
    ),

    NULLIF(TRIM(salary_unit), ''),

    CAST(
        NULLIF(
            TRIM(structured_salary_midpoint_raw),
            ''
        )
        AS DECIMAL(14, 4)
    ),

    CAST(
        NULLIF(
            TRIM(structured_hourly_equivalent),
            ''
        )
        AS DECIMAL(12, 4)
    ),

    TRIM(structured_salary_consistency),
    TRIM(salary_quality_flag),

    CASE
        WHEN LOWER(
            TRIM(salary_usable_for_analysis)
        ) IN ('true', '1', 'yes', 'y')
            THEN TRUE
        ELSE FALSE
    END,

    CAST(
        NULLIF(TRIM(hourly_salary_clean), '')
        AS DECIMAL(12, 4)
    ),

    CAST(
        NULLIF(
            TRIM(annual_salary_equivalent),
            ''
        )
        AS DECIMAL(14, 4)
    ),

    CASE
        WHEN LOWER(
            TRIM(is_potential_duplicate)
        ) IN ('true', '1', 'yes', 'y')
            THEN TRUE
        ELSE FALSE
    END,

    NULLIF(TRIM(url), '')

FROM analysis_job_postings_staging;


-- ---------------------------------------------------------------------
-- Validation
-- Expected values from the Python pipeline:
-- rows                  = 65247
-- unique_record_ids     = 65247
-- usable_salary_rows    = 55562
-- potential_duplicates  = 2
-- ---------------------------------------------------------------------

SELECT
    COUNT(*) AS total_rows,
    COUNT(DISTINCT record_id) AS unique_record_ids,
    SUM(salary_usable_for_analysis) AS usable_salary_rows,
    SUM(is_potential_duplicate) AS potential_duplicate_rows
FROM analysis_job_postings;


-- Check the final schema and types.
DESCRIBE analysis_job_postings;
