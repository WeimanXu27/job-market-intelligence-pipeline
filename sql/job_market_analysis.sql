-- Job Market Intelligence: reproducible SQL analysis
-- Dialect: MySQL 8+
--
-- Assumes the analytics-ready CSV produced by:
--   analytics/build_analysis_dataset.py
-- has been loaded into a table named:
--   analysis_job_postings
--
-- Important semantics:
-- - search_query_title is the historical scraper search category, not a
--   verified occupation classification for each returned job posting.
-- - search_onet_code belongs to that search-category mapping.
-- - salary analyses use only salary_usable_for_analysis = TRUE.
-- - potential duplicates are flagged but not automatically removed.

USE job_market_portfolio;


-- ---------------------------------------------------------------------
-- 1. OVERVIEW KPIs
-- ---------------------------------------------------------------------

SELECT
    COUNT(*) AS total_postings,
    COUNT(DISTINCT NULLIF(company, '')) AS unique_companies,
    COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles,
    COUNT(DISTINCT NULLIF(search_query_title, '')) AS unique_search_queries,
    COUNT(DISTINCT NULLIF(search_market, '')) AS unique_search_markets,
    COUNT(DISTINCT NULLIF(job_city, '')) AS unique_job_cities,
    COUNT(DISTINCT NULLIF(job_county, '')) AS unique_job_counties,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows,
    ROUND(
        AVG(salary_usable_for_analysis = TRUE) * 100,
        2
    ) AS usable_salary_rate_pct
FROM analysis_job_postings;


-- ---------------------------------------------------------------------
-- 2. MEDIAN SALARY KPI
-- MySQL has no built-in MEDIAN aggregate, so use window functions.
-- ---------------------------------------------------------------------

WITH usable_salary AS (
    SELECT
        hourly_salary_clean,
        annual_salary_equivalent,
        ROW_NUMBER() OVER (
            ORDER BY hourly_salary_clean
        ) AS rn,
        COUNT(*) OVER () AS n
    FROM analysis_job_postings
    WHERE salary_usable_for_analysis = TRUE
      AND hourly_salary_clean IS NOT NULL
)
SELECT
    ROUND(AVG(hourly_salary_clean), 2) AS median_hourly_salary,
    ROUND(AVG(annual_salary_equivalent), 0) AS median_annual_equivalent
FROM usable_salary
WHERE rn IN (
    FLOOR((n + 1) / 2),
    FLOOR((n + 2) / 2)
);


-- ---------------------------------------------------------------------
-- 3. POSTING VOLUME BY MONTH
-- ---------------------------------------------------------------------

SELECT
    posting_month,
    COUNT(*) AS postings,
    COUNT(DISTINCT NULLIF(company, '')) AS unique_companies,
    COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows,
    ROUND(
        AVG(salary_usable_for_analysis = TRUE) * 100,
        2
    ) AS usable_salary_rate_pct
FROM analysis_job_postings
GROUP BY posting_month
ORDER BY posting_month;


-- ---------------------------------------------------------------------
-- 4. SEARCH QUERY COVERAGE
-- This describes the scraper search categories, not verified occupations.
-- ---------------------------------------------------------------------

SELECT
    search_query_title,
    search_onet_code,
    COUNT(*) AS postings,
    COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles,
    COUNT(DISTINCT NULLIF(company, '')) AS unique_companies,
    COUNT(DISTINCT NULLIF(job_city, '')) AS unique_job_cities,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows,
    ROUND(
        AVG(salary_usable_for_analysis = TRUE) * 100,
        2
    ) AS usable_salary_rate_pct
FROM analysis_job_postings
GROUP BY
    search_query_title,
    search_onet_code
ORDER BY postings DESC;


-- ---------------------------------------------------------------------
-- 5. SEARCH QUERY -> ACTUAL TITLE CONCENTRATION
-- Shows how heterogeneous the returned job titles are within each search
-- category. This is useful for explaining why search-category/O*NET-level
-- information may be too general for posting-specific skill analysis.
-- ---------------------------------------------------------------------

WITH title_counts AS (
    SELECT
        search_query_title,
        job_title,
        COUNT(*) AS title_postings
    FROM analysis_job_postings
    WHERE job_title IS NOT NULL
      AND job_title <> ''
    GROUP BY
        search_query_title,
        job_title
),
query_totals AS (
    SELECT
        search_query_title,
        COUNT(*) AS total_postings,
        COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles
    FROM analysis_job_postings
    GROUP BY search_query_title
),
ranked_titles AS (
    SELECT
        tc.search_query_title,
        tc.job_title,
        tc.title_postings,
        qt.total_postings,
        qt.unique_actual_job_titles,
        ROW_NUMBER() OVER (
            PARTITION BY tc.search_query_title
            ORDER BY
                tc.title_postings DESC,
                tc.job_title
        ) AS title_rank
    FROM title_counts tc
    JOIN query_totals qt
      ON tc.search_query_title = qt.search_query_title
)
SELECT
    search_query_title,
    total_postings,
    unique_actual_job_titles,
    job_title AS most_common_actual_job_title,
    title_postings AS most_common_title_postings,
    ROUND(
        title_postings / total_postings * 100,
        2
    ) AS most_common_title_share_pct
FROM ranked_titles
WHERE title_rank = 1
ORDER BY
    most_common_title_share_pct ASC,
    total_postings DESC;


-- ---------------------------------------------------------------------
-- 6. TOP 5 ACTUAL JOB TITLES WITHIN EACH SEARCH QUERY
-- ---------------------------------------------------------------------

WITH title_counts AS (
    SELECT
        search_query_title,
        job_title,
        COUNT(*) AS postings
    FROM analysis_job_postings
    WHERE job_title IS NOT NULL
      AND job_title <> ''
    GROUP BY
        search_query_title,
        job_title
),
ranked AS (
    SELECT
        search_query_title,
        job_title,
        postings,
        ROW_NUMBER() OVER (
            PARTITION BY search_query_title
            ORDER BY
                postings DESC,
                job_title
        ) AS title_rank
    FROM title_counts
)
SELECT
    search_query_title,
    title_rank,
    job_title,
    postings
FROM ranked
WHERE title_rank <= 5
ORDER BY
    search_query_title,
    title_rank;


-- ---------------------------------------------------------------------
-- 7. SEARCH MARKET -> ACTUAL GEOGRAPHY COVERAGE
-- ---------------------------------------------------------------------

SELECT
    search_market,
    COUNT(*) AS postings,
    COUNT(DISTINCT NULLIF(job_city, '')) AS unique_job_cities,
    COUNT(DISTINCT NULLIF(job_county, '')) AS unique_job_counties,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows
FROM analysis_job_postings
GROUP BY search_market
ORDER BY postings DESC;


-- ---------------------------------------------------------------------
-- 8. COUNTY-LEVEL MARKET SUMMARY
-- ---------------------------------------------------------------------

SELECT
    job_county,
    COUNT(*) AS postings,
    COUNT(DISTINCT NULLIF(company, '')) AS unique_companies,
    COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows,
    ROUND(
        AVG(salary_usable_for_analysis = TRUE) * 100,
        2
    ) AS usable_salary_rate_pct
FROM analysis_job_postings
WHERE job_county IS NOT NULL
  AND job_county <> ''
GROUP BY job_county
ORDER BY postings DESC;


-- ---------------------------------------------------------------------
-- 9. LARGEST EMPLOYERS IN THE OBSERVED DATASET
-- ---------------------------------------------------------------------

SELECT
    company,
    COUNT(*) AS postings,
    COUNT(DISTINCT NULLIF(search_query_title, '')) AS unique_search_queries,
    COUNT(DISTINCT NULLIF(job_title, '')) AS unique_actual_job_titles,
    COUNT(DISTINCT NULLIF(job_city, '')) AS unique_job_cities,
    SUM(salary_usable_for_analysis = TRUE) AS usable_salary_rows
FROM analysis_job_postings
WHERE company IS NOT NULL
  AND company <> ''
GROUP BY company
ORDER BY postings DESC
LIMIT 100;


-- ---------------------------------------------------------------------
-- 10. MEDIAN HOURLY SALARY BY SEARCH QUERY
-- Require at least 50 usable salary rows to reduce unstable comparisons.
-- ---------------------------------------------------------------------

WITH salary_rows AS (
    SELECT
        search_query_title,
        hourly_salary_clean,
        ROW_NUMBER() OVER (
            PARTITION BY search_query_title
            ORDER BY hourly_salary_clean
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY search_query_title
        ) AS n
    FROM analysis_job_postings
    WHERE salary_usable_for_analysis = TRUE
      AND hourly_salary_clean IS NOT NULL
),
query_medians AS (
    SELECT
        search_query_title,
        MAX(n) AS usable_salary_rows,
        AVG(hourly_salary_clean) AS median_hourly_salary
    FROM salary_rows
    WHERE rn IN (
        FLOOR((n + 1) / 2),
        FLOOR((n + 2) / 2)
    )
    GROUP BY search_query_title
)
SELECT
    search_query_title,
    usable_salary_rows,
    ROUND(median_hourly_salary, 2) AS median_hourly_salary,
    ROUND(
        median_hourly_salary * 2080,
        0
    ) AS median_annual_equivalent
FROM query_medians
WHERE usable_salary_rows >= 50
ORDER BY median_hourly_salary DESC;


-- ---------------------------------------------------------------------
-- 11. SALARY BY COUNTY
-- Require at least 50 usable salary rows.
-- ---------------------------------------------------------------------

WITH salary_rows AS (
    SELECT
        job_county,
        hourly_salary_clean,
        ROW_NUMBER() OVER (
            PARTITION BY job_county
            ORDER BY hourly_salary_clean
        ) AS rn,
        COUNT(*) OVER (
            PARTITION BY job_county
        ) AS n
    FROM analysis_job_postings
    WHERE salary_usable_for_analysis = TRUE
      AND hourly_salary_clean IS NOT NULL
      AND job_county IS NOT NULL
      AND job_county <> ''
),
county_medians AS (
    SELECT
        job_county,
        MAX(n) AS usable_salary_rows,
        AVG(hourly_salary_clean) AS median_hourly_salary
    FROM salary_rows
    WHERE rn IN (
        FLOOR((n + 1) / 2),
        FLOOR((n + 2) / 2)
    )
    GROUP BY job_county
)
SELECT
    job_county,
    usable_salary_rows,
    ROUND(median_hourly_salary, 2) AS median_hourly_salary,
    ROUND(
        median_hourly_salary * 2080,
        0
    ) AS median_annual_equivalent
FROM county_medians
WHERE usable_salary_rows >= 50
ORDER BY median_hourly_salary DESC;


-- ---------------------------------------------------------------------
-- 12. SALARY QUALITY / PIPELINE QA
-- This belongs in the case-study methodology, not necessarily the main
-- recruiter-facing dashboard.
-- ---------------------------------------------------------------------

SELECT
    salary_quality_flag,
    COUNT(*) AS rows_in_flag,
    ROUND(
        COUNT(*) / (
            SELECT COUNT(*)
            FROM analysis_job_postings
        ) * 100,
        2
    ) AS share_pct
FROM analysis_job_postings
GROUP BY salary_quality_flag
ORDER BY rows_in_flag DESC;


SELECT
    structured_salary_consistency,
    COUNT(*) AS rows_in_status
FROM analysis_job_postings
GROUP BY structured_salary_consistency
ORDER BY rows_in_status DESC;


-- ---------------------------------------------------------------------
-- 13. POTENTIAL DUPLICATES
-- Inspect only; do not automatically delete.
-- ---------------------------------------------------------------------

SELECT *
FROM analysis_job_postings
WHERE is_potential_duplicate = TRUE
ORDER BY
    posting_date,
    company,
    job_title;
