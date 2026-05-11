# Data Analysis Report: SHL Product Catalog

## High-Level Analysis Plan
1. **Step 1: Data Loading & Basic Inspection** - Identify data shape, column types, and missing values.
2. **Step 2: Univariate Analysis** - Explore distributions of categories, job levels, languages, remote/adaptive status, and durations.
3. **Step 3: Bivariate/Multivariate Analysis** - Analyze relationships between features (e.g., job levels across keys, duration by test type).
4. **Step 4: Text Analysis** - Extract insights from descriptions to find common topics or tested skills.

---

## Step 1: Data Loading & Basic Inspection
**Goal**: Load the data and understand its basic properties.

**Findings**:
* **Data Shape**: The dataset contains 377 products and 15 columns.
* **Data Types**: The data consists mainly of strings and list-based categorical data (`job_levels`, `languages`, `keys`).
* **Data Completeness**: 
  * There are 0 missing `null` values.
  * Some entries have empty data (empty strings or lists):
    * `job_levels`: 19 missing
    * `languages`: 37 missing
    * `duration`: 61 missing
  * Core attributes like `description`, `name`, `status`, `remote`, `adaptive`, and `keys` are 100% complete for all 377 products.

**Conclusion**: The dataset is relatively small and very clean, making it suitable for direct analysis without extensive imputation, though we need to be mindful of the missing durations and job levels during deeper analysis.

---

## Step 2: Univariate Analysis
**Goal**: Examine the distribution of the key variables in isolation to understand the standard SHL product offering.

**Findings**:
1. **Remote and Adaptive Capabilities**:
   * **100%** of the assessments (377) support **remote** administration.
   * Only ~10% (37 products) are **adaptive**, meaning the vast majority (340) are standard fixed-form assessments.
2. **Product Categories ('Keys')**:
   * The catalog is heavily skewed toward **"Knowledge & Skills"** testing (240 occurrences).
   * The next most common categories are "Personality & Behavior" (67) and "Simulations" (43).
3. **Target Job Levels**:
   * "Mid-Professional" (304) and "Professional Individual Contributor" (296) are the dominant targets, indicating these tests are primarily meant for experienced non-managerial roles.
   * Entry-level/Graduate/Managerial levels appear in about ~110-140 tests each.
4. **Language Support**:
   * **English (USA)** is nearly ubiquitous (321 tests).
   * Other top languages like English International, Spanish, French, and Italian are supported by 40-75 tests each.
5. **Test Duration**:
   * Of the 292 tests with reported durations, the **average is ~13.6 minutes**, with a median of 10 minutes.
   * Most tests are fast (75% are 16 minutes or shorter), while the longest assessment takes 60 minutes.

**Conclusion**: The typical SHL product in this dataset is a non-adaptive, remote, ~10-15 minute "Knowledge & Skills" assessment aimed at Mid-Professionals in US English.

---

## Step 3: Bivariate/Multivariate Analysis
**Goal**: Understand the relationships between different variables (like how duration varies by test type or how categories differ by job level).

**Findings**:
1. **Duration by Adaptive vs. Non-Adaptive**:
   * **Adaptive** tests are significantly longer, with an average duration of **23.2 minutes** (median 24m) compared to non-adaptive tests at **12.2 minutes** (median 10m).
2. **Duration by Test Category**:
   * **Simulations** (25.4 mins), **Personality & Behavior** (23.4 mins), and **Ability & Aptitude** (22.4 mins) are consistently the longest test types.
   * Conversely, the most common category, **Knowledge & Skills**, takes only **11.4 minutes** on average.
3. **Assessment Category Focus Shifts Drastically by Job Level**:
   * **Entry-Level**: Fairly balanced across Knowledge & Skills (29.5%), Personality (22.1%), Simulations (20.1%), and Aptitude (13.4%).
   * **Mid-Professional**: Heavily over-indexes on hard skills (65.5% Knowledge & Skills) with minimal focus on Personality (15.5%).
   * **Manager**: The focus begins to shift back toward soft skills (37.6% Personality) and away from hard skills (21.3% Knowledge).
   * **Executive**: The paradigm completely flips; **61.8%** of executive tests focus on **Personality & Behavior**, and a mere 1.5% focus on Knowledge & Skills.

**Conclusion**: There is a clear strategic structure to the SHL catalog. Short, non-adaptive "Knowledge & Skills" tests are mass-deployed for Mid-Professionals. Meanwhile, more complex, longer assessments (often Adaptive or Simulation-based) are used for "Personality & Behavior" and "Aptitude". Most notably, as the targeted job level increases (from Mid-Professional up to Executive), the focus drastically shifts from hard technical skills to behavioral and personality traits.

---

## Step 4: Text Analysis
**Goal**: Mine the unstructured `name` and `description` text to uncover common terminology, assessment formats, and technical focus areas.

**Findings**:
1. **Product Naming Trends**:
   * The word **"New"** appears in 219 product names, suggesting a recent overhaul or massive update to their catalog.
   * Prominent test subject words in titles include: **"engineering"** (28), **"sales"** (15), **"net"** (10), and **"java"** (10), confirming a large tech/engineering focus.
   * **"OPQ"** (Occupational Personality Questionnaire) is heavily featured (22 times).
2. **Standard Format Exposed in Descriptions**:
   * The words **"multi"** and **"choice"** appear ~190 times in the descriptions. This confirms that roughly half of the catalog consists of standard multiple-choice exams.
3. **Validating the "Executive vs. Mid-Professional" Divide**:
   * **Mid-Professional Descriptions**: Dominated by words like "multi", "choice", "management", and "data". This validates that these are largely multiple-choice knowledge exams.
   * **Executive Descriptions**: Dominated by words like "report", "OPQ", "individual", "competency", "questionnaire", and "personality". The format words ("multi-choice") disappear completely, replaced by psychometric terms ("questionnaire", "personality").

**Conclusion**: Text analysis perfectly corroborates our findings from Step 3. The SHL catalog is broadly split into two camps: technical/functional multiple-choice exams for individual contributors, and deep psychometric/personality questionnaires (like the OPQ) generating individual competency reports for executives.
