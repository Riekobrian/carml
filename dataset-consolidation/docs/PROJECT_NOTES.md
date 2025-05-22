# Project Notes and Memory Bank

This file is intended to store critical insights, important decisions, and any other noteworthy items during the development of the Used Car Price Prediction Model. 

## Phase 1: Data Preparation and Advanced Feature Engineering

### Step 1.1: Load and Verify Final Dataset

*   **Data Quality Issue Identified (2025-05-14):** The `year_of_manufacture` column contains invalid entries with a value of 0. This was detected during initial validation checks (`main_pipeline.py`). This will need to be specifically addressed when engineering the `car_age` feature in Step 1.4 to avoid errors or nonsensical age calculations. 
*   **Currency Handling (2025-05-14):** `currency_code` contains 'USD' (9624), 'KSh' (3223), and 620 NaNs. NaNs imputed with 'USD' (mode) in `main_pipeline.py`. **Action Item:** Consider implementing price conversion to a single currency (e.g., KSh) during feature engineering if reliable conversion rates can be obtained.
*   **Mileage Unit Handling (2025-05-14):** `mileage_unit` is predominantly 'km' (13361) with 106 NaNs. NaNs imputed with 'km' (mode) in `main_pipeline.py`.
*   **Row Removal (2025-05-14):** Dropped rows where both `make_name` AND `model_name` were 'Unknown_Placeholder' to remove records with missing critical identifiers. Implemented in `main_pipeline.py`.

*   **Further Data Cleaning & Imputation for Step 1.4 (Feature Engineering) (2025-05-14):**
    *   **`price`**: Address 622 NaNs. Implement currency conversion (KSh to USD or vice-versa).
    *   **`mileage`**: Address 552 NaNs.
    *   **Other Categorical Features with NaNs to Impute/Handle:**
        *   `fuel_type` (5 NaNs)
        *   `transmission` (21 NaNs)
        *   `drive_type` (161 NaNs)
        *   `body_type` (418 NaNs)
        *   `usage_type` (406 NaNs)
        *   `location` (1055 NaNs)
        *   `exterior_color` (987 NaNs)
    *   **High NaN Count Features for Strategic Handling (Impute, Categorize Unknown, or Drop in Step 1.4):**
        *   `interior_color` (~12.5k NaNs)
        *   `annual_insurance` (~10.7k NaNs)
        *   `availability_status` (~10k NaNs)
        *   `source_specific_id` (526 NaNs)

### Step 1.4.2: EDA Insights and Feature Engineering Plans (Consolidated from Iteration)

This section summarizes key decisions and plans for feature engineering based on EDA.

**Target Variable (`price`):**
*   Highly right-skewed. **Action:** Apply `np.log1p` transformation to create `price_log`. This will be the target for modeling.

**Numeric Features:**

*   **`year_of_manufacture` / `car_age`:**
    *   Contains `0` values (N=322). **Action:** Treat `0`s as NaNs. Impute all NaNs (original and converted `0`s) using the median of valid years (>1950). Create `car_age` (`current_year - year_of_manufacture_num`) and `car_age_squared`. Handle `car_age < 0` by setting to `0`.
*   **`mileage`:**
    *   Right-skewed (skewness: 3.148). Contains 169 NaNs and some `0` values. Negative correlation with `price_log`.
    *   **Action:** Impute NaNs with median. Apply `np.log1p` to create `mileage_log`. Engineer `mileage_per_year`. (Decision: `0` mileage values will be handled by `log1p` and median imputation if they become NaN after `pd.to_numeric`.)
*   **`engine_size_cc`:**
    *   Highly right-skewed (skewness: 5.167). Contains `0` values and extreme outliers (e.g., 46008cc). Positive correlation with `price_log`.
    *   **Action:** Treat `0`s as NaNs. Impute all NaNs with median. Apply `np.log1p` to create `engine_size_cc_log`. (Decision: Extreme outliers will be Winsorized or capped if log transform and robust scaling are insufficient, for now relying on log).
*   **`horse_power`:**
    *   Right-skewed (skewness: 2.167). Contains extreme outliers (e.g., 1841 HP). Positive correlation with `price_log`.
    *   **Action:** Impute NaNs with median. Apply `np.log1p` to create `horse_power_log`. (Decision: Extreme outliers will be Winsorized or capped).
*   **`torque`:**
    *   Right-skewed (skewness: 1.284). Positive correlation with `price_log`.
    *   **Action:** Impute NaNs with median. Apply `np.log1p` to create `torque_log`.
*   **`acceleration`:**
    *   Fairly symmetrical (skewness: 0.286). Weak negative correlation with `price_log`.
    *   **Action:** Impute NaNs with median. Keep as `acceleration_num`. (Log transform considered but deemed not immediately necessary).
*   **`seats`:**
    *   Mostly 5-seaters. Distribution is reasonable.
    *   **Action:** Impute NaNs with median. Keep as `seats_num` (integer).

**Categorical Features:**

*   **General Cleaning:** All categorical features will be lowercased, whitespace stripped. Common missing value placeholders ('na', 'n/a', '-', 'unknown', etc.) converted to `np.nan`.
    *   For most categoricals, `np.nan` will then be filled with the string "unknown".
    *   **Exception:** `condition` and `usage_type` NaNs will be handled by their specific user-defined recoding logic.

*   **`fuel_type`:**
    *   Synonyms like "Petrol" vs "Petroleum", various "Hybrid" forms.
    *   **Action:** Consolidate into standard categories: `petrol`, `diesel`, `hybrid_petrol`, `hybrid_diesel`, `plugin_hybrid_petrol`, `electric`, `unknown`. Create `fuel_type_cleaned`.
*   **`transmission`:**
    *   Synonyms like "At" vs "Automatic", "Mt" vs "Manual", specific speed MTs (5MT, 6MT), Duonic, Smoother.
    *   **Action:** Consolidate: `automatic`, `manual`, `automated_manual` (for Duonic, Smoother etc.), `unknown`. Create `transmission_cleaned`.
*   **`drive_type`:**
    *   Variations like "2WD", "4WD", "AWD", "FF", "FR", "FWD", "RWD". Erroneous "04-FEB" value.
    *   **Action:** Consolidate: `2wd_front`, `2wd_rear`, `2wd_mid_engine`, `2wd_rear_engine`, `4wd`, `awd`, `unknown`. Erroneous values to `unknown`. Create `drive_type_cleaned`. (Further simplification to just '2wd' for all 2wd types is an option).

*   **NEW RECODING (User Specified): `condition` and `usage_type`**
    *   **`condition` Recoding:**
        *   **Goal:** Simplify to a binary 'accident-related' status.
        *   **Previous State:** Mixed descriptive (Excellent, Very Good), numeric grades (4.5, 5), status terms (Foreign Used, Locally Used, New), and NaNs.
        *   **New Action:** Create `condition_clean` with two categories:
            *   `"Accident free"`: Maps from "Foreign Used", "Excellent", "Very Good", "4.5", "Locally Used", "5", "Ready For Import", "6", "New", "4", and `NaN`.
            *   `"Accident involved"`: All other original `condition` values.
        *   **NaN Handling:** Original `NaN` values in `condition` are explicitly mapped to `"Accident free"`.
    *   **`usage_type` Recoding:**
        *   **Goal:** Simplify to a binary origin/history status.
        *   **Previous State:** Mixed terms (Foreign Used, Kenyan Used, Used, Locally Used, Brand New, Ready For Import, New) and NaNs.
        *   **New Action:** Create `usage_type_clean` with two categories:
            *   `"Foreign Used"`: Maps from "Foreign Used", "New", "Ready For Import", "Brand New".
            *   `"Kenyan Used"`: Maps from "Kenyan Used", "Locally Used", "Used", and `NaN`.
        *   **NaN Handling:** Original `NaN` values in `usage_type` are explicitly mapped to `"Kenyan Used"`.
        *   Values not fitting either list (and not NaN) will also default to "Kenyan Used" with a warning.
    *   **Implication:** The more granular information previously planned for `condition_physical` and the more detailed `usage_type_cleaned` categories will be replaced by these binary features. The original `condition` and `usage_type` columns will be dropped.

*   **`body_type`:**
    *   Synonyms like "Saloon" vs "Sedan", "Mini Van" vs "Van", various Pickups. High number of categories.
    *   **Action:** Consolidate synonyms (e.g., `sedan`, `van_minivan`, `wagon_estate`, `pickup_truck`, `suv`, `bus`, `coupe`, `hatchback`, `convertible`, `special_purpose_truck`, `unknown`). Create `body_type_cleaned`. Consider grouping very rare types into "other_body_type".
*   **`make_name`:**
    *   High cardinality.
    *   **Action:** Group rare makes (e.g., appearing < 0.1% of total or < 10 times) into "other_make". Create `make_name_cleaned`.
*   **`model_name`:**
    *   Very high cardinality (999+ unique).
    *   **Action:** Aggressively group rare models (e.g., appearing < 0.05% of total or < 5 times) into "other_model". Create `model_name_cleaned`.
*   **`source`:** (Original data source identifier)
    *   **Action:** After lowercasing and stripping, if it provides valuable categorical info not captured elsewhere, keep as is. Otherwise, it might be dropped if redundant or too noisy. Currently included in generic cleaning and NaN filling.

**Interaction and Boolean Features:**
*   **Action:**
    *   `power_per_cc = horse_power_num / engine_size_cc_num` (handle division by zero, impute NaNs).
    *   `mileage_per_cc = mileage_num / engine_size_cc_num` (handle division by zero, impute NaNs).
    *   `is_luxury_make`: Boolean flag based on `make_name_cleaned`.
    *   `make_model_cleaned = make_name_cleaned + '_' + model_name_cleaned`.

**Columns to Drop Post-Engineering:**
*   Original numeric columns (`year_of_manufacture`, `mileage`, etc.).
*   Original categorical columns (`fuel_type`, `transmission`, `condition`, `usage_type`, etc.).
*   Intermediate helper columns (`year_of_manufacture_num`).
*   Original `price` column.
*   Other irrelevant IDs or unprocessed text columns (e.g., `unnamed:_0`, `source_specific_id`).

---

## EDA Insights (Observed Post-`main_pipeline.py` Execution, Pre-Step 1.4 Engineering)

**Overall Critical Finding: Dual Price Banding**
*   **Description:** A consistent observation of two distinct horizontal price bands in `price_log` plots across ALL numeric features (`mileage`, `engine_size_cc`, `horse_power`, `torque`, `acceleration`) and persisting *within* major categories of ALL categorical features analyzed (`seats`, `make_name`, `model_name`, `fuel_type`, `transmission`, `drive_type`, `condition`, `usage_type`).
*   **Hypothesis:** Strong indication of unresolved currency differences (e.g., KSh vs. USD prices not fully standardized to a single currency despite `currency_code` imputation in `main_pipeline.py`) or fundamental market segmentation not yet captured by existing features.
*   **Action (CRITICAL - Before Step 1.4 Feature Engineering):**
    1.  Thoroughly investigate the original `price` and `currency_code` columns from `cars_modeling_input.csv`.
    2.  Verify the effectiveness of the currency imputation and any implicit/explicit price conversion logic in `main_pipeline.py`.
    3.  Implement robust price standardization to a single currency (e.g., KSh) as a top priority. This might involve revisiting the raw data sources if currency information is ambiguous.

**Numeric Features Insights (Post `main_pipeline.py`):**
*   **`mileage`**:
    *   Right-skewed (skewness: 3.148). Plan: Consider log transformation (`np.log1p`).
    *   Contains 0 mileage entries. Plan: Investigate if these represent new cars, data errors, or placeholders.
    *   169 NaNs observed (these are numeric NaNs after `pd.to_numeric(errors='coerce')` in EDA; original `main_pipeline.py` input has 552 NaNs to address). Plan: Impute (median suggested due to skew).
    *   Clear negative correlation with `price_log`.
*   **`engine_size_cc`**:
    *   Heavily right-skewed (skewness: 5.167). Plan: Consider log transformation.
    *   Contains 0cc entries. Plan: Investigate (errors, placeholders, specific vehicle types like EVs?).
    *   Extreme max value (46008cc). Plan: Treat as an outlier (cap or remove).
    *   0 NaNs observed (original data was numeric or coerced cleanly).
    *   Clear positive correlation with `price_log`.
*   **`horse_power`**:
    *   Right-skewed (skewness: 2.167). Plan: Consider log transformation.
    *   Extreme max value (1841 HP). Plan: Treat as an outlier.
    *   0 NaNs observed.
    *   Clear positive correlation with `price_log`.
*   **`torque`**:
    *   Moderately right-skewed (skewness: 1.284). Plan: Consider log transformation.
    *   0 NaNs observed (original data was numeric or coerced cleanly).
    *   Clear positive correlation with `price_log`.
*   **`acceleration`**:
    *   Slightly skewed (0.286), fairly symmetrical, multi-modal distribution. Plan: Likely usable as is or with minimal transformation.
    *   0 NaNs observed.
    *   Clear negative correlation with `price_log` (lower time = faster = higher price).
*   **`seats`**:
    *   Discrete integer values, dominated by 5-seaters. Slight positive skew (0.533).
    *   0 NaNs observed.
    *   Relationship with price is non-linear. Plan: Treat as categorical or discrete numeric for tree models.

**Categorical Features Insights (Post `astype(str).fillna('placeholder_string')` in EDA scripts - original NaNs counts from `main_pipeline.py` input are in prior notes):**
*   **`make_name`**:
    *   High cardinality. Dominated by Toyota. Plan: Group rare makes (e.g., those below a certain frequency threshold) into an "Other_Make" category.
*   **`model_name`**:
    *   Extremely high cardinality (~999 unique values). Plan: Significant grouping of rare models into "Other_Model" is essential. Consider interaction with `make_name` if it helps manage cardinality.
*   **`fuel_type`**: (5 'nan' string placeholders seen in EDA, matching original 5 NaNs)
    *   Dominated by Petrol. Plan: Consolidate "Petroleum" with "Petrol". Standardize hybrid naming (e.g., "Hybrid(Petrol)" and "Petrol Hybrid" to "Petrol Hybrid"; "Hybrid(Diesel)" and "Diesel Hybrid" to "Diesel Hybrid"). Handle generic "Hybrid" (e.g., map to most common hybrid type or "Unknown_Hybrid"). Impute the 5 NaNs.
*   **`transmission`**: (21 'nan' string placeholders seen in EDA, matching original 21 NaNs)
    *   Dominated by "At" and "Automatic". Plan: Investigate and likely merge "At" and "Automatic" (note their different median prices in EDA). Consolidate "Mt", "Manual", and specific speed manuals (e.g., 4Mt, 5Mt, 6Mt, 7Mt) into a single "Manual" category. Group rare/specialized types (Duonic, Smoother, Proshift) into "Other_Transmission". Impute the 21 NaNs.
*   **`drive_type`**: (155 'nan' string placeholders and 1 '04-FEB' error seen in EDA; original 161 NaNs)
    *   Dominated by 2WD. Plan: Consolidate related terms (e.g., `2WD` with `FF`/`FWD` if appropriate after investigation; `FR` with `RWD`). Remove/correct error "04-FEB". Group rare types (`RR`, possibly `MR` if still too few after cleaning) into "Other_Drive". Impute the 155 NaNs.
*   **`condition`**: (132 'nan' string placeholders seen in EDA)
    *   Highly mixed categories (origin, qualitative, numeric grades, status). Plan: This column needs significant restructuring.
        *   Separate "origin/history" (Foreign Used, Locally Used, New, Ready For Import) from "physical condition". This origin info likely belongs in/with `usage_type`.
        *   Standardize physical condition: Map qualitative terms (Excellent, Very Good, Average, etc.) and numeric grades (4, 4.5, 5, 6 - understand their scale) to a consistent ordinal or categorical scale.
        *   Impute the 132 NaNs for physical condition.
*   **`usage_type`**: (10 'nan' string placeholders seen in EDA; original 406 NaNs)
    *   Dominated by "Foreign Used". Plan: Consolidate "Locally Used" with "Kenyan Used"; "New" with "Brand New". Resolve ambiguous "Used" category (map or group). Impute the NaNs.
    *   Clarify relationship with `condition` column: Designate `usage_type` as the primary column for vehicle origin/history (Foreign Used, Kenyan Used, New, Ready For Import). Refine `condition` to solely reflect physical state.