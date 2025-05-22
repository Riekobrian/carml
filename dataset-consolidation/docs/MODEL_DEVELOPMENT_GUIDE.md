# Comprehensive Guide: Building an Accurate Used Car Price Prediction Model for the Kenyan Market

**Project Goal:** To develop an accurate and interpretable ensemble machine learning model for predicting used car prices in Kenya, as outlined in the research proposal (project.txt).

**Input Dataset (Starting Point):** `combined_cars_dataset_enriched_step4_corrected.csv` (from the latest run of `comprehensive_enrichment_pipeline.py`).

**Core Principles for Accuracy and Professionalism:**
*   **Data-Driven Decisions:** All choices (feature engineering, model selection, etc.) should be justified by data insights or established best practices.
*   **Preventing Data Leakage:** Crucial for unbiased model evaluation. Preprocessing steps must be fitted *only* on training data.
*   **Reproducibility:** Ensure that results can be replicated by using fixed `random_state` seeds, versioning code, and documenting environments.
*   **Iterative Refinement:** Model building is not strictly linear. Be prepared to revisit earlier steps based on later findings.
*   **Robust Evaluation:** Go beyond simple metrics; analyze residuals and compare against sensible baselines.
*   **Interpretability:** Understand *why* the model makes its predictions, aligning with project.txt's emphasis on SHAP.
*   **Modularity and Maintainability:** Structure code logically for clarity and ease of updates.
*   **MLOps Integration:** Incorporate practices for experiment tracking, model versioning, and automated testing.

---

## **NEW:** Setting Up a Professional MLOps Environment (Recommended)

For a professional and maintainable project, consider the following setup:

*   **Project Structure:** Adopt a clear directory layout:
    ```
    dataset-consolidation/
    ├── data/                     # Raw, processed, final datasets
    │   ├── raw/
    │   ├── processed/
    │   └── final/
    ├── models/                   # Saved model artifacts and metadata
    │   ├── base_models/
    │   └── ensemble/
    ├── notebooks/                # Jupyter notebooks for exploration, reporting
    │   └── exploratory/
    ├── src/                      # Source code (Python modules)
    │   ├── data_processing/      # Scripts for data loading, cleaning
    │   ├── feature_engineering/  # Scripts for feature creation
    │   ├── modeling/             # Scripts for model training, tuning, evaluation
    │   ├── utils/                # Utility functions
    │   └── main_pipeline.py      # Orchestration script
    ├── tests/                    # Unit and integration tests (e.g., using pytest)
    ├── docs/                     # Project documentation (like this guide, model cards)
    ├── requirements.txt          # Python package dependencies
    ├── config.yml                # Configuration file for paths, parameters
    └── README.md
    ```
*   **Dependency Management (`requirements.txt`):** Maintain a `requirements.txt` file with pinned versions of all Python packages.
    Example key libraries (add versions as per your environment):
    ```
    pandas==<version>
    numpy==<version>
    scikit-learn==<version>
    xgboost==<version>
    shap==<version>
    optuna==<version>      # For advanced hyperparameter optimization
    mlflow==<version>      # For experiment tracking and model management
    pytest==<version>      # For automated testing
    joblib==<version>
    matplotlib==<version>
    seaborn==<version>
    # Add any other specific libraries
    ```
    Generate with `pip freeze > requirements.txt`.
*   **Experiment Tracking (`mlflow`):** It is highly recommended to use `mlflow` (or a similar tool) to log parameters, code versions, metrics, and model artifacts for every significant run. This helps in comparing experiments, reproducing results, and managing the model lifecycle.
*   **Configuration Management:** Use a configuration file (e.g., `config.yml` or `.env`) for storing file paths, fixed parameters (like `random_state`), and other settings to avoid hardcoding in scripts.

---

## Phase 1: Data Preparation and Advanced Feature Engineering

This phase focuses on ensuring the data is in the optimal state for modeling and that we extract maximum predictive power through thoughtful feature creation.

**Step 1.1: Load and Verify Final Dataset**
    *   **Action:** Load the specified input CSV file using Pandas.
        ```python
        # Example
        import pandas as pd
        import numpy as np
        from datetime import datetime
        import logging # NEW: For better logging

        # NEW: Setup basic logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        # NEW: Load from a config file or define clearly
        # Example: input_file_path = config['data']['final_corrected_csv']
        input_file_path = "D:/SBT-JAPAN/Alpha GO/dataset-consolidation/data/enriched_comprehensive/YYYYMMDD_HHMMSS/combined_cars_dataset_enriched_step4_corrected.csv" # Replace with actual path
        df = pd.read_csv(input_file_path)
        logging.info(f"Loaded dataset from {input_file_path}. Shape: {df.shape}")
        ```
    *   **Verification (Sanity Checks):**
        *   Print shape: `logging.info(f"Dataset shape: {df.shape}")`
        *   Display info: `df.info()` (check Dtypes, non-null counts).
        *   Descriptive statistics: `logging.info(f"Descriptive stats:\n{df.describe(include='all')}")`
        *   Sum of nulls: `logging.info(f"Null value counts:\n{df.isnull().sum().sort_values(ascending=False)}")`
        *   Visually inspect a few rows: `logging.info(f"Sample data:\n{df.head()}")`
        *   **NEW: Implement Proactive Validation Checks:**
            ```python
            # Example Assertions
            assert 'price' in df.columns, "Critical column 'price' is missing."
            assert pd.api.types.is_numeric_dtype(df['price']), "'price' column should be numeric."
            assert df['price'].min() > 0, "Prices should be positive. Negative or zero prices found."
            if 'year_of_manufacture' in df.columns:
                assert df['year_of_manufacture'].max() <= datetime.now().year + 1, "Future manufacturing years found." # Allow for next year models
                assert df['year_of_manufacture'].min() > 1950, "Manufacturing years seem too old." # Domain specific check
            logging.info("Initial data validation checks passed.")
            ```
        *   **NEW: Establish Missing Value Thresholds:**
            *   Define acceptable missing data percentages for critical features (e.g., `price`, `make_name`, `model_name`, `year_of_manufacture`).
            *   Log a warning or halt the process if thresholds are exceeded.
            ```python
            # Example
            # critical_features_for_nan_check = ['price', 'make_name', 'model_name', 'year_of_manufacture']
            # max_nan_percentage = 5.0 # 5%
            # for feature in critical_features_for_nan_check:
            #     if feature in df.columns:
            #         nan_percent = df[feature].isnull().sum() * 100 / len(df)
            #         if nan_percent > max_nan_percentage:
            #             logging.error(f"Feature '{feature}' has {nan_percent:.2f}% missing values, exceeding threshold of {max_nan_percentage}%.")
            #             # raise ValueError(f"Too many NaNs in critical feature: {feature}") # Optionally raise error
            ```
        *   **Goal:** Confirm the dataset is as expected and robustly validated.

**Step 1.2: Focused Exploratory Data Analysis (EDA) for Modeling**
    *   **Objective:** Understand distributions and relationships critical for modeling choices.
    *   **Target Variable (`price`):**
        *   Plot histogram: `df['price'].hist(bins=50)`
        *   Plot boxplot: `df.boxplot(column=['price'])`
        *   Check skewness: `logging.info(f"Price skewness: {df['price'].skew()}")`
    *   **Key Numerical Features (e.g., `mileage`, `engine_size_cc`, `horse_power`, `year_of_manufacture`, `torque`, `acceleration`, `seats`):**
        *   Histograms and boxplots for each.
        *   Correlation matrix (heatmap) with `price` and among themselves:
            ```python
            # import matplotlib.pyplot as plt
            # import seaborn as sns
            # numerical_cols = ['price', 'mileage', 'engine_size_cc', 'horse_power', 'year_of_manufacture', ...] # select relevant
            # corr_matrix = df[numerical_cols].corr()
            # plt.figure(figsize=(12,10))
            # sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            # plt.show()
            ```
    *   **Key Categorical Features (e.g., `make_name`, `fuel_type`, `transmission`, `drive_type`, `condition`):**
        *   Bar plots of value counts: `df['feature_name'].value_counts().plot(kind='bar')`
        *   Relationship with `price` (e.g., boxplots of price per category):
            ```python
            # sns.boxplot(x='fuel_type', y='price', data=df)
            # plt.show()
            ```
    *   **Goal:** Identify severe skewness, outliers needing attention (though outlier correction was done, EDA here is for modeling context), multicollinearity, and features with little variance or strong relationships with the target.

**Step 1.3: Target Variable Transformation (Crucial for Accuracy)**
    *   **Action:** If `price` is significantly skewed (common for price data), apply a log transformation. `np.log1p` (log(1+x)) is robust as it handles zeros if any (though unlikely for price).
        ```python
        df['price_log'] = np.log1p(df['price'])
        target_variable = 'price_log' # This will be our target for modeling
        logging.info(f"Applied log1p transformation to price. New target variable: {target_variable}")
        ```
    *   **Rationale:** Stabilizes variance, makes the distribution more normal, which often improves performance of linear models, tree-based models (especially gradient boosting), and neural networks. Reduces impact of extremely high-priced outliers.
    *   **Verification:** Plot histogram of `df[target_variable]` to confirm reduced skewness.
        `df[target_variable].hist(bins=50)`

**Step 1.4: Engineer Core Predictive Features (as per `project.txt` 3.2.2.2 and best practices)**
    *   **General Note on Numeric Features (from EDA):**
        *   **Log Transformation:** For features identified as right-skewed during EDA (`mileage`, `engine_size_cc`, `horse_power`, `torque`), applying `np.log1p` is recommended to normalize their distributions. This will be part of their respective processing steps below if not handled globally.
        *   **Outlier Handling:** Extreme values noted in EDA (e.g., `engine_size_cc` max of ~46000cc, `horse_power` max of ~1841 HP, 0cc `engine_size_cc`, 0 `mileage`) need specific investigation and handling (e.g., capping, removal if error, or special treatment if valid but rare).
    *   **`Car Age`:**
        ```python
        current_year = datetime.now().year
        # Ensure year_of_manufacture is robustly converted to numeric, handling potential errors and non-integer values
        df['year_of_manufacture_num'] = pd.to_numeric(df['year_of_manufacture'], errors='coerce')
        # Impute NaNs in year_of_manufacture_num if any, before calculating age. Median is often a safe choice.
        if df['year_of_manufacture_num'].isnull().any():
            median_year = df['year_of_manufacture_num'].median()
            df['year_of_manufacture_num'].fillna(median_year, inplace=True)
            logging.warning(f"NaNs in 'year_of_manufacture_num' imputed with median: {median_year}")
        df['year_of_manufacture_num'] = df['year_of_manufacture_num'].round().astype('Int64')

        df['car_age'] = current_year - df['year_of_manufacture_num']
        # Handle potential issues:
        df.loc[df['car_age'] < 0, 'car_age'] = 0 # If year_of_manufacture > current_year, set age to 0
        df.loc[df['car_age'].isnull(), 'car_age'] = df['car_age'].median() # Fallback if year was initially NaN and couldn't compute age
        logging.info("Engineered 'car_age' feature.")
        ```
    *   **NEW: `Age Squared` (captures non-linear age effects):**
        ```python
        df['car_age_squared'] = df['car_age'] ** 2
        logging.info("Engineered 'car_age_squared' feature.")
        ```
    *   **`Mileage_per_Year`:**
        ```python
        # Ensure mileage is numeric
        df['mileage_num'] = pd.to_numeric(df['mileage'], errors='coerce')
        if df['mileage_num'].isnull().any():
            median_mileage = df['mileage_num'].median()
            df['mileage_num'].fillna(median_mileage, inplace=True)
            logging.warning(f"NaNs in 'mileage_num' imputed with median: {median_mileage}")

        # Avoid division by zero or very small age; add a small epsilon or handle explicitly
        df['mileage_per_year'] = df['mileage_num'] / (df['car_age'] + 1e-6) # Add epsilon to avoid division by zero
        df.loc[df['car_age'] == 0, 'mileage_per_year'] = df['mileage_num'] # If age is 0, mileage_per_year is just mileage
        # Impute NaNs that might arise if car_age was NaN (though handled above, this is a safeguard)
        if df['mileage_per_year'].isnull().any():
            df['mileage_per_year'].fillna(df['mileage_per_year'].median(), inplace=True)
        logging.info("Engineered 'mileage_per_year' feature.")
        ```
    *   **Interaction Features (Exploratory, based on domain knowledge and EDA):**
        *   Example: `df['engine_x_hp'] = df['engine_size_cc'] * df['horse_power']`
        *   **NEW: `Mileage_per_CC` and `Power_per_CC` (Handle NaNs and division by zero):**
            ```python
            # Ensure numeric and handle NaNs for engine_size_cc and horse_power first
            for col in ['engine_size_cc', 'horse_power']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        df[col].fillna(df[col].median(), inplace=True) # Impute NaNs

            if 'engine_size_cc' in df.columns and 'mileage_num' in df.columns:
                 df['mileage_per_cc'] = df['mileage_num'] / (df['engine_size_cc'].replace(0, 1e-6)) # Avoid division by zero
                 df.loc[df['mileage_per_cc'].isnull(), 'mileage_per_cc'] = df['mileage_per_cc'].median() # Impute resultant NaNs
                 logging.info("Engineered 'mileage_per_cc' feature.")

            if 'engine_size_cc' in df.columns and 'horse_power' in df.columns:
                df['power_per_cc'] = df['horse_power'] / (df['engine_size_cc'].replace(0, 1e-6)) # Avoid division by zero
                df.loc[df['power_per_cc'].isnull(), 'power_per_cc'] = df['power_per_cc'].median() # Impute resultant NaNs
                logging.info("Engineered 'power_per_cc' feature.")
            ```
        *   Consider interactions that make sense (e.g., premium make with low age).
    *   **Goal:** Create features that capture relationships more directly than raw data.

**Step 1.5: Review and Refine Categorical Features for Modeling**
    *   **General Note on Categorical Features (from EDA):**
        *   **Synonym Consolidation:** Crucial for features like `fuel_type` (Petrol/Petroleum, various Hybrids), `transmission` (At/Automatic, Mt/Manual/specific speeds), `body_type` (Sedan/Saloon, Wagon/Station Wagon, various Pickups). Detailed plans for consolidation should be made based on EDA notes.
        *   **Data Errors/Anomalies:** Address specific errors noted in EDA (e.g., '04-FEB' in `drive_type`).
        *   **NEW: Specific Recoding for `condition` and `usage_type` (User-Defined):**
            *   **`condition` -> `condition_clean`:** This feature will be recoded into two categories: `"Accident free"` or `"Accident involved"`.
                *   `"Accident free"` will include specific listed conditions (e.g., "Foreign Used", "Excellent", "Very Good", "4.5", "Locally Used", "5", "New", etc.) and original `NaN` values.
                *   `"Accident involved"` will be assigned to all other original `condition` values.
                *   The original `condition` column will be dropped.
            *   **`usage_type` -> `usage_type_clean`:** This feature will be recoded into two categories: `"Foreign Used"` or `"Kenyan Used"`.
                *   `"Foreign Used"` will include "Foreign Used", "New", "Ready For Import", "Brand New".
                *   `"Kenyan Used"` will include "Kenyan Used", "Locally Used", "Used", and original `NaN` values. Values not in these lists will also default to "Kenyan Used".
                *   The original `usage_type` column will be dropped.
            *   This binary recoding replaces previous plans for more granular categories for these two features.
        *   **Redundancy Resolution:** The new binary recoding for `condition` and `usage_type` aims to create distinct, simplified features. Original overlaps (e.g., "Foreign Used" appearing in both old `condition` and `usage_type`) are resolved by this focused recoding.
        *   **Standardize `condition` Column:** Replaced by the binary `condition_clean`.
        *   **NaN Handling:**
            *   For most categorical features: Impute with a standard string like "unknown" after converting common missing placeholders to `np.nan`.
            *   For `condition`: Original `NaN` values are mapped to `"Accident free"` in `condition_clean`.
            *   For `usage_type`: Original `NaN` values are mapped to `"Kenyan Used"` in `usage_type_clean`.
    *   **High Cardinality Features (e.g., `model_name`, `make_name`):**
        *   If a feature has too many unique values, one-hot encoding will create too many new columns.
        *   Strategy 1: Group rare categories.
            ```python
            # threshold = 10 # Categories with less than 10 occurrences for a feature
            # for col in ['model_name', 'make_name']: # Example columns
            #     # EDA Note: 'model_name' is extremely high cardinality (~999 unique values), requiring aggressive grouping.
            #     # 'make_name' also high, but more manageable with grouping.
            #     if col in df.columns:
            #         value_counts = df[col].value_counts()
            #         to_replace = value_counts[value_counts <= threshold].index
            #         if len(to_replace) > 0:
            #             df[col + '_grouped'] = df[col].replace(to_replace, 'Other')
            #             logging.info(f"Grouped rare categories in '{col}' into '{col}_grouped'.")
            #         else:
            #             df[col + '_grouped'] = df[col] # No grouping needed
            ```
        *   Strategy 2: Target encoding (use with extreme caution to prevent leakage; apply only within CV folds). For now, prefer grouping or rely on tree models' ability to handle high cardinality.
    *   **Consolidate `source_dataset` and `usage_type`**: (EDA Note: Focus here is on cleaning and standardizing `usage_type` itself, and its relation to `condition`)
        *   If these convey similar information (e.g., "Kenyan Used", "Foreign Used"), create a single 'origin_type' feature. Map values consistently.
    *   **Convert to `category` dtype:** For efficiency, especially for features that won't be one-hot encoded immediately (though the pipeline will handle it).
        ```python
        # for col in ['make_name_grouped', 'fuel_type', ...]: # Use grouped columns if created
        #     if col in df.columns:
        #        df[col] = df[col].astype('category')
        ```
    *   **NEW: Price Segment Feature (CAUTION - Potential Data Leakage):**
        *   Your friend suggested `df['price_segment'] = pd.qcut(df['price'], q=4, labels=['budget', 'economy', 'premium', 'luxury'])`.
        *   **WARNING:** Creating a feature directly from the original `price` (target) and then using it to predict `price_log` can lead to data leakage. If used, it *must* be created based *only* on the training set's price distribution and then applied to the test set, or used for stratified sampling/analysis rather than as a direct predictive feature for the price itself.
        *   **Recommendation for this project:** Given the goal of predicting price accurately, it's safer to **OMIT `price_segment` as a direct predictor** to avoid leakage. It can be used for EDA or post-model analysis to understand if the model performs differently across price segments.
    *   **Goal:** Make categorical features more manageable for encoding and improve model performance.

**Step 1.6: Define Final Feature Set (X) and Target (y)**
    *   **Select Features:**
        ```python
        # Example:
        features_to_drop = [
            'price', 'price_log', # Drop both original and log if target_variable is one of them
            'source_specific_id', 'additional_details_from_source',
            'currency_code', 'mileage_unit',
            'year_of_manufacture', 'year_of_manufacture_num', # If car_age is used
            'mileage', # If mileage_num and mileage_per_year are used
            # Potentially original source_dataset, usage_type if consolidated
            # Potentially original make_name, model_name if grouped versions are used
            # Add any other raw or intermediate columns not needed for modeling
        ]
        # Add target_variable to features_to_drop just to be safe it's not in X
        if target_variable not in features_to_drop:
             features_to_drop.append(target_variable)


        # Explicitly list features to keep, or columns to drop
        all_cols = df.columns.tolist()
        # X_cols are all columns NOT in features_to_drop AND NOT the target_variable itself.
        # This logic needs to be careful. It's often safer to define X_cols explicitly.
        X_cols = [col for col in all_cols if col not in features_to_drop]

        # Ensure X_cols does not contain target or original price if target is price_log
        if 'price' in X_cols and target_variable == 'price_log': X_cols.remove('price')
        if 'price_log' in X_cols and target_variable == 'price': X_cols.remove('price_log')
        if target_variable in X_cols: X_cols.remove(target_variable) # Final check

        X = df[X_cols].copy()
        y = df[target_variable].copy() # Target variable defined in Step 1.3
        logging.info(f"Defined feature set X with columns: {X.columns.tolist()}")
        logging.info(f"Defined target y: {y.name}")
        ```
    *   **NEW: Feature Selection Considerations (Initial Pass):**
        *   **Remove Low Variance Features:** For numerical features, check `X.var()`. Features with near-zero variance might not be useful. For categorical features (post-encoding), this can also be checked.
        *   **Remove Highly Correlated Features:** After EDA (Step 1.2), if strong multicollinearity was observed (e.g., VIF > 10 or correlation > 0.95 between two predictors), consider removing one of them.
        *   These can be done more formally after train-test split on `X_train` or within CV.
    *   **Verification:** `logging.info(f"X shape: {X.shape}, y shape: {y.shape}")`
    *   **Goal:** Clearly separate features and target for model input.

**Step 1.7: Train-Test Split (Fundamental for Unbiased Evaluation)**
    *   **Action:**
        ```python
        from sklearn.model_selection import train_test_split
        # Ensure random_state is consistent for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Data split into training and testing sets. X_train: {X_train.shape}, X_test: {X_test.shape}")
        ```
    *   **Rationale:** `random_state=42` ensures reproducibility. `test_size=0.2` is a standard split. The test set is held out and used *only* for final model evaluation.
    *   **Verification:** Print shapes of all resulting dataframes.
    *   **Goal:** Create independent training and testing datasets.

---

## Phase 2: Preprocessing, Pipeline Construction, and Baseline Modeling

**Goal:** Prepare the feature-engineered data for modeling by creating robust preprocessing pipelines and establishing baseline performance with various algorithms.

**Step 2.1: Preprocessing Pipeline Construction**
    *   **Action:**
        1.  Load the final feature-engineered dataset (output of Phase 1).
        2.  Split the data into training (e.g., 80%) and testing (e.g., 20%) sets. Use a fixed `random_state` for reproducibility.
        3.  Identify numeric and categorical features.
        4.  **Initial Imputation for `annual_insurance`:** Given its high NaN count and potential importance, perform an initial imputation (e.g., using `SimpleImputer(strategy='median')`) for `annual_insurance`. This will be revisited in Step 2.3.
        5.  **Numeric Feature Preprocessing:** Apply `StandardScaler` to all numeric features (including the imputed `annual_insurance`).
        6.  **Categorical Feature Preprocessing:** Apply `OneHotEncoder` (with `handle_unknown='ignore'`) to categorical features. For features with extremely high cardinality post-EDA grouping (e.g., `make_model_cleaned`), monitor the resulting feature space.
        7.  Use `sklearn.compose.ColumnTransformer` to apply these different transformations to the appropriate columns.
        8.  Fit the `ColumnTransformer` ONLY on the training data and then transform both training and testing data to prevent data leakage.
    *   **Output:** Preprocessed training and testing datasets (X_train_processed, X_test_processed, y_train, y_test).
    *   **Logging:** Record shapes of data at each step, parameters used for splitting and preprocessing.
    *   **Save:** Store the preprocessed datasets and the fitted `ColumnTransformer` object (e.g., using `joblib`).

**Step 2.2: Base Model Training and Evaluation**
    *   **Action:**
        1.  Train a diverse set of baseline regression models on the preprocessed training data. Include:
            *   Linear Regression
            *   Decision Tree Regressor
            *   Random Forest Regressor (with sensible defaults, e.g., `n_estimators=100`)
            *   Gradient Boosting Regressor (with sensible defaults)
            *   K-Nearest Neighbors Regressor
            *   Support Vector Regressor (SVR)
        2.  Evaluate each model on the preprocessed test set using:
            *   R-squared (R²)
            *   Mean Absolute Error (MAE)
            *   Root Mean Squared Error (RMSE)
            *   Training Time
        3.  Visualize predictions (Actual vs. Predicted scatter plots).
        4.  For tree-based models, visualize feature importances.
    *   **Output:** A summary table of performance metrics for all baseline models. Identification of top-performing candidates.
    *   **Script/Notebook:** `dataset-consolidation/notebooks/modeling/02_base_model_training_and_evaluation.ipynb` (or a .py script).

**Step 2.3: Advanced Imputation for Key Features (Iterative Improvement)**
    *   **Goal:** Improve data quality for `annual_insurance` (and potentially other features if identified) given its high NaN count and observed importance in baseline models.
    *   **Action:**
        1.  Modify the preprocessing pipeline from Step 2.1 to use model-based imputation for `annual_insurance` (e.g., `sklearn.impute.IterativeImputer` or `KNNImputer`).
        2.  Retrain and re-evaluate the top 2-3 baseline models (e.g., Random Forest, SVR) using this updated imputation strategy.
        3.  Compare performance (R², MAE, RMSE) against the results from Step 2.2 (with median imputation for `annual_insurance`).
    *   **Decision Point:** If model-based imputation provides a significant improvement, adopt it for subsequent steps. Otherwise, revert to the simpler imputation method to avoid unnecessary complexity.
    *   **Output:** Decision on `annual_insurance` imputation strategy. Potentially updated preprocessed datasets or a refined preprocessing pipeline.

---

## Phase 3: Advanced Modeling and Optimization

**Goal:** Systematically optimize the most promising models and explore advanced techniques to achieve the best possible predictive performance.

**Step 3.1: Hyperparameter Tuning for Top Models**
    *   **Goal:** Maximize the performance of the selected individual models.
    *   **Action:**
        1.  Select the top 2-3 models based on results from Phase 2 (likely Random Forest, SVR, Gradient Boosting, potentially KNN depending on Step 2.3 outcome).
        2.  For each selected model, define a hyperparameter search space.
        3.  Perform systematic hyperparameter tuning using cross-validation (e.g., 5-fold `KFold`):
            *   Start with `RandomizedSearchCV` for a broad search of the parameter space.
            *   Follow up with `GridSearchCV` for a finer-grained search around the best parameters found by `RandomizedSearchCV`.
        4.  Optimize for a primary metric (e.g., R-squared or minimizing RMSE), while monitoring MAE.
    *   **Tools:** `scikit-learn.model_selection.RandomizedSearchCV`, `GridSearchCV`. Consider `Optuna` for more advanced optimization if needed.
    *   **Experiment Tracking:** Log all tuning experiments (parameters, scores, best model) using `mlflow` or similar.
    *   **Output:** Tuned versions of the top models with their optimal hyperparameters and cross-validated performance scores.

**Step 3.2: Feature Selection / Dimensionality Reduction (Conditional)**
    *   **Goal:** Improve model efficiency, reduce potential overfitting, and possibly enhance performance by removing noisy or redundant features, especially if the high dimensionality (~999 features from OHE) proves problematic for tuned models.
    *   **Action (Perform if deemed necessary after Step 3.1):**
        1.  Analyze feature importances from tuned tree-based models (Random Forest, Gradient Boosting).
        2.  Analyze coefficients from tuned Linear SVR (if SVR with linear kernel is used).
        3.  Consider techniques such as:
            *   `SelectFromModel` (using feature importances/coefficients from a fitted estimator).
            *   Recursive Feature Elimination with Cross-Validation (`RFECV`).
            *   Principal Component Analysis (PCA) – use cautiously, as it reduces interpretability. Only consider if other methods are insufficient and performance is paramount over direct feature interpretation for all features.
        4.  If a feature selection/reduction method is applied, retrain the tuned models (from Step 3.1) on the reduced feature set and evaluate their performance.
    *   **Decision Point:** Compare performance with and without feature selection/reduction.
    *   **Output:** Potentially a reduced feature set. Re-evaluated performance of tuned models on the new feature set.

**Step 3.3: Advanced Ensemble Techniques (Stacking/Blending)**
    *   **Goal:** Combine the strengths of different well-performing and diverse tuned models to achieve potentially superior predictive accuracy and generalization.
    *   **Action (Perform if individual tuned models show promise but diversity):**
        1.  Select a diverse set of 2-3 best performing *tuned* models from Step 3.1 (or Step 3.2 if feature selection was applied). Diversity means models that make different kinds of errors.
        2.  Implement stacking: Use the predictions of these base models as input features for a meta-regressor (e.g., Linear Regression, a simple Random Forest). Ensure proper cross-validation to generate out-of-fold predictions for training the meta-regressor to prevent leakage.
        3.  Alternatively, implement blending (a simpler form of stacking).
    *   **Tools:** `sklearn.ensemble.StackingRegressor`.
    *   **Output:** A final ensemble model with its performance metrics on the test set.

**Step 3.4: Final Model Selection and Robust Evaluation**
    *   **Goal:** Choose the single best predictive model for the project based on a holistic assessment.
    *   **Action:**
        1.  Compare the best tuned individual model (from Step 3.1 or 3.2) and the best ensemble model (from Step 3.3) on the hold-out test set.
        2.  Evaluate based on:
            *   Primary metric (e.g., R-squared).
            *   Secondary metrics (MAE, RMSE).
            *   Training and inference time.
            *   Model complexity and maintainability.
            *   Interpretability requirements (from `project.txt`).
        3.  Perform a final detailed error analysis on the chosen model:
            *   Plot residuals (prediction error vs. predicted value, residuals vs. actual value).
            *   Look for patterns in errors (e.g., does the model perform worse for certain types of cars or price ranges?).
    *   **Output:** The "champion" model chosen for the project. A comprehensive report of its test set performance and characteristics.

---

## Phase 4: Model Interpretation and Deployment Preparation
**(This phase remains largely the same as your current guide's intent but is reiterated here for completeness based on the champion model from Phase 3)**

**Step 4.1: Model Interpretability (SHAP/LIME)**
    *   **Goal:** Understand the drivers of the final selected model's predictions, aligning with the research proposal's emphasis on SHAP.
    *   **Action:**
        1.  Apply SHAP (SHapley Additive exPlanations) to the final champion model.
        2.  Generate global interpretability plots: SHAP summary plot (feature importance), SHAP dependence plots for key features.
        3.  Analyze local interpretability: Explain individual predictions for a few representative or interesting samples.
        4.  If SHAP is computationally intensive for some models (e.g., SVR), consider LIME (Local Interpretable Model-agnostic Explanations) as an alternative or for local explanations.
    *   **Output:** Visualizations and insights into how features influence model predictions.

**Step 4.2: Final Documentation and Reporting**
    *   **Goal:** Consolidate all findings, decisions, and results into comprehensive documentation.
    *   **Action:**
        1.  Ensure `PROJECT_NOTES.md` captures key decisions, experiments, and learnings throughout the project.
        2.  Verify this `MODEL_DEVELOPMENT_GUIDE.md` accurately reflects the final process undertaken.
        3.  Prepare a final summary report or presentation detailing the methodology, results, model performance, interpretations, and conclusions, addressing the research questions from `project.txt`.
    *   **Output:** Thoroughly documented project.

**Step 4.3: Save Final Model and Preprocessing Pipeline**
    *   **Goal:** Persist the fully trained champion model and the complete preprocessing pipeline for future use, evaluation, or potential deployment.
    *   **Action:** Use `joblib` (preferred for scikit-learn objects with NumPy arrays) or `pickle` to serialize:
        1.  The final trained champion model object.
        2.  The fitted `ColumnTransformer` (or the full scikit-learn `Pipeline` if it includes the model).
    *   **Versioning:** Store these artifacts with clear versioning, possibly linked to `mlflow` runs if used.
    *   **Output:** Serialized model and pipeline files (e.g., `final_model.joblib`, `preprocessing_pipeline.joblib`).

---
**(Any subsequent phases like Deployment, Monitoring would follow here if in scope for this guide)**
## **NEW:** Key MLOps, Safety, and Professional Practices Summary

Adhering to these practices throughout the project lifecycle enhances quality, reproducibility, and maintainability:

1.  **Version Control (Git):**
    *   Commit all code, notebooks, configuration files, and documentation (`.md` files).
    *   Use meaningful commit messages.
    *   Tag important milestones (e.g., model versions like `v1.0.0`).

2.  **Experiment Tracking (e.g., `mlflow`):**
    *   Log parameters for each experiment run.
    *   Log code versions or script names.
    *   Log performance metrics (CV scores, test scores).
    *   Log model artifacts (saved models, preprocessors) and plots.
    *   Organize runs into experiments for better comparison.

3.  **Reproducibility:**
    *   Use fixed `random_state` seeds in all stochastic processes (train/test split, model initializations, CV).
    *   Maintain a detailed `requirements.txt` with pinned package versions.
    *   Document the environment setup (Python version, OS).
    *   Version control data if feasible, or at least document data sources and preprocessing steps precisely.

4.  **Automated Testing (e.g., `pytest`):**
    *   Write unit tests for critical data processing functions.
    *   Write unit tests for feature engineering logic.
    *   Implement model validation tests for saved models (as in Step 7.1.5).
    *   Consider integration tests for parts of the pipeline.

5.  **Data Safety & Validation:**
    *   Implement proactive data validation checks at loading and key transformation stages.
    *   Regularly back up critical datasets (raw, processed, final).
    *   Be mindful of data privacy if any PII was inadvertently collected (though less likely with car data).

6.  **Model Safety & Robustness:**
    *   For prediction services (future consideration): Implement input validation (data types, ranges).
    *   Check output ranges of predictions.
    *   Monitor model performance over time for drift (future consideration).

7.  **Code Quality & Maintainability:**
    *   Follow PEP 8 style guide for Python.
    *   Write modular, reusable functions and classes.
    *   Add clear docstrings and comments to your code.
    *   Use type hints for better readability and static analysis.
    *   Keep notebooks clean and focused on exploration/reporting; move complex logic to Python scripts in `src/`.

8.  **Configuration Management:**
    *   Use external configuration files (e.g., YAML, JSON, `.env`) for file paths, model parameters, thresholds, etc., instead of hardcoding.

9.  **Ethical Considerations & Bias Awareness:**
    *   Continuously reflect on potential biases in the data collection process (e.g., certain car types or regions over/underrepresented).
    *   Analyze model performance across different segments (if data allows) to check for fairness.
    *   Document these considerations in the Model Card and research report.

---
**NEW: Considerations for Future Work and MLOps Deployment (Beyond Initial Research)**
*   **Automated Retraining Pipelines:** Set up triggers for retraining when model performance degrades or new data becomes available.
*   **Continuous Integration/Continuous Deployment (CI/CD):** Automate testing and deployment of model updates.
*   **Model Monitoring:** Track live model performance, input data drift, and prediction drift.
*   **API for Model Serving:** Expose the model via a REST API (e.g., using Flask, FastAPI) for integration into applications.
*   **User Interface:** Develop a simple UI for users to input car features and get price predictions.
---

This detailed guide should equip you to build your model professionally and with a strong focus on achieving high accuracy and meaningful insights. Good luck! 