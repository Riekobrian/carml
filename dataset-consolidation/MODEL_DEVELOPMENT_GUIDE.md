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
        *   **Synonym Consolidation:** Crucial for features like `fuel_type` (Petrol/Petroleum, various Hybrids), `transmission` (At/Automatic, Mt/Manual/specific speeds), `body_type` (Sedan/Saloon, Wagon/Station Wagon, various Pickups), `usage_type` (Kenyan Used/Locally Used, New/Brand New). Detailed plans for consolidation should be made based on EDA notes.
        *   **Data Errors/Anomalies:** Address specific errors noted in EDA (e.g., '04-FEB' in `drive_type`).
        *   **Redundancy Resolution:** Resolve overlap between `condition` and `usage_type` (e.g., "Foreign Used", "Locally Used", "New"). Plan: Designate `usage_type` as primary for origin/history and refine `condition` for physical state only.
        *   **Standardize `condition` Column:** The `condition` column has mixed data types (qualitative, numeric grades, status). Plan: Standardize into a consistent categorical or ordinal scale for physical condition after separating out origin/status information.
        *   **NaN Handling:** Address NaNs for all categorical features based on specific counts and insights from EDA (e.g., impute, create "Unknown" category).
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

## Phase 2: Preprocessing and Pipeline Construction

This phase focuses on transforming raw features into a format suitable for machine learning algorithms, robustly and without data leakage.

**Step 2.1: Identify Feature Types for Preprocessing from `X_train`**
    *   **Action:**
        ```python
        # It's crucial to do this on X_train to avoid data leakage from X_test
        numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        logging.info(f"Numerical features for preprocessing: {numerical_features}")
        logging.info(f"Categorical features for preprocessing: {categorical_features}")
        ```
    *   **Goal:** Explicitly list features from the training set for separate preprocessing paths.

**Step 2.2: Construct Preprocessing Pipelines using `sklearn.pipeline` and `ColumnTransformer`**
    *   **Action:**
        ```python
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer

        # Numerical pipeline
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Or mean, if distribution is not too skewed
            ('scaler', StandardScaler())
        ])

        # Categorical pipeline
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Fill NaNs with the most common value
            # sparse_output=False for dense array, easier to work with for some SHAP versions.
            # sparse_output=True (default) is more memory efficient for high cardinality.
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create preprocessor with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, numerical_features),
                ('cat', cat_transformer, categorical_features)
            ],
            remainder='passthrough' # In case some columns were missed (e.g., boolean) or you want to keep them as is (use with caution)
        )
        logging.info("Constructed preprocessing pipelines for numerical and categorical features.")
        ```
    *   **Rationale:** `Pipeline` chains steps. `ColumnTransformer` applies different transformations to different columns. This is key for clean code and preventing data leakage during cross-validation. `handle_unknown='ignore'` in `OneHotEncoder` prevents errors if new categories appear in test/future data (they get all-zero encoding).
    *   **Goal:** Define a single preprocessor object that handles all feature transformations.

**Step 2.3: Fit Preprocessor on Training Data (Illustrative - typically done within CV/model pipeline)**
    *   **Action (for understanding, usually not done standalone like this before CV):**
        ```python
        # This FIT is ONLY on X_train.
        # X_train_processed_example = preprocessor.fit_transform(X_train)
        # X_test_processed_example = preprocessor.transform(X_test) # Use TRANSFORM only for X_test
        # logging.info(f"Example: X_train_processed shape: {X_train_processed_example.shape}")
        ```
    *   **IMPORTANT:** The `preprocessor` will be integrated into model pipelines in the next phase, ensuring it's correctly fitted on training folds during Cross-Validation.
    *   **Goal:** Understand how the preprocessor works. The actual fitting for modeling will happen within the model training loop or pipeline.

---

## Phase 3: Base Model Training and Hyperparameter Tuning (Cross-Validation)

This phase involves training individual models and finding their optimal hyperparameters.

**Step 3.1: Select Base Models (as per `project.txt` 3.2.2.3 and discussion)**
    *   Random Forest: `RandomForestRegressor(random_state=42)`
    *   Support Vector Regressor: `SVR()`
    *   K-Nearest Neighbors: `KNeighborsRegressor()`
    *   Gradient Boosting: `GradientBoostingRegressor(random_state=42)`
    *   XGBoost (highly recommended): `XGBRegressor(random_state=42, objective='reg:squarederror')` (explicit objective)
        ```python
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.svm import SVR
        from sklearn.neighbors import KNeighborsRegressor
        from xgboost import XGBRegressor # Ensure xgboost is installed
        ```

**Step 3.2: Define Hyperparameter Search Space for Each Model**
    *   **Example for `RandomForestRegressor`:**
        ```python
        rf_param_grid = {
            'regressor__n_estimators': [100, 200, 300], # Modest range for initial search
            'regressor__max_depth': [10, 20, None], # None means nodes expanded until all leaves are pure
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4],
            'regressor__max_features': ['sqrt', 'log2', None] # For RF
        }
        ```
    *   **Example for `XGBRegressor`:**
        ```python
        xgb_param_grid = {
            'regressor__n_estimators': [100, 300, 500],
            'regressor__learning_rate': [0.01, 0.05, 0.1],
            'regressor__max_depth': [3, 5, 7],
            'regressor__subsample': [0.7, 0.8, 1.0],
            'regressor__colsample_bytree': [0.7, 0.8, 1.0]
        }
        ```
    *   **Note:** The `'regressor__'` prefix is used because the model will be a step in a pipeline. Research appropriate ranges for SVR (kernel, C, gamma, epsilon) and KNN (n_neighbors, weights, p).
    *   **Goal:** Define a sensible search space for tuning.

**Step 3.3: Perform Hyperparameter Tuning using Cross-Validation**
    *   **Action (using GridSearchCV, with Optuna as an advanced alternative):**
        ```python
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        import joblib # For saving models
        # NEW: For Optuna (if used)
        # import optuna

        models_to_tune = {
            'RandomForest': (RandomForestRegressor(random_state=42), rf_param_grid),
            'XGBoost': (XGBRegressor(random_state=42, objective='reg:squarederror', early_stopping_rounds=10), xgb_param_grid), # Added early stopping for XGB
            # Add SVR, KNN, GradientBoostingRegressor with their param_grids
        }

        best_estimators = {}
        cv_results_summary = {} # To store key CV results

        # NEW: MLflow Tracking (Conceptual - requires MLflow setup)
        # import mlflow
        # mlflow.set_experiment("UsedCarPricePrediction_BaseModels") # Define experiment name

        for model_name, (model, param_grid) in models_to_tune.items():
            # with mlflow.start_run(run_name=f"Tune_{model_name}"): # NEW: Start MLflow run
            logging.info(f"\n--- Tuning {model_name} ---")
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor), # Defined in Step 2.2
                ('regressor', model)
            ])

            # Option 1: GridSearchCV (as before)
            search = GridSearchCV(pipeline, param_grid, cv=5,
                                  scoring='neg_root_mean_squared_error', # Using RMSE
                                  n_jobs=-1, verbose=1)
            
            # For XGBoost with early stopping, you need to pass eval_set to fit params
            fit_params = {}
            if model_name == 'XGBoost':
                 # This requires splitting X_train further for an eval set if using early stopping within GridSearchCV
                 # A bit complex with scikit-learn's GridSearchCV directly.
                 # Optuna or custom CV loop is better for this.
                 # For simplicity here, we'll omit direct early_stopping in GridSearchCV with XGB.
                 # Early stopping is more naturally handled by XGBoost's own CV or when using Optuna.
                 pass


            search.fit(X_train, y_train) # Fit on the raw training data

            best_estimators[model_name] = search.best_estimator_
            cv_results_summary[model_name] = {
                'best_score_rmse': -search.best_score_, # Invert because scoring was neg_rmse
                'best_params': search.best_params_
            }
            logging.info(f"Best RMSE for {model_name}: {-search.best_score_:.4f}")
            logging.info(f"Best params for {model_name}: {search.best_params_}")

            # NEW: MLflow logging
            # mlflow.log_params(search.best_params_)
            # mlflow.log_metric(f"best_cv_rmse_{model_name}", -search.best_score_)
            # mlflow.sklearn.log_model(search.best_estimator_, f"tuned_{model_name}_pipeline")

            joblib.dump(search.best_estimator_, f'./models/base_models/tuned_{model_name}_pipeline.joblib') # NEW: Save to models dir
            logging.info(f"Saved tuned pipeline for {model_name}.")
        ```
    *   **NEW: Advanced Option - Optuna for Hyperparameter Optimization:**
        *   "Optuna is a powerful hyperparameter optimization framework that uses efficient search algorithms. It can be particularly useful for large search spaces or complex models."
        *   (Conceptual Optuna integration would involve defining an `objective` function that takes `trial` object, creates a pipeline, suggests parameters using `trial.suggest_...`, performs cross-validation, and returns the score. Then `study.optimize(objective, n_trials=...)` is called.)
    *   **Rationale:** `GridSearchCV` (or `RandomizedSearchCV`/`Optuna`) systematically searches for the best hyperparameters using k-fold cross-validation on the *training set*. This prevents overfitting to a specific train/validation split.
    *   **Goal:** Find the best hyperparameter set for each base model and save these tuned models (pipelines).

---

## Phase 4: Ensemble Model Development

Combine the strengths of individual models.

**Step 4.1: Stacking Ensemble (Primary Strategy, as per `project.txt` recommendation)**
    *   **Action:**
        ```python
        from sklearn.ensemble import StackingRegressor
        from sklearn.linear_model import RidgeCV # A good meta-learner

        # Load the best base model pipelines if in a new session
        # estimators_for_stacking = []
        # for model_name in best_estimators.keys(): # Or a predefined list of top models
        #     estimator_pipeline = joblib.load(f'./models/base_models/tuned_{model_name}_pipeline.joblib')
        #     estimators_for_stacking.append((model_name.lower().replace(" ", "_"), estimator_pipeline))
        
        # If using best_estimators dict from current session:
        estimators_for_stacking = [
            (name.lower().replace(" ", "_"), est) for name, est in best_estimators.items() if name in ['RandomForest', 'XGBoost'] # Select top models
        ]
        # Ensure you have at least 2 estimators for StackingRegressor

        if len(estimators_for_stacking) < 2:
            logging.error("Stacking requires at least two base estimators. Check previous step.")
            # Handle error or proceed with a single best model
        else:
            final_meta_learner = RidgeCV(alphas=np.logspace(-6, 6, 13)) # RidgeCV does its own alpha selection
            stacked_model_pipelines = StackingRegressor(
                estimators=estimators_for_stacking,
                final_estimator=final_meta_learner,
                cv=5, # CV for training the meta-learner on out-of-fold predictions
                n_jobs=-1,
                passthrough=False # If True, original features also go to meta-learner; often False is better
            )

            logging.info("\n--- Training Stacking Ensemble (using pipelines as estimators) ---")
            # with mlflow.start_run(run_name="StackingEnsemble_Training"): # NEW: MLflow
            stacked_model_pipelines.fit(X_train, y_train)
            final_ensemble_model = stacked_model_pipelines
            logging.info("Stacking ensemble training complete.")

            # mlflow.sklearn.log_model(final_ensemble_model, "final_stacked_ensemble_model")
            joblib.dump(final_ensemble_model, './models/ensemble/final_stacked_ensemble_model.joblib') # NEW: Save to models dir
            logging.info("Saved final stacked ensemble model.")
        ```
    *   **Rationale:** Stacking uses predictions from base models as features for a meta-learner, potentially capturing more complex relationships.
    *   **Goal:** Create a powerful ensemble model.

**Step 4.2: Weighted Averaging (Alternative/Benchmark - `VotingRegressor`)**
    *   Can be implemented with `VotingRegressor` from `sklearn.ensemble`. Weights can be optimized or set based on individual model performance. Stacking is generally preferred for its learning-based combination.

---

## Phase 5: Rigorous Model Evaluation on the Test Set

Evaluate the final ensemble model on unseen data.

**Step 5.1: Make Predictions on the Test Set**
    *   **Action:**
        ```python
        # Load the final model if in a new session
        # final_ensemble_model = joblib.load('./models/ensemble/final_stacked_ensemble_model.joblib')

        y_pred_log_test = final_ensemble_model.predict(X_test) # X_test is raw

        # Inverse transform predictions if target was log-transformed
        if target_variable == 'price_log':
            y_pred_test_orig = np.expm1(y_pred_log_test)
            y_test_orig = np.expm1(y_test) # Ensure y_test is also in original scale for comparison
        else:
            y_pred_test_orig = y_pred_log_test
            y_test_orig = y_test
        logging.info("Predictions made on the test set and inverse transformed to original scale.")
        ```
    *   **Goal:** Get predictions for the held-out test set.

**Step 5.2: Calculate Performance Metrics (as per `project.txt` 3.2.4)**
    *   **Action:**
        ```python
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        def mean_absolute_percentage_error(y_true, y_pred): # Friend's guide MAPE
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            # Avoid division by zero if y_true contains zeros
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


        mae = mean_absolute_error(y_test_orig, y_pred_test_orig)
        rmse = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))
        r2 = r2_score(y_test_orig, y_pred_test_orig)
        mape = mean_absolute_percentage_error(y_test_orig, y_pred_test_orig)

        logging.info(f"\n--- Final Model Evaluation on Test Set ---")
        logging.info(f"MAE: {mae:.2f}")
        logging.info(f"RMSE: {rmse:.2f}")
        logging.info(f"R-squared: {r2:.4f}")
        logging.info(f"MAPE: {mape:.2f}%") # Be cautious if y_test_orig contains zeros

        # NEW: MLflow logging for final model
        # with mlflow.active_run() if mlflow.active_run() else mlflow.start_run(run_id=PARENT_RUN_ID_OF_STACKING): # Or associate with stacking run
        #    mlflow.log_metric("test_mae", mae)
        #    mlflow.log_metric("test_rmse", rmse)
        #    mlflow.log_metric("test_r2", r2)
        #    mlflow.log_metric("test_mape", mape)
        ```
    *   **NEW: Assess Metric Stability with Confidence Intervals:**
        ```python
        # from sklearn.utils import resample
        # n_iterations = 1000 # Number of bootstrap samples
        # bootstrap_r2_scores = []
        # for i in range(n_iterations):
        #     # Sample with replacement from the test set predictions and true values
        #     indices = resample(np.arange(len(y_test_orig)), replace=True)
        #     if len(indices) > 0: # Ensure non-empty sample
        #        bootstrap_r2_scores.append(r2_score(y_test_orig.iloc[indices], y_pred_test_orig[indices]))
        #
        # if bootstrap_r2_scores: # Check if list is not empty
        #    r2_ci_lower = np.percentile(bootstrap_r2_scores, 2.5)
        #    r2_ci_upper = np.percentile(bootstrap_r2_scores, 97.5)
        #    logging.info(f"R-squared 95% CI (Bootstrap): [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
        #    # mlflow.log_metric("test_r2_ci_lower", r2_ci_lower)
        #    # mlflow.log_metric("test_r2_ci_upper", r2_ci_upper)
        ```
    *   **Goal:** Quantify model performance on unseen data and assess its stability.

**Step 5.3: Residual Analysis (Insight into Model Behavior)**
    *   **Action:**
        ```python
        import matplotlib.pyplot as plt
        import seaborn as sns

        residuals = y_test_orig - y_pred_test_orig

        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_pred_test_orig, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted Values (Original Scale)")
        plt.ylabel("Residuals (Original Scale)")
        plt.title("Residuals vs. Predicted Values")
        # plt.savefig('./docs/residual_plot.png') # NEW: Save plot
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel("Residuals (Original Scale)")
        plt.title("Histogram of Residuals")
        # plt.savefig('./docs/residuals_histogram.png') # NEW: Save plot
        plt.show()
        # NEW: MLflow logging for plots
        # mlflow.log_figure(plt.gcf(), "residual_plot.png")
        ```
    *   **Rationale:** Checks for homoscedasticity (constant variance of residuals), bias (residuals centered around zero), and normality of residuals. Patterns indicate areas where the model can be improved.
    *   **Goal:** Understand error patterns.

**Step 5.4: Compare with a Baseline Model**
    *   **Action:** E.g., predict the median price from `y_train` (original scale) for all test instances.
        ```python
        # median_price_train_orig = np.expm1(y_train.median()) if target_variable == 'price_log' else y_train.median()
        # baseline_predictions_orig = np.full_like(y_test_orig, median_price_train_orig)
        #
        # mae_baseline = mean_absolute_error(y_test_orig, baseline_predictions_orig)
        # rmse_baseline = np.sqrt(mean_squared_error(y_test_orig, baseline_predictions_orig))
        # r2_baseline = r2_score(y_test_orig, baseline_predictions_orig)
        # logging.info(f"Baseline MAE (predicting median from train): {mae_baseline:.2f}")
        ```
    *   **Goal:** Demonstrate the value added by the machine learning model.

---

## Phase 6: Model Interpretability using SHAP (as per `project.txt` 1.5, 3.2.2.3)

Understand *why* the model makes its predictions.

**Step 6.1: Prepare Data and Model for SHAP**
    *   **Challenge with Stacking:** SHAP for `StackingRegressor` can be complex. It's often more practical to apply SHAP to the strongest base models or the meta-learner if it's simple (like RidgeCV).
    *   **Focus on a Strong Base Model (e.g., XGBoost or RandomForest pipeline from `best_estimators`):**
        ```python
        import shap # Ensure shap is installed
        shap.initjs() # For JS plots in notebooks

        # Example: Use the tuned XGBoost pipeline for SHAP
        # Ensure 'XGBoost' is one of the keys in best_estimators
        if 'XGBoost' in best_estimators:
            shap_model_pipeline = best_estimators['XGBoost']
        elif 'RandomForest' in best_estimators: # Fallback to RF if XGB not tuned
            shap_model_pipeline = best_estimators['RandomForest']
        else: # Or your overall best single model
            logging.warning("No preferred model (XGBoost/RandomForest) found in best_estimators for SHAP. Skipping SHAP.")
            shap_model_pipeline = None

        if shap_model_pipeline:
            shap_model_preprocessor = shap_model_pipeline.named_steps['preprocessor']
            shap_model_regressor = shap_model_pipeline.named_steps['regressor']

            # Preprocess X_test (or a sample of X_train for background, X_test for explanation)
            # It's common to use a sample of X_train as background for TreeExplainer
            # And X_test (or a sample) for the actual explanations.
            X_test_transformed_for_shap = shap_model_preprocessor.transform(X_test) # Transform the raw X_test

            # Get feature names AFTER one-hot encoding from the preprocessor
            try:
                # For ColumnTransformer, get_feature_names_out is preferred
                feature_names_shap = shap_model_preprocessor.get_feature_names_out()
            except AttributeError:
                # Fallback for older scikit-learn or simpler transformers
                # This part can be tricky and depends on the preprocessor structure
                feature_names_shap = X_test.columns # Placeholder, needs to be accurate
                logging.warning("Could not get feature names from preprocessor automatically for SHAP. Using X_test.columns as placeholder.")


            X_test_transformed_df_for_shap = pd.DataFrame(X_test_transformed_for_shap, columns=feature_names_shap)
            logging.info("Prepared data for SHAP analysis.")
        ```
    *   **Recommendation:** Start SHAP with your best single *tree-based* model pipeline. For Stacking, interpreting the `final_estimator` (if it's linear like Ridge) over the base model predictions can be insightful.

**Step 6.2: Initialize SHAP Explainer and Calculate Values**
    *   **Action (for a Tree Model):**
        ```python
        if shap_model_pipeline and hasattr(shap_model_regressor, 'predict'): # Check if it's a tree-like model SHAP supports well
            if isinstance(shap_model_regressor, (RandomForestRegressor, GradientBoostingRegressor, XGBRegressor)):
                explainer = shap.TreeExplainer(shap_model_regressor, X_test_transformed_df_for_shap) # Background data
                # For newer shap versions, explainer object itself might be callable
                # shap_values_test_obj = explainer(X_test_transformed_df_for_shap)
                # shap_values_test = shap_values_test_obj.values
                # For older versions:
                shap_values_test = explainer.shap_values(X_test_transformed_df_for_shap)

                logging.info("Calculated SHAP values for the selected model.")
            else:
                logging.warning(f"SHAP TreeExplainer may not be optimal for {type(shap_model_regressor)}. Consider KernelExplainer.")
                # explainer = shap.KernelExplainer(shap_model_regressor.predict, shap.sample(X_test_transformed_df_for_shap, 50)) # Sample for Kernel
                # shap_values_test = explainer.shap_values(X_test_transformed_df_for_shap) # Can be slow
                shap_values_test = None # Placeholder if KernelExplainer not run
        else:
            shap_values_test = None
            logging.info("Skipping SHAP value calculation as no suitable model was selected.")

        ```
    *   **Goal:** Calculate feature contributions for each prediction.

**Step 6.3: Generate and Analyze SHAP Plots**
    *   **Action:**
        ```python
        if shap_values_test is not None and X_test_transformed_df_for_shap is not None:
            plt.figure()
            shap.summary_plot(shap_values_test, X_test_transformed_df_for_shap, plot_type="bar", show=False)
            plt.title("SHAP Global Feature Importance (Bar)")
            # plt.savefig('./docs/shap_summary_bar.png') # NEW: Save plot
            plt.show()

            plt.figure()
            shap.summary_plot(shap_values_test, X_test_transformed_df_for_shap, show=False) # Beeswarm plot
            plt.title("SHAP Global Feature Importance (Beeswarm)")
            # plt.savefig('./docs/shap_summary_beeswarm.png') # NEW: Save plot
            plt.show()

            # Dependence plot for a key feature (e.g., first transformed numerical feature)
            # Ensure the feature name exists in X_test_transformed_df_for_shap.columns
            if X_test_transformed_df_for_shap.shape[1] > 0:
                 key_feature_for_dependence = X_test_transformed_df_for_shap.columns[0]
                 plt.figure()
                 shap.dependence_plot(key_feature_for_dependence, shap_values_test, X_test_transformed_df_for_shap, interaction_index=None, show=False)
                 plt.title(f"SHAP Dependence Plot for {key_feature_for_dependence}")
                 # plt.savefig(f'./docs/shap_dependence_{key_feature_for_dependence}.png') # NEW: Save plot
                 plt.show()
            # NEW: MLflow logging for SHAP plots
            # mlflow.log_figure(plt.gcf(), "shap_summary_bar.png")
        ```
    *   **Goal:** Visualize feature importances and their effects.

**Step 6.4: Document SHAP Insights**
    *   Identify top N most important features from the summary plot.
    *   Explain the direction of impact (e.g., "higher `car_age` negatively impacts `price_log` based on beeswarm plot").
    *   Note any interesting interactions or non-linearities revealed by dependence plots.
    *   Compare with domain knowledge – do the importances make sense?
    *   These insights are crucial for your research objectives (1.3.2, 1.5 in `project.txt`).

---

## Phase 7: Finalization, Documentation, and Reporting

Concluding the modeling process.

**Step 7.1: Save All Key Artifacts with Metadata**
    *   Final Ensemble Model: (e.g., `final_stacked_ensemble_model.joblib` in `./models/ensemble/`)
    *   Best Base Models: (e.g., `tuned_XGBoost_pipeline.joblib` in `./models/base_models/`)
    *   Preprocessor: (if saved separately, though it's part of pipelines here: `preprocessor.joblib`)
    *   List of features used for modeling (`X_cols`).
    *   Key evaluation metrics and plots (saved to `./docs/` or logged via `mlflow`).
    *   SHAP plots and insights summary.
    *   **NEW: Model Metadata File:** For each saved model (especially the final ensemble), create a corresponding `.json` metadata file.
        ```python
        # import json
        # import sys
        # # Example metadata for the final ensemble model
        # model_metadata = {
        #     'model_name': 'final_stacked_ensemble_model',
        #     'model_version': '1.0.0', # Increment as you retrain
        #     'training_timestamp': datetime.now().isoformat(),
        #     'target_variable': target_variable,
        #     'input_data_source': input_file_path, # Original raw data path
        #     'model_type': str(type(final_ensemble_model)),
        #     'performance_metrics_test_set': {
        #         'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape
        #     },
        #     # 'r2_ci_bootstrap': [r2_ci_lower, r2_ci_upper] if bootstrap_r2_scores else None,
        #     'python_version': sys.version,
        #     # 'key_dependencies': { # From requirements.txt
        #     #    'scikit-learn': sklearn.__version__,
        #     #    'xgboost': xgboost.__version__, ...
        #     # },
        #     'features_used': X_cols,
        #     'notes': 'Initial version of the stacked ensemble model.'
        # }
        # with open('./models/ensemble/final_stacked_ensemble_model_metadata.json', 'w') as f:
        #    json.dump(model_metadata, f, indent=4)
        # logging.info("Saved model metadata for final_stacked_ensemble_model.")
        ```

**NEW: Step 7.1.5: Implement Model Validation Tests for Saved Model**
    *   Create a small set of diverse test cases (e.g., 5-10 sample input data rows with known/expected output ranges). These can be hand-crafted or sampled from your test set.
    *   Write a script (e.g., in `tests/test_model_predictions.py` using `pytest`) that:
        *   Loads the saved model (e.g., `final_stacked_ensemble_model.joblib`).
        *   Loads the preprocessor if it was saved separately (or relies on the model pipeline).
        *   Takes a test case input (a Pandas DataFrame row or dict).
        *   Preprocesses the input.
        *   Makes a prediction.
        *   Asserts that the prediction falls within an acceptable range or is close to an expected value (define a tolerance).
    *   **Rationale:** Ensures the model loads correctly and produces sensible outputs for known inputs, guarding against issues in serialization/deserialization or environment changes.

**Step 7.2: Code Structuring and Versioning**
    *   Organize code into logical Python scripts/modules within the `src/` directory (e.g., `src/data_processing/load.py`, `src/feature_engineering/transformers.py`, `src/modeling/train.py`, `src/modeling/predict.py`).
    *   Use Git for version control of all code, notebooks, configuration files, and documentation. Commit frequently with meaningful messages. Tag releases/versions of your model.
    *   Maintain `requirements.txt` for the Python environment: `pip freeze > requirements.txt`.

**Step 7.3: Comprehensive Documentation**
    *   Write a detailed report or Jupyter notebook (`notebooks/final_model_report.ipynb`) explaining each step:
        *   Data loading, validation, and initial checks.
        *   EDA findings and visualizations.
        *   Feature engineering choices, rationale, and implementation details.
        *   Preprocessing steps.
        *   Model selection rationale, hyperparameter tuning process (CV scores, best params for each base model).
        *   Ensemble strategy and construction.
        *   Final model evaluation results (all metrics, residual plots, comparison to baseline).
        *   SHAP interpretability findings (key features, impact direction, plots).
    *   Clearly state assumptions made and limitations of the model.
    *   **NEW: Create a Model Card.** Use a structured format (like your friend's template or Google's Model Card Toolkit) to document:
        *   Model Details (name, version, type).
        *   Intended Use (primary uses, out-of-scope uses for the Kenyan market).
        *   Factors (relevant input features, evaluation factors).
        *   Metrics (performance on test set).
        *   Evaluation Data (dataset source, preprocessing).
        *   Quantitative Analyses.
        *   Ethical Considerations & Potential Biases (e.g., data source bias, fairness across car types/locations if analyzed).
        *   Caveats and Recommendations.
        *   Save as `./docs/model_card_v1.md`.

**Step 7.4: Prepare Research Deliverables (Thesis Chapter / Paper)**
    *   Structure the findings from your comprehensive documentation to directly answer your research questions from `project.txt`.
    *   Discuss how the developed ensemble model addresses the problem statement of inconsistent pricing.
    *   Highlight the significance of the results for the Kenyan used car market, including accuracy achieved and key price-driving factors identified by SHAP.
    *   Propose areas for future work (e.g., more diverse data sources, different model architectures, deployment strategies, ongoing monitoring).

---

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