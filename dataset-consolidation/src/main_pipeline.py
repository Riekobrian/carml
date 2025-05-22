import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the input file path (ensure this path is correct for your Build environment)
input_file_path = r"D:\SBT-JAPAN\Alpha GO\dataset-consolidation\data\processed\cars_modeling_input.csv" # Using double backslashes for Windows paths in Python strings

try:
    df = pd.read_csv(input_file_path)
    logging.info(f"Successfully loaded dataset from {input_file_path}. Shape: {df.shape}")

    # --- Initial Verification (as per MODEL_DEVELOPMENT_GUIDE.md Step 1.1) ---
    print("\n--- df.info() ---")
    df.info()
    print("\n--- df.describe(include='all') ---")
    print(df.describe(include='all'))

    print("\n--- df.isnull().sum() ---")
    print(df.isnull().sum().sort_values(ascending=False))

    print("\n--- df.head() ---")
    print(df.head())

    # Proactive Validation Checks (as per MODEL_DEVELOPMENT_GUIDE.md Step 1.1)
    logging.info("Performing proactive validation checks...")
    all_validations_passed = True # Flag to track overall validation status

    # Price column checks
    if 'price' in df.columns:
        logging.info("'price' column exists.")
        if not pd.api.types.is_numeric_dtype(df['price']):
            logging.error("DATA QUALITY ISSUE: 'price' column is not numeric.")
            all_validations_passed = False
        else:
            logging.info("'price' column is numeric.")
            if df['price'].notna().any():
                if not (df['price'][df['price'].notna()].min() > 0):
                    logging.error(f"DATA QUALITY ISSUE: Prices should be positive. Min non-NaN price found: {df['price'][df['price'].notna()].min()}.")
                    all_validations_passed = False
                else:
                    logging.info("Minimum price (for non-NaN values) is positive.")
            else:
                logging.warning("'price' column contains all NaN values. Min value check skipped.")
    else:
        logging.error("CRITICAL ERROR: Critical column 'price' is missing.")
        all_validations_passed = False

    # Year of manufacture column checks
    if 'year_of_manufacture' in df.columns:
        logging.info("'year_of_manufacture' column exists.")
        if not pd.api.types.is_numeric_dtype(df['year_of_manufacture']):
            logging.error("DATA QUALITY ISSUE: 'year_of_manufacture' column is not numeric.")
            all_validations_passed = False
        else:
            min_year_overall = df['year_of_manufacture'].min()
            if min_year_overall <= 1950:
                logging.error(f"DATA QUALITY ISSUE: Minimum 'year_of_manufacture' is {min_year_overall}, which is <= 1950. This indicates potential data errors that will need handling in subsequent steps.")
                all_validations_passed = False 
                # Log specific count for years between 1 and 1950 if 0 is the min
                if min_year_overall == 0:
                    very_old_years = df[(df['year_of_manufacture'] > 0) & (df['year_of_manufacture'] <= 1950)]['year_of_manufacture']
                    if not very_old_years.empty:
                        logging.warning(f"Found {len(very_old_years)} entries with 'year_of_manufacture' between 1 and 1950 (inclusive). Min such year: {very_old_years.min()}.")
            else: # All years are > 1950
                logging.info(f"Minimum 'year_of_manufacture' ({min_year_overall}) is > 1950 and looks reasonable.")

            # Max year check
            try:
                assert df['year_of_manufacture'].max() <= datetime.now().year + 1, f"Future manufacturing years found. Max year: {df['year_of_manufacture'].max()}"
                logging.info(f"Maximum 'year_of_manufacture' ({df['year_of_manufacture'].max()}) is valid.")
            except AssertionError as e_max_year:
                logging.error(f"DATA QUALITY ISSUE: {e_max_year}")
                all_validations_passed = False
    else:
        logging.warning("'year_of_manufacture' column not found. Year validation checks skipped.")
        # Not finding this column might be critical depending on planned features
        # all_validations_passed = False # Decide if this should fail validation

    if all_validations_passed:
        logging.info("Initial data validation checks completed. Some issues might have been logged as warnings/errors above - please review.")
    else:
        logging.error("CRITICAL DATA QUALITY ISSUES FOUND during initial validation checks. Please review logs carefully before proceeding.")

    # Establish Missing Value Thresholds (as per MODEL_DEVELOPMENT_GUIDE.md Step 1.1)
    logging.info("Checking missing value thresholds for critical features...")
    critical_features_for_nan_check = ['price', 'make_name', 'model_name', 'year_of_manufacture']
    max_nan_percentage = 5.0 # 5%
    for feature in critical_features_for_nan_check:
        if feature in df.columns:
            nan_percent = df[feature].isnull().sum() * 100 / len(df)
            logging.info(f"Feature '{feature}' has {nan_percent:.2f}% missing values.")
            if nan_percent > max_nan_percentage:
                logging.warning(f"WARNING: Feature '{feature}' has {nan_percent:.2f}% missing values, exceeding threshold of {max_nan_percentage}%. Consider investigation.")
        else:
            logging.warning(f"Critical feature '{feature}' for NaN check not found in DataFrame.")
    logging.info("Missing value threshold checks completed.")

    # --- Custom Data Cleaning (Added before formal EDA) ---
    logging.info("--- Starting Custom Data Cleaning --- ")

    # 1. Handle invalid 'year_of_manufacture' (e.g., values of 0)
    if 'year_of_manufacture' in df.columns and pd.api.types.is_numeric_dtype(df['year_of_manufacture']):
        invalid_year_count = df[df['year_of_manufacture'] == 0].shape[0]
        if invalid_year_count > 0:
            logging.warning(f"Found {invalid_year_count} entries with 'year_of_manufacture' == 0. Converting to NaN.")
            df['year_of_manufacture'] = df['year_of_manufacture'].replace(0, np.nan)
            logging.info(f"NaNs in 'year_of_manufacture' after converting 0s: {df['year_of_manufacture'].isnull().sum()}")
        else:
            logging.info("No 'year_of_manufacture' entries found with value 0.")

    # 2. Inspect 'currency_code'
    if 'currency_code' in df.columns:
        logging.info("\n--- Inspecting 'currency_code' --- ")
        print("Value counts for 'currency_code':")
        print(df['currency_code'].value_counts(dropna=False))
        print(f"Number of NaNs in 'currency_code': {df['currency_code'].isnull().sum()}")
    else:
        logging.warning("'currency_code' column not found for inspection.")

    # 3. Inspect 'mileage_unit'
    if 'mileage_unit' in df.columns:
        logging.info("\n--- Inspecting 'mileage_unit' --- ")
        print("Value counts for 'mileage_unit':")
        print(df['mileage_unit'].value_counts(dropna=False))
        print(f"Number of NaNs in 'mileage_unit': {df['mileage_unit'].isnull().sum()}")
    else:
        logging.warning("'mileage_unit' column not found for inspection.")

    # --- Apply Cleaning Based on Inspection ---
    logging.info("--- Applying Imputation Cleaning --- ")
    # Clean 'currency_code' - Impute NaNs with mode (USD)
    if 'currency_code' in df.columns:
        mode_currency = df['currency_code'].mode()[0]
        df['currency_code'] = df['currency_code'].fillna(mode_currency) # Avoid FutureWarning
        logging.info(f"Imputed NaNs in 'currency_code' with mode: '{mode_currency}'. NaN count now: {df['currency_code'].isnull().sum()}")
        if df['currency_code'].nunique() > 1:
            logging.warning(f"'currency_code' still contains multiple values: {df['currency_code'].unique().tolist()}. Price conversion might be needed later.")

    # Clean 'mileage_unit' - Impute NaNs with mode (km)
    if 'mileage_unit' in df.columns:
        mode_mileage_unit = df['mileage_unit'].mode()[0]
        df['mileage_unit'] = df['mileage_unit'].fillna(mode_mileage_unit) # Avoid FutureWarning
        logging.info(f"Imputed NaNs in 'mileage_unit' with mode: '{mode_mileage_unit}'. NaN count now: {df['mileage_unit'].isnull().sum()}")

    # Drop rows with 'Unknown_Placeholder' in both make_name and model_name
    if 'make_name' in df.columns and 'model_name' in df.columns:
        initial_rows = df.shape[0]
        condition_to_drop = (df['make_name'] == 'Unknown_Placeholder') & (df['model_name'] == 'Unknown_Placeholder')
        rows_to_drop_count = df[condition_to_drop].shape[0]
        if rows_to_drop_count > 0:
            df = df[~condition_to_drop].copy() # Use .copy() to ensure it's a new DataFrame
            logging.warning(f"Dropped {rows_to_drop_count} rows where both 'make_name' and 'model_name' were 'Unknown_Placeholder'. New shape: {df.shape}")
        else:
            logging.info("No rows found with 'Unknown_Placeholder' in both 'make_name' and 'model_name'.")
    else:
        logging.warning("'make_name' or 'model_name' not found, skipping drop for 'Unknown_Placeholder'.")

    logging.info("--- Custom Data Cleaning Initial Inspection Done --- ")
    logging.info("Next: Review output for currency_code and mileage_unit to decide cleaning strategy.")

except FileNotFoundError:
    logging.error(f"CRITICAL ERROR: The file was not found at {input_file_path}")
    logging.error("Please ensure the file 'cars_modeling_input.csv' exists in the 'Build/data/final/' directory.")
except Exception as e:
    logging.error(f"An error occurred during data loading: {e}")

# Further code for Step 1.1 and subsequent steps will be added below.
