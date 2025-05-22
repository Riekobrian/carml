import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# --- Configuration ---
# Correct WORKSPACE_ROOT to point to the actual project root (three levels up)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # D:\SBT-JAPAN\Alpha GO

# Configure logging to save logs in a top-level 'logs' directory within 'dataset-consolidation'
LOG_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "logs", "feature_engineering")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"step_1_4_feature_engineering_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Define file paths based on the corrected WORKSPACE_ROOT and data location
# Data is in WORKSPACE_ROOT/dataset-consolidation/data/
INPUT_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "processed") # Raw data for this script is pre-combined
OUTPUT_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "processed") # Output of this script goes here
INPUT_CSV_NAME = "cars_modeling_input.csv" # This file should be in data/processed/
OUTPUT_CSV_NAME = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_cars_feature_engineered.csv"

INPUT_FILE_PATH = os.path.join(INPUT_DATA_DIR, INPUT_CSV_NAME)
OUTPUT_FILE_PATH = os.path.join(OUTPUT_DATA_DIR, OUTPUT_CSV_NAME)

# Ensure output directory exists (INPUT_DATA_DIR and OUTPUT_DATA_DIR are the same here)
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# --- Helper Functions ---
def clean_column_names(df):
    """Standardizes column names to lowercase and replaces spaces with underscores."""
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '') for col in df.columns]
    return df

# --- Main Processing ---
def main():
    logging.info(f"Starting Step 1.4: Advanced Data Cleaning and Feature Engineering.")
    logging.info(f"Loading data from: {INPUT_FILE_PATH}")

    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        logging.info(f"Successfully loaded dataset. Shape: {df.shape}")
        logging.info(f"Original columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"CRITICAL ERROR: Input file not found at {INPUT_FILE_PATH}. Please check the path.")
        return
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Could not load data. Error: {e}")
        return

    # Standardize column names first
    df = clean_column_names(df)
    logging.info(f"Cleaned column names: {df.columns.tolist()}")

    # Make a copy for transformations to avoid SettingWithCopyWarning
    df_processed = df.copy()

    # --- NEW: Currency Standardization ---
    logging.info("Standardizing currency to KES...")
    USD_TO_KES_RATE = 130  # Agreed exchange rate

    if 'price' in df_processed.columns and 'currency_code' in df_processed.columns:
        # Ensure currency_code is standardized (lowercase) for consistent checks
        df_processed['currency_code'] = df_processed['currency_code'].astype(str).str.lower().str.strip()
        
        logging.info(f"Original currency_code value counts:\\n{df_processed['currency_code'].value_counts(dropna=False)}")

        # Identify rows with USD currency
        usd_rows = df_processed['currency_code'] == 'usd'
        
        # --- NEW: Adjust FOB USD prices to estimated total USD prices ---
        if usd_rows.any():
            logging.info(f"Adjusting {usd_rows.sum()} USD (FOB) prices to estimated total USD prices before KES conversion.")
            df_processed.loc[usd_rows, 'price'] = 1179.11 + (1.0445 * df_processed.loc[usd_rows, 'price'])
            logging.info("USD (FOB) prices adjusted to estimated total USD.")
        # --- END NEW: Adjust FOB USD prices ---

        # Convert USD prices (now estimated total USD) to KES
        if usd_rows.any(): # Check again as 'price' column was modified if any USD rows existed
            df_processed.loc[usd_rows, 'price'] = df_processed.loc[usd_rows, 'price'] * USD_TO_KES_RATE
            logging.info(f"Converted estimated total USD prices to KES for {usd_rows.sum()} rows using rate {USD_TO_KES_RATE}.")
        
        # Handle other currency codes if necessary, or log them
        other_currencies = df_processed[~df_processed['currency_code'].isin(['usd', 'ksh', 'nan'])].copy() # 'nan' for original NaNs that became strings
        if not other_currencies.empty:
            logging.warning(f"Found {len(other_currencies)} rows with currency codes other than 'usd', 'ksh', or 'nan'. "
                            f"Their prices have NOT been converted. Unique other codes: {other_currencies['currency_code'].unique()}")
            # Decide on a strategy: e.g., assume they are KES, or attempt other conversions, or flag them.
            # For now, they proceed as is, which assumes they might be KES or will be handled by NaN imputation if price is NaN.

        # Update currency_code column to reflect standardization (optional, as price is now all KES)
        # df_processed['currency_code'] = 'kes' 
        # logging.info("All prices are now standardized to KES (currency_code column updated).")
        # Or, we can just proceed knowing prices are KES and drop currency_code later.
        # For now, let's log that prices are KES and the original currency_code might be dropped later.
        logging.info("Price column is now assumed to be in KES after USD conversion.")

    else:
        logging.warning("'price' or 'currency_code' column not found. Cannot perform currency standardization.")
    # --- END NEW: Currency Standardization ---

    # --------------------------------------------------------------------------
    # Step 1.3: Target Variable Transformation (from MODEL_DEVELOPMENT_GUIDE.md)
    # --------------------------------------------------------------------------
    logging.info("Transforming target variable 'price'...")
    if 'price' in df_processed.columns:
        # Impute NaNs in price before log transformation
        if df_processed['price'].isnull().any():
            median_price = df_processed['price'].median()
            # df_processed['price'].fillna(median_price, inplace=True) # Old way
            df_processed['price'] = df_processed['price'].fillna(median_price)
            logging.info(f"NaNs in 'price' imputed with median: {median_price}")
        
        df_processed['price_log'] = np.log1p(df_processed['price'])
        logging.info("'price_log' created.")
        # Verify transformation (optional, good for debugging)
        # logging.info(f"Price skewness: {df_processed['price'].skew()}, Price_log skewness: {df_processed['price_log'].skew()}")
    else:
        logging.error("Target variable 'price' not found. Cannot create 'price_log'.")
        # Decide if to proceed or stop if price is essential

    # --------------------------------------------------------------------------
    # Step 1.4 (Cont.): Numeric Feature Engineering
    # --------------------------------------------------------------------------
    logging.info("Engineering numeric features...")

    # Year of Manufacture / Car Age
    # (Referencing MODEL_DEVELOPMENT_GUIDE.md Step 1.4 and PROJECT_NOTES.md)
    if 'year_of_manufacture' in df_processed.columns:
        logging.info("Processing 'year_of_manufacture' and 'car_age'...")
        current_year = datetime.now().year
        
        # Ensure 'year_of_manufacture' is numeric, coercing errors
        df_processed['year_of_manufacture_num'] = pd.to_numeric(df_processed['year_of_manufacture'], errors='coerce')
        
        # Handle 0s as per EDA notes (interpreting as NaNs for imputation)
        df_processed.loc[df_processed['year_of_manufacture_num'] == 0, 'year_of_manufacture_num'] = np.nan
        
        # Impute NaNs (including coerced errors and original 0s)
        # Using median of valid years (greater than a reasonable threshold, e.g., 1950)
        valid_years = df_processed.loc[df_processed['year_of_manufacture_num'] > 1950, 'year_of_manufacture_num']
        if not valid_years.empty:
            median_year_overall = valid_years.median()
            # df_processed['year_of_manufacture_num'].fillna(median_year_overall, inplace=True) # Old way
            df_processed['year_of_manufacture_num'] = df_processed['year_of_manufacture_num'].fillna(median_year_overall)
            logging.info(f"NaNs and 0s in 'year_of_manufacture_num' imputed with overall median: {median_year_overall:.0f}")
        else:
            # Fallback if no valid years, though unlikely
            # df_processed['year_of_manufacture_num'].fillna(current_year - 10, inplace=True) # e.g., 10 years old # Old way
            df_processed['year_of_manufacture_num'] = df_processed['year_of_manufacture_num'].fillna(current_year - 10)
            logging.warning("No valid years found for median imputation of 'year_of_manufacture_num'. Using a fallback.")

        df_processed['year_of_manufacture_num'] = df_processed['year_of_manufacture_num'].round().astype('Int64')

        df_processed['car_age'] = current_year - df_processed['year_of_manufacture_num']
        df_processed.loc[df_processed['car_age'] < 0, 'car_age'] = 0 # Cars from future, set age to 0
        # Impute any remaining NaNs in car_age
        if df_processed['car_age'].isnull().any():
            median_car_age = df_processed['car_age'].median()
            df_processed['car_age'] = df_processed['car_age'].fillna(median_car_age)

        df_processed['car_age_squared'] = df_processed['car_age'] ** 2
        logging.info("'car_age' and 'car_age_squared' engineered.")
    else:
        logging.warning("'year_of_manufacture' column not found. Cannot engineer 'car_age'.")

    # --- Mileage ---
    if 'mileage' in df_processed.columns:
        logging.info("Processing 'mileage'...")
        df_processed['mileage_num'] = pd.to_numeric(df_processed['mileage'], errors='coerce')
        # Handle 0s: np.log1p will handle 0s appropriately. Explicit investigation for true 0s vs placeholders is a deeper dive.
        # Impute NaNs (169 identified in EDA) with median
        median_mileage = df_processed['mileage_num'].median()
        # df_processed['mileage_num'].fillna(median_mileage, inplace=True) # Old way
        df_processed['mileage_num'] = df_processed['mileage_num'].fillna(median_mileage)
        logging.info(f"NaNs in 'mileage_num' imputed with median: {median_mileage}")

        df_processed['mileage_log'] = np.log1p(df_processed['mileage_num'])
        logging.info("'mileage_log' engineered.")

        # Mileage per Year (Requires car_age to be processed first)
        if 'car_age' in df_processed.columns:
            # Ensure car_age is not zero for division; add small epsilon or handle explicitly
            df_processed['mileage_per_year'] = df_processed['mileage_num'] / (df_processed['car_age'] + 1e-6)
            # If car_age is 0, mileage_per_year is effectively mileage_num (or could be set to 0 or mileage_num based on interpretation)
            df_processed.loc[df_processed['car_age'] == 0, 'mileage_per_year'] = df_processed['mileage_num'] 
            # Impute any NaNs that might arise (e.g., if car_age was NaN and median was also NaN, though unlikely)
            if df_processed['mileage_per_year'].isnull().any():
                median_mileage_per_year = df_processed['mileage_per_year'].median()
                df_processed['mileage_per_year'] = df_processed['mileage_per_year'].fillna(median_mileage_per_year)
            logging.info("'mileage_per_year' engineered.")
        else:
            logging.warning("'car_age' not available, cannot compute 'mileage_per_year'.")
    else:
        logging.warning("'mileage' column not found.")

    # --- Engine Size (cc) ---
    # Assuming 'engine_size_cc_numeric' from EDA is the cleaned numeric version. If original is 'engine_size_cc'.
    # The EDA output used 'engine_size_cc_numeric' from a column named 'engine_size_cc' after pd.to_numeric
    # Let's assume the input column is 'engine_size_cc' and we create 'engine_size_cc_num'
    if 'engine_size_cc' in df_processed.columns:
        logging.info("Processing 'engine_size_cc'...")
        df_processed['engine_size_cc_num'] = pd.to_numeric(df_processed['engine_size_cc'], errors='coerce')
        
        # Handle 0s (identified in EDA, treat as NaN for imputation or specific value if known)
        df_processed.loc[df_processed['engine_size_cc_num'] == 0, 'engine_size_cc_num'] = np.nan 
        # EDA showed a max of ~46008cc. Capping extreme outliers can be done here if decided.
        # For now, let's rely on log transform and robust imputation.
        # Example capping: upper_cap_engine = df_processed['engine_size_cc_num'].quantile(0.999)
        # df_processed.loc[df_processed['engine_size_cc_num'] > upper_cap_engine, 'engine_size_cc_num'] = upper_cap_engine

        median_engine_size = df_processed['engine_size_cc_num'].median()
        # df_processed['engine_size_cc_num'].fillna(median_engine_size, inplace=True) # Old way
        df_processed['engine_size_cc_num'] = df_processed['engine_size_cc_num'].fillna(median_engine_size)
        logging.info(f"NaNs (and 0s) in 'engine_size_cc_num' imputed with median: {median_engine_size}")
        
        df_processed['engine_size_cc_log'] = np.log1p(df_processed['engine_size_cc_num'])
        logging.info("'engine_size_cc_log' engineered.")
    else:
        logging.warning("'engine_size_cc' column not found.")

    # --- Horse Power ---
    # Similar to engine_size, assuming input column is 'horse_power'
    if 'horse_power' in df_processed.columns:
        logging.info("Processing 'horse_power'...")
        df_processed['horse_power_num'] = pd.to_numeric(df_processed['horse_power'], errors='coerce')
        # EDA showed a max of ~1841 HP. Capping extreme outliers.
        # upper_cap_hp = df_processed['horse_power_num'].quantile(0.999) # Example: 99.9th percentile
        # df_processed.loc[df_processed['horse_power_num'] > upper_cap_hp, 'horse_power_num'] = upper_cap_hp
        # logging.info(f"Capped 'horse_power_num' at {upper_cap_hp} (0.999 quantile).")
        
        median_hp = df_processed['horse_power_num'].median()
        # df_processed['horse_power_num'].fillna(median_hp, inplace=True) # Old way
        df_processed['horse_power_num'] = df_processed['horse_power_num'].fillna(median_hp)
        logging.info(f"NaNs in 'horse_power_num' imputed with median: {median_hp}")

        df_processed['horse_power_log'] = np.log1p(df_processed['horse_power_num'])
        logging.info("'horse_power_log' engineered.")
    else:
        logging.warning("'horse_power' column not found.")

    # --- Torque ---
    if 'torque' in df_processed.columns:
        logging.info("Processing 'torque'...")
        df_processed['torque_num'] = pd.to_numeric(df_processed['torque'], errors='coerce')
        median_torque = df_processed['torque_num'].median()
        # df_processed['torque_num'].fillna(median_torque, inplace=True) # Old way
        df_processed['torque_num'] = df_processed['torque_num'].fillna(median_torque)
        logging.info(f"NaNs in 'torque_num' imputed with median: {median_torque}")

        df_processed['torque_log'] = np.log1p(df_processed['torque_num'])
        logging.info("'torque_log' engineered.")
    else:
        logging.warning("'torque' column not found.")

    # --- Acceleration ---
    if 'acceleration' in df_processed.columns:
        logging.info("Processing 'acceleration'...")
        df_processed['acceleration_num'] = pd.to_numeric(df_processed['acceleration'], errors='coerce')
        median_accel = df_processed['acceleration_num'].median()
        # df_processed['acceleration_num'].fillna(median_accel, inplace=True) # Old way
        df_processed['acceleration_num'] = df_processed['acceleration_num'].fillna(median_accel)
        logging.info(f"NaNs in 'acceleration_num' imputed with median: {median_accel}")
        # Log transform was considered but distribution was fairly symmetrical. 
        # We can add it if model performance suggests it later: df_processed['acceleration_log'] = np.log1p(df_processed['acceleration_num'])
    else:
        logging.warning("'acceleration' column not found.")

    # --- Seats ---
    if 'seats' in df_processed.columns:
        logging.info("Processing 'seats'...")
        df_processed['seats_num'] = pd.to_numeric(df_processed['seats'], errors='coerce')
        # Seats is somewhat discrete/ordinal. Median imputation is reasonable.
        median_seats = df_processed['seats_num'].median()
        # df_processed['seats_num'].fillna(median_seats, inplace=True) # Old way
        df_processed['seats_num'] = df_processed['seats_num'].fillna(median_seats)
        # Ensure it's integer after imputation
        df_processed['seats_num'] = df_processed['seats_num'].round().astype('Int64') 
        logging.info(f"NaNs in 'seats_num' imputed with median: {median_seats} and converted to Int64.")
    else:
        logging.warning("'seats' column not found.")

    # We will add Mileage, Engine Size, Horsepower, Torque, Acceleration, Seats here # THIS COMMENT IS NOW REDUNDANT

    # --------------------------------------------------------------------------
    # Step 1.5: Categorical Feature Engineering
    # --------------------------------------------------------------------------
    logging.info("Engineering categorical features...")

    categorical_cols_to_process = [
        'make_name', 'model_name', 'fuel_type', 'transmission',
        'drive_type', 'condition', 'body_type', 'usage_type', 'source' # Added 'source' as it was in original df
        # Add any other categorical columns that need generic cleaning
    ]

    # Standardize all specified categorical columns first
    for col in categorical_cols_to_process:
        if col in df_processed.columns:
            logging.info(f"Standardizing categorical column: {col}...")
            # Convert to string, strip whitespace, convert to lowercase
            df_processed[col] = df_processed[col].astype(str).str.strip().str.lower()
            # Replace various forms of missing/generic values with a standard np.nan for consistent NaN handling later
            missing_value_placeholders = [
                'nan', 'na', 'n/a', '-', '', 'unknown', 'unspecified', 
                'other', 'others', 'none', 'null', 'undefined'
            ]
            for placeholder in missing_value_placeholders:
                df_processed.loc[df_processed[col] == placeholder, col] = np.nan
            
            # Specific handling for NaN based on user's new logic for condition and usage_type
            if col not in ['condition', 'usage_type']:
                # df_processed[col].fillna("unknown", inplace=True) # Old way
                df_processed[col] = df_processed[col].fillna("unknown")
                logging.info(f"Standardized '{col}' (lowercase, stripped) and filled NaNs with 'unknown'.")
            else:
                # For 'condition' and 'usage_type', NaNs will be handled by their specific logic sections below
                logging.info(f"Standardized '{col}' (lowercase, stripped). NaN handling deferred to specific logic.")
        else:
            logging.warning(f"Categorical column '{col}' not found for standardization.")

    # Now, specific consolidations based on EDA for each categorical column:

    # --- Fuel Type Consolidation ---
    if 'fuel_type' in df_processed.columns:
        logging.info("Consolidating 'fuel_type'...")
        fuel_type_map = {
            'petroleum': 'petrol',
            'hybrid(petrol)': 'hybrid_petrol',
            'petrol hybrid': 'hybrid_petrol',
            'hybrid': 'hybrid_petrol', # Assuming generic hybrid is petrol based on prevalence
            'hybrid(diesel)': 'hybrid_diesel',
            'diesel hybrid': 'hybrid_diesel',
            'plug-in hybrid(petrol)': 'plugin_hybrid_petrol'
            # 'electric' and 'diesel' remain as is after lowercase and stripping
        }
        df_processed['fuel_type_cleaned'] = df_processed['fuel_type'].replace(fuel_type_map)
        # For values not in map, they remain as they are (already lowercased and stripped)
        # Recount values after cleaning for logging
        logging.info(f"Value counts for 'fuel_type_cleaned':\n{df_processed['fuel_type_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'fuel_type' column not found for consolidation.")

    # --- Transmission Consolidation ---
    if 'transmission' in df_processed.columns:
        logging.info("Consolidating 'transmission'...")
        # Values from EDA: at, automatic, mt, 6mt, 5mt, manual, nan, 7mt, duonic, smoother, proshift, 4mt
        transmission_map = {
            'at': 'automatic',
            'mt': 'manual',
            '5mt': 'manual',
            '6mt': 'manual',
            '7mt': 'manual',
            '4mt': 'manual',
            'duonic': 'automated_manual', # Or other_automatic, decide based on prevalence or desired granularity
            'smoother': 'automated_manual',
            'proshift': 'automated_manual'
            # 'automatic', 'manual' remain as is after lowercase
        }
        df_processed['transmission_cleaned'] = df_processed['transmission'].replace(transmission_map)
        logging.info(f"Value counts for 'transmission_cleaned':\n{df_processed['transmission_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'transmission' column not found for consolidation.")

    # --- Drive Type Consolidation ---
    if 'drive_type' in df_processed.columns:
        logging.info("Consolidating 'drive_type'...")
        # EDA: 2wd, 4wd, awd, fr, nan, ff, mr, fwd, rwd, rr, 04-feb
        # Standard cleaning should have made '04-feb' into np.nan, then 'unknown'
        drive_type_map = {
            'ff': '2wd_front',
            'fr': '2wd_rear',
            'fwd': '2wd_front',
            'rwd': '2wd_rear',
            'mr': '2wd_mid_engine', # Often rear-wheel drive but distinct
            'rr': '2wd_rear_engine', # Often rear-wheel drive but distinct
            # '2wd', '4wd', 'awd' remain as is after lowercase
            # '04-feb' should have become 'unknown' if it was string '04-feb', or NaN if it was a parsing artifact.
            # If '04-feb' is still present and not 'unknown', it implies it bypassed initial nan conversion.
            # We will explicitly map it to unknown if it survived.
            '04-feb': 'unknown' 
        }
        df_processed['drive_type_cleaned'] = df_processed['drive_type'].replace(drive_type_map)
        # Further group less common 2WD types if desired, e.g., all specific 2WDs to just '2wd'
        # simplified_2wd_map = {
        #     '2wd_front': '2wd',
        #     '2wd_rear': '2wd',
        #     '2wd_mid_engine': '2wd',
        #     '2wd_rear_engine': '2wd'
        # }
        # df_processed['drive_type_cleaned'] = df_processed['drive_type_cleaned'].replace(simplified_2wd_map)
        logging.info(f"Value counts for 'drive_type_cleaned':\n{df_processed['drive_type_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'drive_type' column not found for consolidation.")

    # --- Condition Recoding (User Specified Logic) ---
    if 'condition' in df_processed.columns:
        logging.info("Recoding 'condition' to 'condition_clean' (Accident free / Accident involved)...")
        # Original distinct values from user: 
        # Foreign Used, Excellent, Very Good, 4.5, NaN, Locally Used, 5, Ready For Import, 6, New, 4
        # Note: Our generic cleaner above would have turned string 'nan' into np.nan.
        # The user's list `accident_free` included "NaN" as a string.
        # We will map np.nan to "Accident free" as per the user's rule (NaN is mapped into Accident free).
        
        # First, ensure the column is of string type to handle np.nan correctly in .isin() if it were string
        # However, after our generic cleaning, it contains np.nan, not string "nan"
        # df_processed['condition'] = df_processed['condition'].astype(str)

        accident_free_values_cleaned = [
            "foreign used", "excellent", "very good", "4.5", # np.nan will be handled separately
            "locally used", "5", "ready for import", "6", "new", "4", "5.0", "6.0", "4.0" # Added numeric grade strings
            # We need to be careful about float strings like '4.5' vs float 4.5 if original dtypes varied.
            # Generic cleaning converted to string and lowercased.
        ]

        # Initialize the new column
        # All values not explicitly in accident_free_values_cleaned (and not np.nan) become "Accident involved"
        df_processed['condition_clean'] = "Accident involved" 
        
        # Mark those that are accident-free
        df_processed.loc[df_processed['condition'].isin(accident_free_values_cleaned), 'condition_clean'] = "Accident free"
        
        # Explicitly map np.nan in original 'condition' to "Accident free"
        df_processed.loc[df_processed['condition'].isnull(), 'condition_clean'] = "Accident free"
        
        logging.info(f"Value counts for 'condition_clean':\n{df_processed['condition_clean'].value_counts(dropna=False)}")
    else:
        logging.warning("'condition' column not found for user-specified recoding.")

    # --- Usage Type Recoding (User Specified Logic) ---
    if 'usage_type' in df_processed.columns:
        logging.info("Recoding 'usage_type' to 'usage_type_clean' (Foreign Used / Kenyan Used)...")
        # Original distinct values from user for foreign_used: "Foreign Used", "New", "Ready For Import", "Brand New"
        # Original distinct values from user for kenyan_used: "Kenyan Used", "Locally Used", "Used"
        # NaN was to be mapped to "Kenyan Used".

        foreign_used_values_cleaned = ["foreign used", "new", "ready for import", "brand new"]
        kenyan_used_values_cleaned  = ["kenyan used", "locally used", "used"]

        # Initialize new column - default can be an intermediate or one of the categories
        df_processed['usage_type_clean'] = "Unknown_Usage" # Default, will be overwritten

        df_processed.loc[df_processed['usage_type'].isin(foreign_used_values_cleaned), 'usage_type_clean'] = "Foreign Used"
        df_processed.loc[df_processed['usage_type'].isin(kenyan_used_values_cleaned), 'usage_type_clean'] = "Kenyan Used"
        
        # Map np.nan in original 'usage_type' to "Kenyan Used"
        df_processed.loc[df_processed['usage_type'].isnull(), 'usage_type_clean'] = "Kenyan Used"

        # Check if any "Unknown_Usage" remains, implies values not in either list and not NaN
        if "Unknown_Usage" in df_processed['usage_type_clean'].unique():
            logging.warning(f"Some 'usage_type' values were not mapped to Foreign/Kenyan Used and were not NaN. They are: {df_processed[df_processed['usage_type_clean'] == 'Unknown_Usage']['usage_type'].unique()}. Defaulting them to Kenyan Used as per NaN rule or review.")
            df_processed.loc[df_processed['usage_type_clean'] == 'Unknown_Usage', 'usage_type_clean'] = "Kenyan Used"

        logging.info(f"Value counts for 'usage_type_clean':\n{df_processed['usage_type_clean'].value_counts(dropna=False)}")
    else:
        logging.warning("'usage_type' column not found for user-specified recoding.")

    # --- Body Type Consolidation ---
    if 'body_type' in df_processed.columns:
        logging.info("Consolidating 'body_type'...")
        # EDA: suv, hatchback, sedan, mini van, coupe, truck, saloon, van, station wagon, wagon, convertible, bus, pickup, 
        # mini suv, nan, minivan, double cab, pick up, pickups, pickup truck, buses and vans, coupes, estate, other, truck wing
        body_type_map = {
            'saloon': 'sedan',
            'mini van': 'van_minivan',
            'van': 'van_minivan',
            'minivan': 'van_minivan',
            'station wagon': 'wagon_estate',
            'estate': 'wagon_estate',
            'mini suv': 'suv',
            'double cab': 'pickup_truck',
            'pick up': 'pickup_truck',
            'pickups': 'pickup_truck',
            'pickup truck': 'pickup_truck',
            'truck': 'pickup_truck', # Broader category, could be separate if enough data
            'buses and vans': 'bus', # Or keep as 'van_minivan' if bus count is too low
            'coupes': 'coupe',
            'truck wing': 'special_purpose_truck' # Or 'other' if very rare
            # 'suv', 'hatchback', 'sedan', 'coupe', 'wagon', 'convertible', 'bus', 'pickup' remain
        }
        df_processed['body_type_cleaned'] = df_processed['body_type'].replace(body_type_map)
        # Group rare body types into 'other_body_type' after this specific mapping
        # threshold_body_type = 20 # Example threshold
        # counts_body_type = df_processed['body_type_cleaned'].value_counts()
        # to_replace_body_type = counts_body_type[counts_body_type < threshold_body_type].index
        # if len(to_replace_body_type) > 0:
        #     df_processed.loc[df_processed['body_type_cleaned'].isin(to_replace_body_type), 'body_type_cleaned'] = 'other_body_type'
        #     logging.info(f"Grouped rare body types into 'other_body_type'. Categories replaced: {to_replace_body_type.tolist()}")

        logging.info(f"Value counts for 'body_type_cleaned':\n{df_processed['body_type_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'body_type' column not found for consolidation.")

    # --- Make Name & Model Name (High Cardinality) ---
    # For make_name and model_name, we will group rare categories.
    # The generic cleaning already lowercased, stripped, and filled NaNs with 'unknown'.

    if 'make_name' in df_processed.columns:
        logging.info("Processing 'make_name' (high cardinality)...")
        make_counts = df_processed['make_name'].value_counts()
        # Determine a threshold for grouping. E.g., makes appearing less than N times.
        # This threshold might need tuning. Let's use 0.1% of total data as a starting point, or a fixed number like 10-20.
        make_threshold = max(10, int(len(df_processed) * 0.001))
        makes_to_group = make_counts[make_counts < make_threshold].index
        
        df_processed['make_name_cleaned'] = df_processed['make_name'].apply(lambda x: 'other_make' if x in makes_to_group else x)
        logging.info(f"Grouped {len(makes_to_group)} rare makes into 'other_make' (threshold < {make_threshold} occurrences).")
        logging.info(f"Value counts for 'make_name_cleaned':\n{df_processed['make_name_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'make_name' column not found.")

    if 'model_name' in df_processed.columns:
        logging.info("Processing 'model_name' (very high cardinality)...")
        model_counts = df_processed['model_name'].value_counts()
        # More aggressive threshold for model_name
        model_threshold = max(5, int(len(df_processed) * 0.0005)) # Even smaller percentage or fixed low number
        models_to_group = model_counts[model_counts < model_threshold].index
        
        df_processed['model_name_cleaned'] = df_processed['model_name'].apply(lambda x: 'other_model' if x in models_to_group else x)
        logging.info(f"Grouped {len(models_to_group)} rare models into 'other_model' (threshold < {model_threshold} occurrences).")
        logging.info(f"Value counts for 'model_name_cleaned':\n{df_processed['model_name_cleaned'].value_counts(dropna=False)}")
    else:
        logging.warning("'model_name' column not found.")

    # We will add Make, Model, Drive Type, Condition, Body Type, Usage Type here # THIS COMMENT IS NOW REDUNDANT

    # --------------------------------------------------------------------------
    # Create New Interaction and Boolean Features
    # --------------------------------------------------------------------------
    logging.info("Creating interaction and boolean features...")

    # --- Interaction Features (Numeric) ---
    # power_per_cc = horse_power_num / engine_size_cc_num
    if 'horse_power_num' in df_processed.columns and 'engine_size_cc_num' in df_processed.columns:
        # Ensure engine_size_cc_num is not zero to avoid division by zero
        # np.log1p would have made 0s into log(1)=0 for _log versions. _num versions might still have 0s if not imputed to NaN first.
        # Our current engine_size_cc_num processing converts 0s to NaN then imputes, so direct 0s should be rare here.
        df_processed['power_per_cc'] = df_processed['horse_power_num'] / (df_processed['engine_size_cc_num'] + 1e-6) # Add epsilon
        # df_processed['power_per_cc'].fillna(df_processed['power_per_cc'].median(), inplace=True) # Impute any NaNs from division # Old way
        if df_processed['power_per_cc'].isnull().any():
            median_power_per_cc = df_processed['power_per_cc'].median()
            df_processed['power_per_cc'] = df_processed['power_per_cc'].fillna(median_power_per_cc)
        logging.info("Engineered 'power_per_cc'.")
    else:
        logging.warning("Cannot create 'power_per_cc'. Missing 'horse_power_num' or 'engine_size_cc_num'.")

    # mileage_per_cc = mileage_num / engine_size_cc_num 
    if 'mileage_num' in df_processed.columns and 'engine_size_cc_num' in df_processed.columns:
        df_processed['mileage_per_cc'] = df_processed['mileage_num'] / (df_processed['engine_size_cc_num'] + 1e-6)
        # df_processed['mileage_per_cc'].fillna(df_processed['mileage_per_cc'].median(), inplace=True) # Old way
        if df_processed['mileage_per_cc'].isnull().any():
            median_mileage_per_cc = df_processed['mileage_per_cc'].median()
            df_processed['mileage_per_cc'] = df_processed['mileage_per_cc'].fillna(median_mileage_per_cc)
        logging.info("Engineered 'mileage_per_cc'.")
    else:
        logging.warning("Cannot create 'mileage_per_cc'. Missing 'mileage_num' or 'engine_size_cc_num'.")

    # --- Boolean/Flag Features ---
    if 'make_name_cleaned' in df_processed.columns:
        # Define luxury makes - these should match the format in 'make_name_cleaned' (lowercase, underscores for spaces/hyphens)
        luxury_makes_list = [
            'mercedes_benz', 'bmw', 'audi', 'porsche', 'lexus', 'land_rover', 'jaguar', 
            'bentley', 'ferrari', 'lamborghini', 'maserati', 'rolls_royce', 'aston_martin', 
            'mclaren', 'bugatti', 'alfa_romeo' 
            # 'amg' was a make in EDA, often associated with mercedes_benz. Add if distinct enough.
            # 'land' was also seen, likely needs mapping to 'land_rover' if not already done.
            # For simplicity, assuming make_name_cleaned has these consistently.
        ]
        df_processed['is_luxury_make'] = df_processed['make_name_cleaned'].isin(luxury_makes_list).astype(int)
        logging.info("Engineered 'is_luxury_make' flag.")
        logging.info(f"Count of luxury makes: {df_processed['is_luxury_make'].sum()}")
    else:
        logging.warning("'make_name_cleaned' not found. Cannot create 'is_luxury_make'.")

    # --- Interaction Feature (Categorical) ---
    if 'make_name_cleaned' in df_processed.columns and 'model_name_cleaned' in df_processed.columns:
        df_processed['make_model_cleaned'] = df_processed['make_name_cleaned'] + '_' + df_processed['model_name_cleaned']
        logging.info("Engineered 'make_model_cleaned' interaction feature.")
        # This new feature will also be high cardinality and might need target encoding or similar strategies later.
        # For now, we create it. It can be optionally dropped before one-hot encoding if too complex.
        # logging.info(f"Unique make_model_cleaned combinations: {df_processed['make_model_cleaned'].nunique()}")
    else:
        logging.warning("Cannot create 'make_model_cleaned'. Missing 'make_name_cleaned' or 'model_name_cleaned'.")

    # --------------------------------------------------------------------------
    # Drop Unnecessary Columns
    # --------------------------------------------------------------------------
    logging.info("Dropping unnecessary columns...")
    
    original_numeric_cols = [
        'year_of_manufacture', 'mileage', 'engine_size_cc', 
        'horse_power', 'torque', 'acceleration', 'seats'
    ]
    intermediate_numeric_cols = [
        'year_of_manufacture_num' # Add others if any were created and are no longer needed
    ]
    original_categorical_cols = [
        'make_name', 'model_name', 'fuel_type', 'transmission',
        'drive_type', # 'condition', 'usage_type', # These are now handled by condition_clean, usage_type_clean
        'body_type', 'source' 
        # Original 'condition' and 'usage_type' will be dropped in favor of 'condition_clean' and 'usage_type_clean'
    ]
    # Other columns to consider dropping based on MODEL_DEVELOPMENT_GUIDE.md Step 1.6
    # These might or might not be present in your 'cars_modeling_input.csv'
    other_cols_to_drop = [
        'price', # Original price, as we use price_log
        'condition', 'usage_type', # Add original columns here for explicit drop
        'source_specific_id', 'additional_details_from_source',
        'currency_code', 'mileage_unit', # If these were present and not used
        'unnamed:_0', 'id', # Common index or redundant ID columns from various sources
        # Add newly specified columns to drop:
        'exterior_color',
        'interior_color',
        'source_dataset',
        'location',
        'availability_status',
        'urban_consumption',
        'highway_consumption'
        # 'annual_insurance' is intentionally kept
        # Add any other columns that are irrelevant for modeling, e.g., free text fields not processed,
        # or columns that were fully captured by engineered features.
    ]

    cols_to_drop = []
    for col_list in [original_numeric_cols, intermediate_numeric_cols, original_categorical_cols, other_cols_to_drop]:
        for col_name in col_list:
            if col_name in df_processed.columns:
                cols_to_drop.append(col_name)
            # else:
            #     logging.info(f"Column '{col_name}' intended for dropping not found in DataFrame.")

    # Ensure no duplicate columns in cols_to_drop, though list append should be fine
    cols_to_drop = list(set(cols_to_drop)) # Get unique columns to drop

    # Critical check: Ensure we are not dropping the target variable if it's not 'price'
    # Our target is 'price_log'. 'price' is correctly in other_cols_to_drop.
    # If target_variable was different, it would need protection here.

    df_processed.drop(columns=cols_to_drop, inplace=True, errors='ignore') # errors='ignore' if some cols might be missing
    logging.info(f"Dropped columns: {cols_to_drop}")
    logging.info(f"Remaining columns after drops: {df_processed.columns.tolist()}")

    # --------------------------------------------------------------------------
    # Final Checks and Save
    # --------------------------------------------------------------------------
    logging.info(f"Feature engineering complete. Shape of processed data: {df_processed.shape}")
    logging.info(f"Columns in processed data: {df_processed.columns.tolist()}")

    # Example: Log null value counts for key engineered features
    key_engineered_cols = ['price_log', 'car_age', 'car_age_squared'] # Add more as they are created
    for col in key_engineered_cols:
        if col in df_processed.columns:
            logging.info(f"Nulls in '{col}': {df_processed[col].isnull().sum()}")

    try:
        df_processed.to_csv(OUTPUT_FILE_PATH, index=False)
        logging.info(f"Successfully saved processed data to: {OUTPUT_FILE_PATH}")
    except Exception as e:
        logging.error(f"Error saving processed data: {e}")

if __name__ == '__main__':
    main() 