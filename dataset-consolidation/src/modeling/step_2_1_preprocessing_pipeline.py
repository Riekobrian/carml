import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import logging
import os
from datetime import datetime
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')) # Adjust if script moves
FEATURE_ENGINEERED_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "processed")
PREPROCESSED_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "final") # Saving final preprocessed data here
# INPUT_CSV_NAME = "20250520_123812_cars_feature_engineered.csv" # This will need to be updated if re-run or use a dynamic way to get the latest
# For now, let's make it more robust by finding the latest feature engineered file if possible,
# or fall back to a specific name if provided/needed.

# Ensure output directory exists
os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)

def find_latest_feature_engineered_file(directory):
    """Finds the most recent '_cars_feature_engineered.csv' file in the given directory."""
    try:
        files = [f for f in os.listdir(directory) if f.endswith("_cars_feature_engineered.csv") and os.path.isfile(os.path.join(directory, f))]
        if not files:
            return None
        # Sort files by creation time (or modification time as a proxy if creation time is not easily accessible on all OS)
        # For simplicity, sorting by name assuming YYYYMMDD_HHMMSS prefix ensures latest.
        files.sort(reverse=True)
        return files[0]
    except Exception as e:
        logging.warning(f"Could not automatically find the latest feature engineered file: {e}")
        return None

# --- Main Preprocessing Function ---
def main():
    logging.info("Starting Step 2.1: Preprocessing and Pipeline Construction.")

    # Dynamically find the latest feature engineered file
    latest_fe_file = find_latest_feature_engineered_file(FEATURE_ENGINEERED_DATA_DIR)
    if not latest_fe_file:
        logging.error(f"CRITICAL: No feature engineered CSV file found in {FEATURE_ENGINEERED_DATA_DIR}. Exiting.")
        # As a fallback, you could uncomment and set INPUT_CSV_NAME manually if needed:
        # INPUT_CSV_NAME = "your_specific_file_name.csv" 
        # if not INPUT_CSV_NAME: return
        return
    else:
        INPUT_CSV_NAME = latest_fe_file

    INPUT_FILE_PATH = os.path.join(FEATURE_ENGINEERED_DATA_DIR, INPUT_CSV_NAME)
    logging.info(f"Loading feature-engineered data from: {INPUT_FILE_PATH}")

    try:
        df = pd.read_csv(INPUT_FILE_PATH)
        logging.info(f"Successfully loaded dataset. Shape: {df.shape}")
        logging.info(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        logging.error(f"CRITICAL ERROR: Input file not found at {INPUT_FILE_PATH}. Please check the path.")
        return
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Could not load data. Error: {e}")
        return

    # Define target variable and features
    TARGET_COLUMN = 'price_log'
    if TARGET_COLUMN not in df.columns:
        logging.error(f"CRITICAL ERROR: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return
        
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    # Split data into training and testing sets
    logging.info("Splitting data into training and testing sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    logging.info(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

    # Identify feature types for preprocessing
    # Note: 'annual_insurance' is numeric but needs NaN imputation first.
    # All other numeric features created in step_1_4 should be clean of NaNs.
    
    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train.select_dtypes(include='object').columns.tolist()

    # Adjust numeric_features if 'annual_insurance' is misclassified or to ensure it's handled
    # It should be np.number, but we'll handle its NaNs specifically.
    # For now, assume 'annual_insurance' is in numeric_features if it's numeric type.
    # If it's object due to many NaNs and then read as string, it would be in categorical.
    # Let's be explicit with 'annual_insurance' handling.

    logging.info(f"Identified numeric features: {numeric_features}")
    logging.info(f"Identified categorical features: {categorical_features}")

    # --- Preprocessing Steps ---

    # Create preprocessing pipelines for numeric and categorical features

    # Numeric features pipeline:
    # 1. Impute NaNs in 'annual_insurance' (e.g., with median or 0).
    # 2. Scale all numeric features.
    # We will use a ColumnTransformer, so we need to separate 'annual_insurance' for imputation if it's the only one.
    # Or, if all numeric features might have NaNs (e.g. from joins or future data), use SimpleImputer more broadly.
    # For now, let's assume only annual_insurance needs special NaN handling before scaling.
    
    # If 'annual_insurance' is present and numeric:
    if 'annual_insurance' in numeric_features:
        numeric_transformer_main = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        # Pipeline for 'annual_insurance' specifically
        numeric_transformer_insurance = Pipeline(steps=[
            ('imputer', IterativeImputer(random_state=42)),
            ('scaler', StandardScaler())
        ])
        
        # Identify other numeric features that are not 'annual_insurance'
        other_numeric_features = [col for col in numeric_features if col != 'annual_insurance']

        # Categorical features pipeline:
        # 1. Impute NaNs (if any, with a constant like 'missing'). Our FE script fills with 'unknown'.
        # 2. One-hot encode.
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value_runtime')), # Should not hit if FE worked
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier handling now
        ])

        # Create the ColumnTransformer
        # Order of transformers in list matters for column indexing if transformered df is desired directly for inspection
        preprocessor = ColumnTransformer(
            transformers=[
                ('num_insurance', numeric_transformer_insurance, ['annual_insurance']),
                ('num_main', numeric_transformer_main, other_numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough' # Keep any columns not specified (should be none if lists are correct)
        )
    elif 'annual_insurance' in categorical_features: # Should not happen if dtype is float
        logging.warning("'annual_insurance' found in categorical features. Check its dtype and NaN handling from FE step.")
        # Basic numeric pipeline (excluding annual_insurance)
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        # Basic categorical pipeline (will include annual_insurance if it's object)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value_runtime')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features), # numeric_features here won't have annual_insurance
                ('cat', categorical_transformer, categorical_features) # categorical_features here will have annual_insurance
            ],
            remainder='passthrough'
        )
    else: # 'annual_insurance' not found at all
        logging.warning("'annual_insurance' column not found in the dataset. Proceeding without it.")
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value_runtime')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
    # --- Apply preprocessing ---
    logging.info("Applying preprocessing to training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    logging.info("Preprocessing applied to training data.")

    logging.info("Applying preprocessing to testing data...")
    X_test_processed = preprocessor.transform(X_test)
    logging.info("Preprocessing applied to testing data.")

    # Get feature names after one-hot encoding for creating DataFrames (optional, but good for inspection)
    try:
        # Get feature names from ColumnTransformer
        # For OneHotEncoder, get_feature_names_out is available in recent sklearn versions
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        
        # Construct the full list of feature names
        # Order must match the ColumnTransformer's transformers list
        processed_feature_names = []
        if 'annual_insurance' in numeric_features: # Check how it was actually handled
            if 'annual_insurance' in preprocessor.transformers_[0][2]: # If it was in the first transformer (num_insurance)
                 processed_feature_names.extend(['annual_insurance'])
                 processed_feature_names.extend(other_numeric_features)
            else: # It means annual_insurance was not found or was in other_numeric_features already
                 processed_feature_names.extend(numeric_features) # All numeric features
        elif 'annual_insurance' not in numeric_features and 'annual_insurance' not in categorical_features:
            processed_feature_names.extend(numeric_features) # All numeric features if annual_insurance was missing
        else: # annual_insurance was in categorical or only num_main was used
            processed_feature_names.extend(numeric_features) # All numeric features as defined originally

        processed_feature_names.extend(ohe_feature_names)

        # Capture remainder columns if any were passed through
        # current 'remainder' is 'passthrough', if it was 'drop' this would be simpler.
        # For 'passthrough', need to get names of columns not in numeric/categorical lists.
        # This part can be complex if remainder='passthrough' and original column order isn't strictly maintained.
        # For now, assuming all columns are handled by the transformers.
        # If remainder='drop', this would be simpler.
        # If preprocessor.remainder == 'passthrough':
        #    original_cols = list(X_train.columns)
        #    transformed_cols = []
        #    if 'annual_insurance' in numeric_features: transformed_cols.extend(['annual_insurance'])
        #    if other_numeric_features: transformed_cols.extend(other_numeric_features)
        #    else: transformed_cols.extend(numeric_features) # if annual_insurance wasn't special
        #    transformed_cols.extend(categorical_features)
        #    remainder_cols = [col for col in original_cols if col not in transformed_cols]
        #    processed_feature_names.extend(remainder_cols)


        X_train_processed_df = pd.DataFrame(X_train_processed, columns=processed_feature_names, index=X_train.index)
        X_test_processed_df = pd.DataFrame(X_test_processed, columns=processed_feature_names, index=X_test.index)
        
        logging.info(f"Shape of X_train_processed_df: {X_train_processed_df.shape}")
        logging.info(f"First 5 rows of X_train_processed_df:\\n{X_train_processed_df.head()}")

    except Exception as e:
        logging.warning(f"Could not generate feature names for processed DataFrame: {e}. Processed data remains as numpy arrays.")
        X_train_processed_df = pd.DataFrame(X_train_processed, index=X_train.index) # No column names
        X_test_processed_df = pd.DataFrame(X_test_processed, index=X_test.index)   # No column names


    # --- Save processed data ---
    # We can save the processed arrays/DataFrames and the target variable sets.
    # Alternatively, save the fitted preprocessor pipeline itself for later use.
    # For MLOps, saving the pipeline is often preferred.

    # Saving processed DataFrames and Series
    train_processed_df_path = os.path.join(PREPROCESSED_DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_X_train_processed.csv")
    test_processed_df_path = os.path.join(PREPROCESSED_DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_X_test_processed.csv")
    y_train_path = os.path.join(PREPROCESSED_DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_y_train.csv")
    y_test_path = os.path.join(PREPROCESSED_DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_y_test.csv")

    try:
        X_train_processed_df.to_csv(train_processed_df_path, index=False)
        logging.info(f"Saved X_train_processed to {train_processed_df_path}")
        X_test_processed_df.to_csv(test_processed_df_path, index=False)
        logging.info(f"Saved X_test_processed to {test_processed_df_path}")
        
        y_train.to_csv(y_train_path, index=False, header=True) # Save y_train as Series (or DataFrame)
        logging.info(f"Saved y_train to {y_train_path}")
        y_test.to_csv(y_test_path, index=False, header=True)   # Save y_test as Series (or DataFrame)
        logging.info(f"Saved y_test to {y_test_path}")

        # Save the preprocessor (ColumnTransformer)
        preprocessor_path = os.path.join(PREPROCESSED_DATA_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        logging.info(f"Saved fitted preprocessor to {preprocessor_path}")

    except Exception as e:
        logging.error(f"Error saving processed data or preprocessor: {e}")

    logging.info("Step 2.1: Preprocessing and Pipeline Construction complete.")

if __name__ == '__main__':
    main() 