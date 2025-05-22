import pandas as pd
import numpy as np
import os
import logging
import time
from datetime import datetime
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # D:\SBT-JAPAN\Alpha GO

LOG_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "logs", "modeling")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"step_3_2_feature_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()]
)

# Define data/model paths
FEATURE_ENGINEERED_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "processed")
PREPROCESSED_DATA_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "data", "final") # For X_train_processed, y_train, and saving selected data
MODELS_OUTPUT_DIR = os.path.join(WORKSPACE_ROOT, "models") # General models folder
BASE_MODELS_DIR = os.path.join(MODELS_OUTPUT_DIR, "base_models") # For tuned base models and feature selection outputs
os.makedirs(BASE_MODELS_DIR, exist_ok=True)

# --- Helper function to find the latest file ---
def find_latest_feature_engineered_file(base_dir, relative_path_to_dir, file_name_suffix):
    # Construct a pattern that glob should be able to use more reliably
    # Example: D:\\SBT-JAPAN\\Alpha GO\\data\\processed\\*_cars_feature_engineered.csv
    search_pattern = os.path.join(base_dir, relative_path_to_dir, f"*{file_name_suffix}")
    
    logging.info(f"Attempting to find files with pattern: {search_pattern}")
    files = glob.glob(search_pattern)
    
    if not files:
        logging.error(f"No files found matching {search_pattern}")
        # Let's try to list the directory content directly via os.listdir for diagnostics
        try:
            actual_dir_path = os.path.join(base_dir, relative_path_to_dir)
            logging.info(f"Contents of directory {actual_dir_path} (via os.listdir): {os.listdir(actual_dir_path)}")
        except Exception as e:
            logging.error(f"Could not list directory {actual_dir_path} for diagnostics: {e}")
        return None
        
    latest_file = max(files, key=os.path.getctime)
    logging.info(f"Found latest file: {latest_file}")
    return latest_file

# --- Keep the old find_latest_file for other file types for now ---
def find_latest_file_original(directory, pattern_suffix_main_name, initial_wildcard=True, end_suffix_extension=".csv"): # Renamed to avoid conflict, and params adjusted
    # Now expects a pattern like *X_train_processed.csv or *y_train.csv
    if initial_wildcard:
        search_pattern = os.path.join(directory, f"*{pattern_suffix_main_name}{end_suffix_extension}")
    else: # Fallback or specific case if needed for other types of patterns
        search_pattern = os.path.join(directory, f"{pattern_suffix_main_name}_*{end_suffix_extension}") 
    
    logging.info(f"Attempting to find files (original_logic) with pattern: {search_pattern}")
    files = glob.glob(search_pattern)
    if not files:
        logging.error(f"No files found matching {search_pattern} in {directory}")
        return None
    latest_file = max(files, key=os.path.getctime)
    logging.info(f"Found latest file: {latest_file}")
    return latest_file

def main():
    logging.info("Starting Step 3.2: Feature Selection / Dimensionality Reduction.")
    
    # --- 1. Load Original Feature-Engineered Data (for preprocessor fitting) ---
    latest_fe_file_path = find_latest_feature_engineered_file(WORKSPACE_ROOT, os.path.join("dataset-consolidation", "data", "processed"), "_cars_feature_engineered.csv")
    
    if not latest_fe_file_path:
        logging.critical(f"No feature engineered CSV file found in {os.path.join(WORKSPACE_ROOT, 'dataset-consolidation', 'data', 'processed')}. Exiting.")
        return
    logging.info(f"Loading original feature-engineered data from: {latest_fe_file_path}")
    df_original_fe = pd.read_csv(latest_fe_file_path)

    TARGET_COLUMN_ORIGINAL = 'price_log'
    if TARGET_COLUMN_ORIGINAL not in df_original_fe.columns:
        logging.error(f"Target column '{TARGET_COLUMN_ORIGINAL}' not found in {latest_fe_file_path}. Exiting.")
        return

    X_original_fe = df_original_fe.drop(columns=[TARGET_COLUMN_ORIGINAL])
    y_original_fe = df_original_fe[TARGET_COLUMN_ORIGINAL]
    logging.info(f"Original full feature set X_original_fe shape: {X_original_fe.shape}")

    X_train_unprocessed, _, _, _ = train_test_split(
        X_original_fe, y_original_fe, test_size=0.2, random_state=42
    )
    logging.info(f"X_train_unprocessed (for preprocessor fitting) shape: {X_train_unprocessed.shape}")

    # --- 2. Re-define and Fit Preprocessor (to get consistent feature names) ---
    numeric_features = X_train_unprocessed.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_train_unprocessed.select_dtypes(include='object').columns.tolist()
    
    preprocessor_for_names = None
    other_numeric_features = []

    if 'annual_insurance' in numeric_features:
        numeric_transformer_main = Pipeline(steps=[('scaler', StandardScaler())])
        numeric_transformer_insurance = Pipeline(steps=[
            ('imputer', IterativeImputer(random_state=42, max_iter=10, tol=1e-3)), # Standard params
            ('scaler', StandardScaler())
        ])
        other_numeric_features = [col for col in numeric_features if col != 'annual_insurance']
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value_runtime')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor_for_names = ColumnTransformer(
            transformers=[
                ('num_insurance', numeric_transformer_insurance, ['annual_insurance']),
                ('num_main', numeric_transformer_main, other_numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough'
        )
    elif 'annual_insurance' not in numeric_features and 'annual_insurance' not in categorical_features :
        logging.warning("'annual_insurance' not in features for preprocessor_for_names. Using standard numeric processing.")
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())]) # No imputation for other numerics here
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing_value_runtime')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor_for_names = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ], remainder='passthrough'
        )
    else:
         logging.error("Preprocessor setup for feature names failed due to 'annual_insurance' location. Exiting.")
         return

    logging.info("Fitting preprocessor_for_names on X_train_unprocessed...")
    preprocessor_for_names.fit(X_train_unprocessed)

    try:
        ohe_feature_names = preprocessor_for_names.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)
        transformed_feature_names = []
        if 'annual_insurance' in numeric_features: # This logic must exactly match the ColumnTransformer structure
            transformed_feature_names.extend(['annual_insurance']) # From 'num_insurance'
            transformed_feature_names.extend(other_numeric_features) # From 'num_main'
        else:
            transformed_feature_names.extend(numeric_features) # From 'num'
        transformed_feature_names.extend(ohe_feature_names) # From 'cat'
        
        # Handle remainder='passthrough' features if any (though unlikely with typical setup)
        if preprocessor_for_names.remainder == 'passthrough' and hasattr(preprocessor_for_names, '_remainder_feature_names_out'):
             remainder_features = preprocessor_for_names.get_feature_names_out()
             # This part is tricky, need to carefully extract only remainder names
             # For simplicity, assuming remainder is empty or handled if it arises
        
        logging.info(f"Successfully generated {len(transformed_feature_names)} transformed feature names.")
    except Exception as e:
        logging.error(f"Could not get transformed feature names from preprocessor: {e}. Exiting.")
        return

    # --- 3. Load Processed Data (X_train_processed, y_train) ---
    X_train_processed_path = find_latest_file_original(PREPROCESSED_DATA_DIR, 'X_train_processed.csv', initial_wildcard=True, end_suffix_extension="") # Suffix is now part of main name
    y_train_path = find_latest_file_original(PREPROCESSED_DATA_DIR, 'y_train.csv', initial_wildcard=True, end_suffix_extension="")

    if not X_train_processed_path or not y_train_path:
        logging.critical(f"X_train_processed or y_train file not found in {PREPROCESSED_DATA_DIR}. Exiting.")
        return
        
    logging.info(f"Loading X_train_processed from: {X_train_processed_path}")
    X_train_processed_df = pd.read_csv(X_train_processed_path)
    logging.info(f"Loading y_train from: {y_train_path}")
    y_train_series = pd.read_csv(y_train_path).squeeze()

    if X_train_processed_df.shape[1] != len(transformed_feature_names):
        logging.error(f"Mismatch in column count between loaded X_train_processed ({X_train_processed_df.shape[1]}) and generated feature names ({len(transformed_feature_names)}). Check preprocessor logic. Exiting.")
        return
    X_train_processed_df.columns = transformed_feature_names
    logging.info(f"X_train_processed_df shape after loading and naming: {X_train_processed_df.shape}")

    # --- 4. Load Tuned Random Forest Model ---
    tuned_rf_model_path = find_latest_file_original(BASE_MODELS_DIR, "Random_Forest_tuned", initial_wildcard=False, end_suffix_extension=".joblib")
    if not tuned_rf_model_path:
        logging.critical(f"Tuned Random Forest model not found in {BASE_MODELS_DIR}. Please run tuning script first. Exiting.")
        return
    logging.info(f"Loading tuned Random Forest model from: {tuned_rf_model_path}")
    try:
        tuned_rf_model = joblib.load(tuned_rf_model_path)
    except Exception as e:
        logging.error(f"Error loading tuned Random Forest model: {e}. Exiting.")
        return

    # --- 5. Feature Importances ---
    logging.info("Extracting feature importances from tuned Random Forest.")
    importances = tuned_rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': transformed_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)
    logging.info(f"""Top 20 feature importances:
{feature_importance_df.head(20)}""")

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df.head(30), palette='viridis_r')
    plt.title('Top 30 Feature Importances from Tuned Random Forest')
    plt.tight_layout()
    plot_filename = os.path.join(BASE_MODELS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_feature_importances.png")
    plt.savefig(plot_filename)
    logging.info(f"Saved feature importance plot to {plot_filename}")
    plt.close()

    # --- 6. Apply Feature Selection using SelectFromModel ---
    logging.info("Applying feature selection using SelectFromModel with tuned Random Forest.")
    selector = SelectFromModel(tuned_rf_model, threshold='median', prefit=True)
    
    X_train_selected_data = selector.transform(X_train_processed_df)
    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [transformed_feature_names[i] for i in selected_feature_indices]

    logging.info(f"Original number of features: {X_train_processed_df.shape[1]}")
    logging.info(f"Number of features selected: {X_train_selected_data.shape[1]}")
    # logging.info(f"Selected feature names (first 20): {selected_feature_names[:20]}")

    selected_features_path = os.path.join(BASE_MODELS_DIR, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_selected_feature_names.txt")
    with open(selected_features_path, 'w') as f:
        for name in selected_feature_names:
            f.write(f"{name}\n")
    logging.info(f"Saved selected feature names list to {selected_features_path}")

    X_train_selected_df = pd.DataFrame(X_train_selected_data, columns=selected_feature_names, index=X_train_processed_df.index)

    # --- 7. Save X_train_selected_df ---
    current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    x_train_selected_filename = f"{current_timestamp}_X_train_processed_selected.csv"
    x_train_selected_path = os.path.join(PREPROCESSED_DATA_DIR, x_train_selected_filename)
    X_train_selected_df.to_csv(x_train_selected_path, index=False)
    logging.info(f"Saved X_train_selected_df to: {x_train_selected_path}")

    # --- 8. Load and Process X_test_processed ---
    X_test_processed_path = find_latest_file_original(PREPROCESSED_DATA_DIR, 'X_test_processed.csv', initial_wildcard=True, end_suffix_extension="")
    if not X_test_processed_path:
        logging.critical(f"X_test_processed.csv file not found in {PREPROCESSED_DATA_DIR}. Cannot create X_test_selected. Exiting.")
        return
    logging.info(f"Loading X_test_processed from: {X_test_processed_path}")
    X_test_processed_df = pd.read_csv(X_test_processed_path)

    if X_test_processed_df.shape[1] == len(transformed_feature_names):
        X_test_processed_df.columns = transformed_feature_names
        logging.info(f"Assigned {len(transformed_feature_names)} column names to X_test_processed_df.")
    else:
        logging.error(f"CRITICAL: Mismatch in column count between loaded X_test_processed ({X_test_processed_df.shape[1]}) and generated feature names ({len(transformed_feature_names)}). Cannot apply selection.")
        return

    logging.info("Applying feature selection to X_test_processed_df...")
    X_test_selected_data = selector.transform(X_test_processed_df)
    X_test_selected_df = pd.DataFrame(X_test_selected_data, columns=selected_feature_names, index=X_test_processed_df.index)
    logging.info(f"X_test_selected_df shape: {X_test_selected_df.shape}")

    # --- 9. Save X_test_selected_df ---
    x_test_selected_filename = f"{current_timestamp}_X_test_processed_selected.csv"
    x_test_selected_path = os.path.join(PREPROCESSED_DATA_DIR, x_test_selected_filename)
    X_test_selected_df.to_csv(x_test_selected_path, index=False)
    logging.info(f"Saved X_test_selected_df to: {x_test_selected_path}")
    
    # --- 10. Save the selector itself (optional but good practice) ---
    selector_path = os.path.join(BASE_MODELS_DIR, f"{current_timestamp}_feature_selector.joblib")
    joblib.dump(selector, selector_path)
    logging.info(f"Saved feature selector (SelectFromModel) to: {selector_path}")


    # --- 11. Retrain and Evaluate Models on Reduced Feature Set ---
    logging.info("Retraining and evaluating models on the reduced feature set.")
    models_to_re_evaluate = {}
    
    # Reload the already tuned RF model (it's not modified by SelectFromModel if prefit=True)
    if tuned_rf_model:
         models_to_re_evaluate["Tuned Random Forest (Selected Feats)"] = tuned_rf_model
    
    tuned_gb_model_path = find_latest_file_original(BASE_MODELS_DIR, "Gradient_Boosting_tuned", initial_wildcard=False, end_suffix_extension=".joblib")
    if not tuned_gb_model_path:
        logging.warning(f"Tuned Gradient Boosting model (.joblib) file not found with prefix 'Gradient_Boosting_tuned' in {BASE_MODELS_DIR}. GB model will not be re-evaluated on selected features.")
    
    cv_results_selected_features = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model_instance in models_to_re_evaluate.items():
        logging.info(f"Cross-validating {model_name} on selected features (X_train_selected_df)...")
        start_time = time.time()
        cv_scores = cross_val_score(model_instance, X_train_selected_df, y_train_series, cv=kf, scoring='r2', n_jobs=-1)
        cv_time = time.time() - start_time
        
        logging.info(f"{model_name} - CV R2 scores: {cv_scores}")
        logging.info(f"{model_name} - Mean CV R2: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
        logging.info(f"{model_name} - CV Time: {cv_time:.2f}s")
        
        cv_results_selected_features.append({
            "Model": model_name,
            "Mean CV R2": np.mean(cv_scores),
            "Std CV R2": np.std(cv_scores),
            "CV Time (s)": cv_time
        })

    if cv_results_selected_features:
        results_selected_df = pd.DataFrame(cv_results_selected_features)
        results_selected_df = results_selected_df.sort_values(by="Mean CV R2", ascending=False).reset_index(drop=True)
        logging.info(f"\n--- Model Performance on Selected Features (Cross-Validation) ---\n{results_selected_df.to_string()}")
    else:
        logging.info("\nNo model evaluation results on selected features to display.")

    logging.info("Step 3.2: Feature Selection script finished.")

if __name__ == '__main__':
    main() 