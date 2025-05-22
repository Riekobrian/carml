import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os
import logging
import glob
import numpy as np
from datetime import datetime

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))

LOG_DIR = os.path.join(WORKSPACE_ROOT, "dataset-consolidation", "logs", "modeling")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"step_4_1_shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)

# Update paths to match your project structure
FINAL_DATA_DIR = os.path.join(WORKSPACE_ROOT, 'dataset-consolidation', 'data', 'final')
MODELS_DIR = os.path.join(WORKSPACE_ROOT, 'dataset-consolidation', 'models')
BASE_MODELS_DIR = os.path.join(MODELS_DIR, 'base_models')
ENSEMBLE_MODELS_DIR = os.path.join(MODELS_DIR, 'ensemble')
DOCS_DIR = os.path.join(WORKSPACE_ROOT, 'dataset-consolidation', 'docs')
SHAP_DIR = os.path.join(DOCS_DIR, "shap_plots")
os.makedirs(SHAP_DIR, exist_ok=True)

def find_latest_file(directory, pattern_middle_or_end, initial_wildcard=True, end_suffix_extension=".csv"):
    """Finds the most recent file in a directory matching a pattern."""
    if initial_wildcard:
        search_pattern = os.path.join(directory, f"*{pattern_middle_or_end}{end_suffix_extension}")
    else:
        search_pattern = os.path.join(directory, f"{pattern_middle_or_end}_*{end_suffix_extension}")
    
    logging.info(f"Searching for file with pattern: {search_pattern}")
    files = glob.glob(search_pattern)
    if not files:
        logging.error(f"No files found for pattern: {search_pattern} in {directory}")
        return None
    latest_file = max(files, key=os.path.getctime)
    logging.info(f"Found latest file: {latest_file}")
    return latest_file

def run_shap_analysis():
    logging.info("Starting SHAP Analysis (Step 4.1)...")
    logging.info(f"SHAP plots will be saved to: {SHAP_DIR}")

    # First, load the selected feature names
    logging.info("Loading selected feature names...")
    selected_features_path = find_latest_file(BASE_MODELS_DIR, "selected_feature_names", initial_wildcard=True, end_suffix_extension=".txt")
    if not selected_features_path:
        logging.critical("Could not find selected_feature_names.txt. Please ensure feature selection step completed successfully.")
        return
        
    with open(selected_features_path, 'r') as f:
        selected_features = [line.strip() for line in f.readlines()]
    logging.info(f"Loaded {len(selected_features)} selected feature names.")

    # Load data
    logging.info("Loading processed data...")
    try:
        x_train_path = find_latest_file(FINAL_DATA_DIR, "X_train_processed", initial_wildcard=True)
        x_test_path = find_latest_file(FINAL_DATA_DIR, "X_test_processed", initial_wildcard=True)
        y_train_path = find_latest_file(FINAL_DATA_DIR, "y_train", initial_wildcard=True)

        if not all([x_train_path, x_test_path, y_train_path]):
            logging.critical("Could not find all required data files. Please check data directory.")
            return

        x_train_df = pd.read_csv(x_train_path)
        x_test_df = pd.read_csv(x_test_path)
        y_train_df = pd.read_csv(y_train_path)
        
        # Filter to selected features
        x_train_df = x_train_df[selected_features]
        x_test_df = x_test_df[selected_features]
        
        logging.info(f"Loaded and filtered training data shape: {x_train_df.shape}")
        logging.info(f"Loaded and filtered test data shape: {x_test_df.shape}")

    except Exception as e:
        logging.error(f"Error loading data for SHAP analysis: {e}")
        return

    # Load the champion model (Stacking Regressor)
    logging.info(f"Loading champion Stacking Regressor model from: {ENSEMBLE_MODELS_DIR}")
    try:
        model_path = find_latest_file(ENSEMBLE_MODELS_DIR, "StackingRegressor_RF_GB_Linear", initial_wildcard=True, end_suffix_extension=".joblib")
        if not model_path:
            logging.critical(f"Champion model not found in {ENSEMBLE_MODELS_DIR}. Please ensure model training completed successfully.")
            return
        
        model = joblib.load(model_path)
        logging.info(f"Successfully loaded model: {model_path}")
    except Exception as e:
        logging.error(f"Error loading the model: {e}")
        return

    # Prepare data for SHAP
    N_BACKGROUND_SAMPLES = 100
    if len(x_train_df) > N_BACKGROUND_SAMPLES:
        background_data = shap.sample(x_train_df, N_BACKGROUND_SAMPLES, random_state=42)
    else:
        background_data = x_train_df

    N_EXPLAIN_SAMPLES = 200
    if len(x_test_df) > N_EXPLAIN_SAMPLES:
        explain_data_sample = x_test_df.sample(N_EXPLAIN_SAMPLES, random_state=42)
    else:
        explain_data_sample = x_test_df

    logging.info(f"Using {background_data.shape[0]} samples as background data for SHAP KernelExplainer.")
    logging.info(f"Explaining predictions for {explain_data_sample.shape[0]} samples from the test set.")

    # Initialize SHAP KernelExplainer
    logging.info("Initializing SHAP KernelExplainer...")
    try:
        explainer = shap.KernelExplainer(model.predict, background_data)
        logging.info("SHAP KernelExplainer initialized.")

        logging.info("Calculating SHAP values... (this may take some time for KernelExplainer)")
        shap_values_sample = explainer.shap_values(explain_data_sample)
        logging.info("SHAP values calculated.")

    except Exception as e:
        logging.error(f"Error during SHAP explainer initialization or value calculation: {e}", exc_info=True)
        return
        
    if not isinstance(explain_data_sample, pd.DataFrame):
        explain_data_sample_df = pd.DataFrame(explain_data_sample, columns=x_test_df.columns)
    else:
        explain_data_sample_df = explain_data_sample

    # Generate and save SHAP plots
    try:
        # 1. Summary plot (bar)
        logging.info("Generating SHAP summary bar plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_sample, explain_data_sample_df, plot_type="bar", show=False)
        plt.title("SHAP Global Feature Importance (Bar)")
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("SHAP summary bar plot saved.")

        # 2. Summary plot (beeswarm)
        logging.info("Generating SHAP summary beeswarm plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_sample, explain_data_sample_df, show=False)
        plt.title("SHAP Feature Importance and Impact (Beeswarm)")
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, "shap_summary_beeswarm.png"), dpi=300, bbox_inches='tight')
        plt.close()
        logging.info("SHAP summary beeswarm plot saved.")

        # 3. Dependence plots for top features
        N_TOP_FEATURES_FOR_DEPENDENCE = 10
        logging.info(f"Generating SHAP dependence plots for top {N_TOP_FEATURES_FOR_DEPENDENCE} features...")
        
        abs_shap_values = np.abs(shap_values_sample)
        mean_abs_shap = np.mean(abs_shap_values, axis=0)
        
        feature_names = explain_data_sample_df.columns
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'shap_importance': mean_abs_shap})
        top_features = feature_importance_df.sort_values(by='shap_importance', ascending=False).head(N_TOP_FEATURES_FOR_DEPENDENCE)['feature'].tolist()

        for feature_name in top_features:
            logging.info(f"Generating dependence plot for feature: {feature_name}")
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(feature_name, shap_values_sample, explain_data_sample_df, show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(SHAP_DIR, f"shap_dependence_{feature_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()
        logging.info(f"SHAP dependence plots saved for top {N_TOP_FEATURES_FOR_DEPENDENCE} features.")

    except Exception as e:
        logging.error(f"Error during SHAP plot generation: {e}", exc_info=True)
        return

    logging.info("Step 4.1: SHAP Analysis completed successfully.")
    logging.info(f"All SHAP plots have been saved in: {SHAP_DIR}")

if __name__ == "__main__":
    run_shap_analysis() 