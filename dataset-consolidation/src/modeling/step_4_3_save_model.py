import os
import json
import joblib
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
MODEL_ARTIFACTS_DIR = os.path.join(WORKSPACE_ROOT, 'dataset-consolidation', 'models')
CURRENT_MODEL_DIR = os.path.join(MODEL_ARTIFACTS_DIR, f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

# Create directories if they don't exist
os.makedirs(CURRENT_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(CURRENT_MODEL_DIR, 'preprocessing'), exist_ok=True)

def save_model_metadata(metadata_dict, filepath):
    """Save model metadata to a JSON file."""
    # Convert any non-serializable objects to strings in a new dict
    json_safe_metadata = {}
    for key, value in metadata_dict.items():
        if isinstance(value, dict):
            json_safe_metadata[key] = {k: str(v) for k, v in value.items()}
        else:
            json_safe_metadata[key] = str(value) if not isinstance(value, (str, int, float, bool, list, type(None))) else value
            if isinstance(value, list):
                json_safe_metadata[key] = [str(item) if not isinstance(item, (str, int, float, bool, type(None))) else item for item in value]
    
    with open(filepath, 'w') as f:
        json.dump(json_safe_metadata, f, indent=4)
    logging.info(f"Saved model metadata to: {filepath}")

def save_feature_importance(feature_importance_df, filepath):
    """Save feature importance information."""
    feature_importance_df.to_csv(filepath, index=True)
    logging.info(f"Saved feature importance to: {filepath}")

def find_latest_file(directory, pattern):
    """Find the latest file in directory matching the pattern."""
    files = [f for f in os.listdir(directory) if f.endswith(pattern)]
    if not files:
        return None
    return max(files, key=lambda x: os.path.getctime(os.path.join(directory, x)))

def get_safe_params(model):
    """Get model parameters in a JSON-safe format."""
    if not hasattr(model, 'get_params'):
        return str(model)
    
    params = model.get_params()
    safe_params = {}
    
    for key, value in params.items():
        # Handle nested estimators
        if hasattr(value, 'get_params'):
            safe_params[key] = get_safe_params(value)
        # Handle lists/tuples of estimators
        elif isinstance(value, (list, tuple)):
            safe_params[key] = [get_safe_params(v) if hasattr(v, 'get_params') else str(v) for v in value]
        # Handle all other cases
        else:
            safe_params[key] = str(value)
    
    return safe_params

def main():
    try:
        # Find and load the latest ensemble model
        ensemble_dir = os.path.join(MODEL_ARTIFACTS_DIR, 'ensemble')
        model_filename = find_latest_file(ensemble_dir, '.joblib')
        if not model_filename:
            raise FileNotFoundError("No model file found in ensemble directory")
        
        model_path = os.path.join(ensemble_dir, model_filename)
        logging.info(f"Loading model from: {model_path}")
        final_model = joblib.load(model_path)
        
        # Load feature names from the latest file in base_models
        base_models_dir = os.path.join(MODEL_ARTIFACTS_DIR, 'base_models')
        feature_names_file = find_latest_file(base_models_dir, 'selected_feature_names.txt')
        if feature_names_file:
            feature_names_path = os.path.join(base_models_dir, feature_names_file)
            with open(feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f.readlines()]
            logging.info(f"Loaded feature names from: {feature_names_path}")
        else:
            feature_names = None
            logging.warning("No feature names file found")
        
        # Save the model
        model_save_path = os.path.join(CURRENT_MODEL_DIR, 'final_model.joblib')
        joblib.dump(final_model, model_save_path)
        logging.info(f"Saved final model to: {model_save_path}")
        
        # Create and save model metadata
        metadata = {
            'model_version': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'model_type': str(type(final_model).__name__),
            'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_parameters': get_safe_params(final_model),
            'feature_names': feature_names,
            'target_variable': 'price_log',
            'metrics': {
                'r2_score': None,  # These would need to be loaded from evaluation results
                'rmse': None,
                'mae': None
            },
            'preprocessing_steps': "Feature engineering steps are documented in step_1_4_feature_engineering.py"
        }
        
        metadata_path = os.path.join(CURRENT_MODEL_DIR, 'model_metadata.json')
        save_model_metadata(metadata, metadata_path)
        
        # Copy performance plots if they exist
        plot_files = [
            '20250520_204953_actual_vs_predicted.png',
            '20250520_204953_residuals_vs_predicted.png',
            '20250520_204953_residuals_histogram.png',
            '20250520_204953_residuals_qq_plot.png'
        ]
        
        os.makedirs(os.path.join(CURRENT_MODEL_DIR, 'plots'), exist_ok=True)
        for plot_file in plot_files:
            src_path = os.path.join(ensemble_dir, plot_file)
            if os.path.exists(src_path):
                import shutil
                dst_path = os.path.join(CURRENT_MODEL_DIR, 'plots', plot_file)
                shutil.copy2(src_path, dst_path)
                logging.info(f"Copied plot: {plot_file}")
        
        # Create README
        readme_content = f"""# Car Price Prediction Model

## Model Version: {metadata['model_version']}

This directory contains the trained model and associated artifacts for the car price prediction system.

### Directory Structure
- `final_model.joblib`: The trained model (Stacking Ensemble)
- `model_metadata.json`: Detailed model metadata and parameters
- `plots/`: Model performance visualization plots
  - Actual vs Predicted Values
  - Residuals Analysis
  - QQ Plot
  - Residuals Histogram

### Model Information
- Model Type: {metadata['model_type']}
- Training Date: {metadata['training_date']}
- Target Variable: {metadata['target_variable']}

### Usage
To use this model:
1. Prepare the input data according to feature engineering steps
2. Load the model
3. Make predictions

Example:
```python
import joblib
import numpy as np

# Load the model
model = joblib.load('final_model.joblib')

# Make predictions (ensure X_new has the same features as during training)
predictions = model.predict(X_new)

# Convert predictions back from log scale
predictions_original_scale = np.exp(predictions) - 1
```

### Notes
- The model expects input features as described in the model_metadata.json file
- The target variable (price) predictions are log-transformed and need to be converted back
- Feature engineering steps must match those used during training
"""
        
        readme_path = os.path.join(CURRENT_MODEL_DIR, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logging.info(f"Created README at: {readme_path}")
        
        logging.info("Successfully saved all model artifacts!")
        
    except Exception as e:
        logging.error(f"Error saving model artifacts: {e}")
        raise

if __name__ == '__main__':
    main() 