import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

class CarPricePredictor:
    """
    A class to handle car price predictions using the trained stacking model.
    """
    
    def __init__(self, model_dir=None):
        """
        Initialize the predictor with paths to required model artifacts.
        
        Args:
            model_dir (str): Directory containing model artifacts. If None, uses default path.
        """
        if model_dir is None:
            # Default to the deployment artifacts directory
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'src', 'deployment', 'model_artifacts', 'stacking_model_cond_fix'
            )
        
        self.model_path = os.path.join(model_dir, 'model.joblib')
        self.preprocessor_path = os.path.join(model_dir, 'preprocessor_fitted_for_cond_fix_model.joblib')
        self.feature_names_path = os.path.join(model_dir, 'selected_feature_names_prefixed.txt')
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Load model artifacts
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load all required model artifacts."""
        try:
            self.logger.info("Loading model artifacts...")
            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)
            
            with open(self.feature_names_path, 'r') as f:
                self.selected_features = [line.strip() for line in f.readlines()]
            
            self.logger.info("Successfully loaded all model artifacts.")
        except Exception as e:
            self.logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def _validate_input(self, input_data):
        """
        Validate input data format and required features.
        
        Args:
            input_data (dict or pd.DataFrame): Input data to validate
        
        Returns:
            pd.DataFrame: Validated and formatted input data
        """
        required_numeric_features = [
            'mileage_per_cc',
            'car_age',
            'car_age_squared',
            'annual_insurance',
            'engine_size_cc_num',
            'engine_size_cc_log',
            'mileage_num',
            'mileage_log',
            'mileage_per_year',
            'horse_power_num',
            'horse_power_log',
            'torque_num',
            'torque_log',
            'power_per_cc',
            'acceleration_num',
            'seats_num',
            'is_luxury_make'
        ]
        
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        elif not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input must be a dictionary or pandas DataFrame")
        
        # Ensure imputable columns like 'annual_insurance' exist, fill with NaN if not provided by user.
        # This allows the IterativeImputer in the preprocessor to handle them.
        imputable_features_raw = ['annual_insurance'] # Add other features here if they are similarly handled
        
        # Check if data has prefixes to determine which version of the feature name to check/create
        has_prefixes = any(col.startswith(('num_insurance__', 'num_main__', 'cat__')) for col in input_data.columns)

        for feature_raw_name in imputable_features_raw:
            feature_to_check = feature_raw_name
            if has_prefixes:
                if feature_raw_name == 'annual_insurance': # Specific prefix for annual_insurance
                    feature_to_check = f'num_insurance__{feature_raw_name}'
                # Add more specific prefix logic here if other imputable features have different prefixes
                # else: 
                #     feature_to_check = f'num_main__{feature_raw_name}' # Example for other numeric
            
            if feature_to_check not in input_data.columns:
                self.logger.info(f"Feature '{feature_to_check}' not provided by user. Adding it with NaN for imputation by preprocessor.")
                input_data[feature_to_check] = np.nan

        # Check if data has prefixed feature names (re-evaluate in case column was added)
        has_prefixes = any(col.startswith(('num_insurance__', 'num_main__', 'cat__')) for col in input_data.columns)

        if has_prefixes:
            # For prefixed data, check if we have all required numeric features with prefixes
            prefixed_numeric_features = [
                f'num_insurance__{f}' if f == 'annual_insurance' else f'num_main__{f}'
                for f in required_numeric_features
            ]
            missing_features = [f for f in prefixed_numeric_features if f not in input_data.columns]
            if missing_features:
                raise ValueError(f"Missing required numeric features: {missing_features}")
        else:
            # For raw data, check for all required features including categorical ones
            required_features = required_numeric_features + [
                'usage_type_clean',
                'body_type_cleaned',
                'make_name_cleaned',
                'transmission_cleaned',
                'drive_type_cleaned',
                'fuel_type_cleaned',
                'condition_clean',
                'model_name_cleaned',
                'make_model_cleaned'
            ]
            missing_features = [f for f in required_features if f not in input_data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
        
        return input_data
    
    def predict(self, input_data):
        """
        Make price predictions for input data.
        
        Args:
            input_data (dict or pd.DataFrame): Input data for prediction
        
        Returns:
            dict: Prediction results including predicted price and confidence metrics
        """
        try:
            # Validate and format input
            input_df = self._validate_input(input_data)
            self.logger.info(f"Making prediction for {len(input_df)} samples")
            
            # Check if data has prefixed feature names
            has_prefixes = any(col.startswith(('num_insurance__', 'num_main__', 'cat__')) for col in input_df.columns)
            
            if not has_prefixes:
                # Preprocess input data only if it's not already preprocessed
                X_processed = self.preprocessor.transform(input_df)
                X_processed_df = pd.DataFrame(
                    X_processed,
                    columns=self.preprocessor.get_feature_names_out(),
                    index=input_df.index
                )
            else:
                X_processed_df = input_df
            
            # Select required features
            X_final = X_processed_df[self.selected_features]
            
            # Make prediction
            pred_log = self.model.predict(X_final)
            
            # Convert log prediction back to original scale
            pred_price = np.exp(pred_log)
            
            # Prepare results
            results = {
                'predicted_price': pred_price.tolist(),
                'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': 'stacking_model_cond_fix'
            }
            
            self.logger.info("Successfully generated predictions")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            raise

    def predict_with_model(self, input_data, model_path):
        """
        Make price predictions for input data using a specified model file.

        Args:
            input_data (dict or pd.DataFrame): Input data for prediction
            model_path (str): Path to the .joblib model file

        Returns:
            dict: Prediction results including predicted price
        """
        try:
            # Validate and format input
            input_df = self._validate_input(input_data)
            self.logger.info(f"Making prediction for {len(input_df)} samples using model from {model_path}")

            # Load the specified model
            individual_model = joblib.load(model_path)
            
            # Check if data has prefixed feature names
            has_prefixes = any(col.startswith(('num_insurance__', 'num_main__', 'cat__')) for col in input_df.columns)
            
            if not has_prefixes:
                # Preprocess input data only if it's not already preprocessed
                # Use the class's preprocessor
                X_processed = self.preprocessor.transform(input_df)
                X_processed_df = pd.DataFrame(
                    X_processed,
                    columns=self.preprocessor.get_feature_names_out(),
                    index=input_df.index
                )
            else:
                X_processed_df = input_df
            
            # A robust way is to ensure the preprocessor used for these base models during their training
            # produced features in a consistent order, and that the current self.preprocessor does the same.

            # If the loaded model has 'feature_names_in_', use them
            if hasattr(individual_model, 'feature_names_in_'):
                model_expected_raw_features = list(individual_model.feature_names_in_) # These are non-prefixed

                # Create a de-prefixed version of X_processed_df's columns for mapping
                deprefixed_cols_map = {}
                current_prefixed_cols = list(X_processed_df.columns)

                for prefixed_col in current_prefixed_cols:
                    raw_name = prefixed_col
                    if prefixed_col.startswith('num_main__'):
                        raw_name = prefixed_col.split('num_main__', 1)[1]
                    elif prefixed_col.startswith('cat__'):
                        raw_name = prefixed_col.split('cat__', 1)[1]
                    elif prefixed_col.startswith('num_insurance__'):
                        raw_name = prefixed_col.split('num_insurance__', 1)[1]
                    elif prefixed_col.startswith('remainder__'): # Handle remainder if present
                        raw_name = prefixed_col.split('remainder__', 1)[1]
                    # Add other specific prefix patterns if they exist from your preprocessor
                    deprefixed_cols_map[prefixed_col] = raw_name
                
                # Create a temporary DataFrame with de-prefixed column names
                # Important: X_processed_df contains the *data* correctly transformed. We are just renaming columns.
                X_deprefixed_df = X_processed_df.rename(columns=deprefixed_cols_map)
                
                # Now select using the model_expected_raw_features from X_deprefixed_df
                # Check if all expected raw features are now available as columns in X_deprefixed_df
                missing_in_deprefixed = [f for f in model_expected_raw_features if f not in X_deprefixed_df.columns]
                if missing_in_deprefixed:
                    self.logger.error(f"After de-prefixing, model {model_path} still expects features not present: {missing_in_deprefixed}")
                    self.logger.debug(f"Available de-prefixed columns after rename: {list(X_deprefixed_df.columns)}")
                    self.logger.debug(f"Model expected raw features: {model_expected_raw_features}")
                    raise ValueError(f"Cannot align features for model {model_path}. Missing de-prefixed features: {missing_in_deprefixed}")

                # Ensure the order of columns matches what the model expects
                X_final_individual = X_deprefixed_df[model_expected_raw_features]
                self.logger.info(f"Using de-prefixed and selected features for model {model_path}. Shape: {X_final_individual.shape}. Columns: {X_final_individual.columns[:5].tolist()}...")

            else:
                # Fallback: if model doesn't have feature_names_in_, this path is risky.
                # The base models DO have feature_names_in_ as per logs, so this path shouldn't be taken for them.
                # If it were, it means the loaded model isn't a typical scikit-learn estimator with this attribute.
                # Using self.selected_features (stacker's features) would be incorrect as they are prefixed and a subset.
                # The most plausible (though still risky) fallback would be to use all columns from X_processed_df,
                # hoping the model was trained on prefixed names from a similar preprocessor and somehow lost feature_names_in_.
                # However, given the errors, base models expect NON-PREFIXED names.
                self.logger.warning(f"Model {model_path} does not have 'feature_names_in_'. This is unexpected for the base models.")
                self.logger.warning(f"Attempting to use all (prefixed) columns from the current preprocessor output: {list(X_processed_df.columns[:5])}...")
                X_final_individual = X_processed_df[self.preprocessor.get_feature_names_out()] # Use all prefixed features from current preprocessor

            # Make prediction
            pred_log = individual_model.predict(X_final_individual)
            
            # Convert log prediction back to original scale
            pred_price = np.exp(pred_log)
            
            results = {
                'predicted_price': pred_price.tolist(),
            }
            
            self.logger.info(f"Successfully generated predictions with {model_path}")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during prediction with model {model_path}: {e}")
            raise

def main():
    """Example usage of the CarPricePredictor class."""
    # Example input data
    sample_input = {
        'mileage_per_cc': 15.5,
        'car_age': 5,
        'annual_insurance': 50000,
        'engine_size_cc': 2000,
        'usage_type': 'Kenyan Used',
        'body_type': 'SUV',
        'make_name': 'Toyota',
        'mileage': 80000,
        'horse_power': 150
    }
    
    try:
        # Initialize predictor
        predictor = CarPricePredictor()
        
        # Make prediction
        result = predictor.predict(sample_input)
        
        # Print results
        print("\nPrediction Results:")
        print(f"Predicted Price: KES {result['predicted_price'][0]:,.2f}")
        print(f"Prediction Time: {result['prediction_time']}")
        print(f"Model Version: {result['model_version']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 