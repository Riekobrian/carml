from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI app
app = FastAPI(
    title="Car Price Prediction API",
    description="API to predict car prices based on input features. Uses a pre-trained model and preprocessing pipeline.",
    version="1.2.0"
)

# Define paths
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Not strictly needed if using absolute paths for artifacts
# WORKSPACE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../')) # Not strictly needed

# MODIFIED: Point directly to your specific artifact folder
MODEL_ARTIFACTS_DIR = r"D:\SBT-JAPAN\Alpha GO\dataset-consolidation\src\deployment\model_artifacts\20250520_190019_stacking_model"
# MODEL_ARTIFACTS_DIR = r"D:\SBT-JAPAN\Alpha GO\dataset-consolidation\src\deployment\model_artifacts\stacking_model_cond_fix"

# Global variables
model = None
preprocessor = None
selected_feature_names = None # This will hold the list of 500 prefixed names
model_expected_feature_names = None # From model.feature_names_in_ if available

# Pydantic model for input features
class CarFeatures(BaseModel):
    """Input features for car price prediction, matching feature-engineered data columns."""
    make_name_cleaned: str = Field(..., description="Make of the car (e.g., toyota, honda) - cleaned")
    model_name_cleaned: str = Field(..., description="Model of the car - cleaned")
    mileage_num: float = Field(..., ge=0, description="Mileage in kilometers - numeric")
    engine_size_cc_num: float = Field(..., ge=0, description="Engine size in cc - numeric")
    fuel_type_cleaned: str = Field(..., description="Type of fuel (e.g., petrol, diesel) - cleaned")
    transmission_cleaned: str = Field(..., description="Transmission type (e.g., automatic, manual) - cleaned")
    horse_power_num: Optional[float] = Field(None, ge=0, description="Horsepower of the car - numeric")
    torque_num: Optional[float] = Field(None, ge=0, description="Torque of the car in Nm - numeric")
    annual_insurance: Optional[float] = Field(None, ge=0, description="Annual insurance cost (optional, will be imputed)")
    car_age: float = Field(..., ge=0, description="Age of the car in years")
    car_age_squared: float = Field(..., ge=0, description="Squared age of the car")
    mileage_log: float = Field(..., description="Log transformed mileage")
    mileage_per_year: float = Field(..., ge=0, description="Mileage per year")
    engine_size_cc_log: float = Field(..., description="Log transformed engine size")
    horse_power_log: Optional[float] = Field(None, description="Log transformed horsepower")
    torque_log: Optional[float] = Field(None, description="Log transformed torque")
    acceleration_num: Optional[float] = Field(None, ge=0, description="Acceleration (e.g., 0-100 km/h in seconds) - numeric")
    seats_num: Optional[float] = Field(None, ge=0, le=20, description="Number of seats - numeric")
    drive_type_cleaned: Optional[str] = Field(None, description="Drive type (e.g., fwd, rwd) - cleaned")
    condition_clean: Optional[str] = Field(None, description="Condition of the car - cleaned")
    usage_type_clean: Optional[str] = Field(None, description="Usage type of the car - cleaned")
    body_type_cleaned: Optional[str] = Field(None, description="Body type of the car - cleaned")
    power_per_cc: Optional[float] = Field(None, ge=0, description="Power per CC (HP/EngineSize)")
    mileage_per_cc: Optional[float] = Field(None, ge=0, description="Mileage per CC")
    is_luxury_make: Optional[int] = Field(None, ge=0, le=1, description="Flag for luxury make (0 or 1)")
    make_model_cleaned: Optional[str] = Field(None, description="Combined make and model - cleaned")

    class Config:
        anystr_strip_whitespace = True

# Load model and preprocessor
def load_artifacts():
    global model, preprocessor, selected_feature_names, model_expected_feature_names
    try:
        logging.info(f"Using MODEL_ARTIFACTS_DIR: {MODEL_ARTIFACTS_DIR}")
        if not os.path.isdir(MODEL_ARTIFACTS_DIR):
            logging.error(f"Model artifact directory not found: {MODEL_ARTIFACTS_DIR}")
            raise HTTPException(status_code=500, detail="Model artifacts directory not found.")

        # MODIFIED: Define artifact names directly
        preprocessor_artifact_name = "preprocessor.joblib"
        model_artifact_name = "model.joblib"
        selected_features_filename = "selected_feature_names_prefixed.txt" # Using the consistent name

        preprocessor_path = os.path.join(MODEL_ARTIFACTS_DIR, preprocessor_artifact_name)
        model_path = os.path.join(MODEL_ARTIFACTS_DIR, model_artifact_name)
        selected_features_path = os.path.join(MODEL_ARTIFACTS_DIR, selected_features_filename)
            
        if not os.path.exists(preprocessor_path):
            logging.error(f"Preprocessor artifact '{preprocessor_artifact_name}' not found in {MODEL_ARTIFACTS_DIR}.")
            raise HTTPException(status_code=500, detail="Preprocessor artifact not found.")
            
        if not os.path.exists(model_path):
            logging.error(f"Model file '{model_artifact_name}' not found in {MODEL_ARTIFACTS_DIR}")
            raise HTTPException(status_code=500, detail="Model file not found.")

        if not os.path.exists(selected_features_path):
            logging.error(f"Selected feature names file '{selected_features_filename}' not found in {MODEL_ARTIFACTS_DIR}")
            raise HTTPException(status_code=500, detail="Selected feature names file not found.")

        logging.info(f"Loading preprocessor from: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        logging.info("Preprocessor loaded successfully.")
        
        logging.info(f"Loading model from: {model_path}")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")

        # Try to get feature names from the model (e.g., first base estimator of StackingRegressor or model itself)
        try:
            # For StackingRegressor, check final_estimator_ or base estimators
            if hasattr(model, 'final_estimator_') and hasattr(model.final_estimator_, 'feature_names_in_'):
                model_expected_feature_names = list(model.final_estimator_.feature_names_in_)
                logging.info(f"Extracted {len(model_expected_feature_names)} feature names from StackingRegressor's final_estimator.")
            elif hasattr(model, 'estimators_') and model.estimators_ and hasattr(model.estimators_[0], 'feature_names_in_'):
                # This might give names for one of the base models, which might not be what the stacker's final_estimator sees if passthrough=True for base model outputs
                # However, if passthrough=False (our case), the final_estimator is trained on base model predictions,
                # and feature_names_in_ might reflect original features if the base models were fit on them.
                # For our StackingRegressor (passthrough=False), the .fit() is on X, y directly, so the stacker itself should have feature_names_in_
                 model_expected_feature_names = list(model.estimators_[0].feature_names_in_) # Fallback if StackingRegressor itself doesn't have it.
                 logging.info(f"Extracted {len(model_expected_feature_names)} feature names from model's first base estimator.")
            elif hasattr(model, 'feature_names_in_'): # If the model itself has it (e.g. a simple RF, or StackingRegressor in newer sklearn)
                 model_expected_feature_names = list(model.feature_names_in_)
                 logging.info(f"Extracted {len(model_expected_feature_names)} feature names from the model object itself (model.feature_names_in_).")
            else:
                logging.warning("Could not extract 'feature_names_in_' from the loaded model or its components using common attributes.")
            
            if model_expected_feature_names:
                 logging.debug(f"Sample of model_expected_feature_names: {model_expected_feature_names[:10]}...")
        except Exception as e:
            logging.warning(f"Error trying to extract feature names from model: {e}")
        
        logging.info(f"Loading selected feature names from: {selected_features_path}")
        with open(selected_features_path, 'r') as f:
            selected_feature_names = [line.strip() for line in f.readlines() if line.strip()]
        logging.info(f"Loaded {len(selected_feature_names)} selected feature names successfully from '{selected_features_filename}'.")
        # Example: We expect 500 features based on our last run
        if len(selected_feature_names) != 500: # Adjust 500 if your number of selected features is different
             logging.warning(f"Loaded {len(selected_feature_names)} selected features, but expected 500. Check '{selected_features_filename}'.")
        # if len(selected_feature_names) != 503: # Adjust 500 if your number of selected features is different
        #      logging.warning(f"Loaded {len(selected_feature_names)} selected features, but expected 503. Check '{selected_features_filename}'.")     

    except Exception as e:
        logging.error(f"Error loading artifacts: {e}", exc_info=True)
        # Ensure the API doesn't start successfully if essential artifacts are missing
        raise RuntimeError(f"Failed to load model/preprocessor artifacts: {e}")

# Preprocessing input data
def preprocess_input(car_features: CarFeatures, preprocessor_pipeline, final_selected_features: list) -> pd.DataFrame:
    try:
        input_feature_order = [
            'annual_insurance', 'car_age', 'car_age_squared', 'mileage_num', 'mileage_log', 
            'mileage_per_year', 'engine_size_cc_num', 'engine_size_cc_log', 'horse_power_num', 
            'horse_power_log', 'torque_num', 'torque_log', 'acceleration_num', 'seats_num', 
            'fuel_type_cleaned', 'transmission_cleaned', 'drive_type_cleaned', 'condition_clean', 
            'usage_type_clean', 'body_type_cleaned', 'make_name_cleaned', 'model_name_cleaned', 
            'power_per_cc', 'mileage_per_cc', 'is_luxury_make', 'make_model_cleaned'
        ]
        
        input_data_dict = car_features.model_dump(exclude_unset=True)
        df_dict = {col: np.nan for col in input_feature_order}
        for col in input_feature_order:
            if col in input_data_dict:
                df_dict[col] = input_data_dict[col]
        input_df = pd.DataFrame([df_dict], columns=input_feature_order)

        logging.debug(f"DataFrame before preprocessing (after initial dict population):\\n{input_df.to_string()}")

        # Imputation logic for optional features BEFORE they hit the preprocessor's IterativeImputer for 'annual_insurance'
        # This makes sure that features used to derive others (like horse_power_num for power_per_cc) have values.
        placeholder_medians = {
            'horse_power_num': 100.0, 'torque_num': 150.0,
            'acceleration_num': 12.0, 'seats_num': 5.0,
            # 'annual_insurance' will be handled by IterativeImputer in the preprocessor
        }
        
        numeric_cols_to_check_for_imputation = [
            'horse_power_num', 'torque_num', 'acceleration_num', 'seats_num',
            # 'annual_insurance' is handled by IterativeImputer
            # Log versions will be derived
            # Derived versions like power_per_cc will be recalculated
        ]

        for col in numeric_cols_to_check_for_imputation:
            if col in input_df.columns and input_df.loc[0, col] is None: # Check for None explicitly
                fill_value = placeholder_medians.get(col)
                if fill_value is not None:
                    input_df.loc[0, col] = fill_value
                    logging.info(f"Imputed missing '{col}' with placeholder median: {fill_value}")
                else: # Should not happen if placeholder_medians is comprehensive for these cols
                    logging.warning(f"Missing value for '{col}' and no placeholder median defined. It might remain NaN if not handled by preprocessor.")
        
        # Recalculate log/derived features based on potentially newly imputed values
        # Ensure consistency for log-transformed features
        current_hp = input_df.loc[0, 'horse_power_num']
        if pd.isnull(input_df.loc[0, 'horse_power_log']) and not pd.isnull(current_hp):
            input_df.loc[0, 'horse_power_log'] = np.log1p(current_hp)
            logging.info(f"Recalculated 'horse_power_log' based on 'horse_power_num' ({current_hp}) to {input_df.loc[0, 'horse_power_log']:.4f}")

        current_torque = input_df.loc[0, 'torque_num']
        if pd.isnull(input_df.loc[0, 'torque_log']) and not pd.isnull(current_torque):
            input_df.loc[0, 'torque_log'] = np.log1p(current_torque)
            logging.info(f"Recalculated 'torque_log' based on 'torque_num' ({current_torque}) to {input_df.loc[0, 'torque_log']:.4f}")

        # Ensure consistency for derived features like 'power_per_cc'
        hp_val = input_df.loc[0, 'horse_power_num']
        cc_val = input_df.loc[0, 'engine_size_cc_num'] # Mandatory, should not be null
        
        # Recalculate power_per_cc if hp_val exists
        if not pd.isnull(hp_val) and not pd.isnull(cc_val):
            if cc_val != 0:
                input_df.loc[0, 'power_per_cc'] = hp_val / cc_val
                logging.info(f"Recalculated 'power_per_cc' using hp_val={hp_val}, cc_val={cc_val} to {input_df.loc[0, 'power_per_cc']:.4f}")
            else: # Avoid division by zero
                input_df.loc[0, 'power_per_cc'] = 0.0
                logging.info(f"Set 'power_per_cc' to 0.0 due to cc_val being 0.")
        elif pd.isnull(hp_val): # If HP is still null after imputation (shouldn't be for our defined list)
             input_df.loc[0, 'power_per_cc'] = None # or 0.0 depending on how preprocessor handles it
             logging.info(f"Set 'power_per_cc' to None/0.0 as hp_val is None.")

        # mileage_per_cc
        mileage_val = input_df.loc[0, 'mileage_num'] # Mandatory
        if not pd.isnull(mileage_val) and not pd.isnull(cc_val):
            if cc_val != 0:
                input_df.loc[0, 'mileage_per_cc'] = mileage_val / cc_val
                logging.info(f"Recalculated 'mileage_per_cc' using mileage_val={mileage_val}, cc_val={cc_val} to {input_df.loc[0, 'mileage_per_cc']:.4f}")
            else: # Avoid division by zero
                input_df.loc[0, 'mileage_per_cc'] = 0.0
                logging.info(f"Set 'mileage_per_cc' to 0.0 due to cc_val being 0.")
        
        logging.debug(f"DataFrame after all NaN imputation and recalculations before preprocessor.transform:\\n{input_df.to_string()}")
        logging.debug(f"Data types:\\n{input_df.dtypes}")

        processed_data_np = preprocessor_pipeline.transform(input_df)
        
        try:
            # For ColumnTransformer, get_feature_names_out is the way
            # These are the ~999 prefixed names
            transformed_column_names_from_preprocessor = list(preprocessor_pipeline.get_feature_names_out())
        except AttributeError:
            logging.error("preprocessor.get_feature_names_out() not available. Cannot determine preprocessor output columns.", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal error: Preprocessor misconfiguration.")

        logging.info(f"Number of features after preprocessing (from preprocessor.get_feature_names_out()): {len(transformed_column_names_from_preprocessor)}")
        logging.debug(f"First 10 transformed_column_names_from_preprocessor: {transformed_column_names_from_preprocessor[:10]}")

        processed_df_full = pd.DataFrame(processed_data_np, columns=transformed_column_names_from_preprocessor, index=input_df.index)
        
        # --- DETAILED LOGGING FOR FEATURE NAME COMPARISON ---
        logging.info("--- DETAILED FEATURE NAME AND TYPE LOGGING (API preprocess_input) ---")
        logging.info(f"Number of columns from preprocessor output (transformed_column_names_from_preprocessor): {len(transformed_column_names_from_preprocessor)}")
        logging.debug(f"ALL preprocessor output names ({len(transformed_column_names_from_preprocessor)} total, sorted): {sorted(transformed_column_names_from_preprocessor)}") 
        
        logging.info(f"Number of features from selected_feature_names.txt (final_selected_features): {len(final_selected_features)}")
        logging.debug(f"ALL selected_feature_names from file ({len(final_selected_features)} total, sorted): {sorted(list(final_selected_features))}")
        # --- END DETAILED LOGGING --- 

        set_processed_columns = set(transformed_column_names_from_preprocessor)
        set_final_selected_features = set(final_selected_features)

        # Check if all selected features are present in the preprocessor's output
        missing_in_processed_output = set_final_selected_features - set_processed_columns
        if missing_in_processed_output:
            logging.error(
                f"CRITICAL ERROR: {len(missing_in_processed_output)} selected features are MISSING from the preprocessor's direct output. "
                f"This should not happen if the preprocessor and selected features list are consistent. "
                f"Missing features (sample): {list(missing_in_processed_output)[:20]}"
            )
            # This is a fatal error for prediction consistency
            raise HTTPException(status_code=500, detail=f"Internal error: Feature mismatch. Selected features not found in preprocessed output. Missing: {list(missing_in_processed_output)[:5]}")
        
        logging.info(f"All {len(final_selected_features)} selected features are present in the preprocessor's output of {len(transformed_column_names_from_preprocessor)} features.")
        logging.info(f"Filtering {processed_df_full.shape[1]} preprocessed columns down to {len(final_selected_features)} selected features.")
        
        # Now, select ONLY the features listed in final_selected_features, and in that specific order.
        try:
            final_processed_df = processed_df_full[final_selected_features].copy()
        except KeyError as ke:
            logging.error(f"KeyError when selecting final features: {ke}. This means some selected features were not found in the processed output.", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Internal error: Feature mismatch during final selection. Missing: {ke}")

        logging.info(f"Shape of DataFrame after selecting features: {final_processed_df.shape}")
        if final_processed_df.shape[1] != len(final_selected_features):
            logging.error(f"Column count mismatch! Expected {len(final_selected_features)} selected features, but final_processed_df has {final_processed_df.shape[1]}.")
            raise HTTPException(status_code=500, detail="Internal error: Column count mismatch after selection.")
            
        logging.debug(f"Final selected DataFrame for model (first 5 rows, few columns):\\n{final_processed_df.head().iloc[:, :5].to_string()}")

        return final_processed_df

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logging.error(f"Error during preprocessing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error preprocessing input: {e}")

@app.on_event("startup")
async def startup_event():
    logging.info("Application startup: Loading model and preprocessor...")
    try:
        load_artifacts() # This can raise RuntimeError if loading fails
        if model is None or preprocessor is None or selected_feature_names is None:
            logging.error("CRITICAL: Model, preprocessor, or selected features not loaded after load_artifacts call. API may not function correctly.")
            # This state should ideally be prevented by load_artifacts raising an error.
            raise RuntimeError("Essential artifacts not loaded.")
        logging.info("Application startup complete. Model and preprocessor loaded.")
    except RuntimeError as e:
        logging.critical(f"Application startup failed: {e}", exc_info=True)
        # FastAPI doesn't have a clean way to prevent startup on error in event handler,
        # but we ensure the state reflects failure. The server might still start but endpoints will fail.
        # For production, consider a pre-run check script or more complex startup management.
        # For now, subsequent requests will fail due to missing model/preprocessor.
        # Re-raising to make it very clear in logs if possible.
        raise 

@app.post("/predict/", summary="Predict Car Price", response_description="The predicted price of the car")
async def predict_price(car_features: CarFeatures) -> Dict[str, float]:
    global model, preprocessor, selected_feature_names, model_expected_feature_names # model_expected_feature_names is for logging
    if model is None or preprocessor is None or selected_feature_names is None:
        logging.error("Model, preprocessor, or selected_feature_names not loaded at request time.")
        raise HTTPException(status_code=503, detail="Model, preprocessor or feature list not ready. Please try again shortly.")
    try:
        logging.info(f"Received prediction request with features (first few): make={car_features.make_name_cleaned}, model={car_features.model_name_cleaned}, mileage={car_features.mileage_num}")
        
        processed_df = preprocess_input(car_features, preprocessor, selected_feature_names)

        # --- DETAILED FEATURE NAME DEBUGGING (before model.predict) ---
        final_df_columns = list(processed_df.columns)
        logging.info(f"Number of columns in DataFrame being sent to model.predict(): {len(final_df_columns)}")
        logging.debug(f"Sample of columns in DataFrame for model.predict() (first 10): {final_df_columns[:10]}...")
        logging.debug(f"Sample of columns in DataFrame for model.predict() (last 10): {final_df_columns[-10:] if len(final_df_columns) > 10 else final_df_columns}...")

        if model_expected_feature_names and len(model_expected_feature_names) > 0 : # Ensure it's not None or empty
            logging.info(f"Number of features model was trained on (from model.feature_names_in_ or similar): {len(model_expected_feature_names)}")
            if len(final_df_columns) == len(model_expected_feature_names):
                logging.info("Counts match between DataFrame columns and model's expected features.")
                if set(final_df_columns) == set(model_expected_feature_names):
                    logging.info("Sets of feature names also match.")
                    mismatch_due_to_order_or_type = False
                    for i in range(len(final_df_columns)):
                        if final_df_columns[i] != model_expected_feature_names[i]:
                            logging.error(f"ORDER MISMATCH at index {i}: API has '{final_df_columns[i]}', Model expected '{model_expected_feature_names[i]}'")
                            mismatch_due_to_order_or_type = True
                            # break # Optional: stop at first mismatch for brevity
                    if not mismatch_due_to_order_or_type:
                        logging.info("Feature names and order appear to EXACTLY match those from model's feature_names_in_.")
                    else:
                        logging.error("ORDER MISMATCH DETECTED between API DataFrame and model's expected features.")
                else: # Sets do not match
                    logging.error("CRITICAL: Feature name SETS DO NOT MATCH. This is the primary issue.")
                    unseen_by_model = set(final_df_columns) - set(model_expected_feature_names)
                    missing_from_df = set(model_expected_feature_names) - set(final_df_columns)
                    if unseen_by_model:
                        logging.error(f"Features in DF but NOT expected by model ({len(unseen_by_model)}): {sorted(list(unseen_by_model))[:20]}...")
                    if missing_from_df:
                        logging.error(f"Features expected by model but NOT in DF ({len(missing_from_df)}): {sorted(list(missing_from_df))[:20]}...")
            else: # Counts do not match
                logging.error(f"CRITICAL COUNT MISMATCH: DataFrame has {len(final_df_columns)} cols, model expected {len(model_expected_feature_names)} based on feature_names_in_.")
        else:
            logging.warning("Could not compare with model_expected_feature_names as it was not available or empty. Model will receive {len(final_df_columns)} features.")
        # --- END DETAILED FEATURE NAME DEBUGGING ---

        # --- BEGIN ADDED NaN DEBUGGING ---
        nan_in_processed_df = processed_df.isnull().sum()
        columns_with_nan = nan_in_processed_df[nan_in_processed_df > 0]
        if not columns_with_nan.empty:
            logging.warning(f"NaNs detected in DataFrame sent to model.predict(). Columns with NaNs and their counts:\\n{columns_with_nan.to_string()}")
            # Potentially raise an error or handle, depending on model robustness to NaNs at this stage
            # Most scikit-learn models will error if they receive NaNs unless they specifically handle them.
        else:
            logging.info("No NaNs detected in DataFrame sent to model.predict().")
        # --- END ADDED NaN DEBUGGING ---

        price_log_prediction = model.predict(processed_df)
        predicted_price = np.exp(price_log_prediction[0])

        logging.info(f"Log-transformed prediction: {price_log_prediction[0]:.4f}, Actual predicted price: KES {predicted_price:,.2f}")
        
        return {"predicted_price": predicted_price}

    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

@app.get("/health/", summary="Health Check", response_description="API Health Status")
async def health_check():
    if model is not None and preprocessor is not None and selected_feature_names is not None:
        return {"status": "healthy", "message": "API is running and artifacts are loaded."}
    else:
        missing = []
        if model is None: missing.append("model")
        if preprocessor is None: missing.append("preprocessor")
        if selected_feature_names is None: missing.append("selected_feature_names")
        logging.warning(f"Health check: API is running but artifacts ({', '.join(missing)}) are not loaded.")
        return {"status": "unhealthy", "message": f"API is running but essential artifacts ({', '.join(missing)}) are not loaded."}

if __name__ == "__main__":
    import uvicorn
    # Ensure artifacts are loaded before starting server, or handle failure gracefully
    # A pre-run check or a more robust startup sequence might be needed for production.
    try:
        load_artifacts() # Load them on module execution when running directly
        logging.info("Artifacts loaded successfully via __main__ block.")
    except Exception as e:
        logging.critical(f"Failed to load artifacts in __main__ before starting Uvicorn: {e}", exc_info=True)
        # Decide if you want to exit or let Uvicorn start and fail on first request
        # For development, it might be okay to let it start. For production, better to exit.
        # exit(1) # Uncomment to exit if artifacts don't load

    uvicorn.run(app, host="0.0.0.0", port=8000) 