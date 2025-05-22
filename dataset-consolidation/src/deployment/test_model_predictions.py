import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime

# API endpoint
API_URL = "http://127.0.0.1:8000/predict/" # Ensure trailing slash

# Helper to calculate some date/log dependent features
def calculate_derived_features(year_of_manufacture, mileage, engine_size_cc, horse_power=None, torque=None, acceleration=None, seats=None):
    current_year = datetime.now().year # Use a consistent current year for test runs if needed, or live
    car_age = float(current_year - year_of_manufacture)
    car_age_squared = car_age ** 2.0
    
    # Ensure inputs are float for numpy functions
    mileage = float(mileage) if mileage is not None else 0.0
    engine_size_cc = float(engine_size_cc) if engine_size_cc is not None else 0.0
    horse_power = float(horse_power) if horse_power is not None else None
    torque = float(torque) if torque is not None else None
    # acceleration and seats are handled directly if present, or passed as None

    mileage_log = np.log1p(mileage)
    mileage_per_year = mileage / (car_age + 1e-6) # Add epsilon for new cars
    engine_size_cc_log = np.log1p(engine_size_cc) if engine_size_cc > 0 else 0.0 # handle 0cc
    
    hp_log = np.log1p(horse_power) if horse_power is not None and horse_power > 0 else None
    tq_log = np.log1p(torque) if torque is not None and torque > 0 else None
    
    power_per_cc_val = (horse_power / (engine_size_cc + 1e-6)) if horse_power is not None and engine_size_cc > 0 else None
    mileage_per_cc_val = (mileage / (engine_size_cc + 1e-6)) if engine_size_cc > 0 else None # mileage can be 0 for new car
        
    # These features are directly taken from the CarFeatures Pydantic model
    # and should be provided if they were in the feature-engineered data for that record.
    # The API's Pydantic model and preprocess_input will handle Nones for optional fields.
    
    return {
        "car_age": car_age,
        "car_age_squared": car_age_squared,
        "mileage_log": mileage_log,
        "mileage_per_year": mileage_per_year,
        "engine_size_cc_log": engine_size_cc_log,
        "horse_power_log": hp_log, # Can be None if horse_power is None
        "torque_log": tq_log,       # Can be None if torque is None
        "power_per_cc": power_per_cc_val, # Can be None
        "mileage_per_cc": mileage_per_cc_val # Can be None
        # acceleration_num and seats_num are passed directly in base
    }

# --- Test Case 1: Toyota Crown 2011 (from feature-engineered dataset) ---
# Actual price_log: 13.892470673174047 -> price ~ 1,079,998
tc1_actual_price = np.exp(13.892470673174047)
tc1_base_from_fe = {
    "make_name_cleaned": "toyota",
    "model_name_cleaned": "crown",
    "year_of_manufacture": 2011, # car_age will be 14 if test run in 2025, matches FE data.
    "mileage_num": 108452.0,
    "engine_size_cc_num": 2490.0,
    "fuel_type_cleaned": "petrol",
    "transmission_cleaned": "automatic",
    "horse_power_num": 203.0,
    "torque_num": 250.0,
    "annual_insurance": 43200.0,
    "acceleration_num": 9.0,
    "seats_num": 5.0,
    "drive_type_cleaned": "2wd", # From FE data
    "condition_clean": "Accident involved", # From FE data
    "usage_type_clean": "Kenyan Used", # From FE data
    "body_type_cleaned": "sedan",
    "is_luxury_make": 0,
    "make_model_cleaned": "toyota_crown"
    # Derived features (car_age, *_log, *_per_year, *_per_cc) will be calculated by helper
}
tc1_derived = calculate_derived_features(
    tc1_base_from_fe["year_of_manufacture"], tc1_base_from_fe["mileage_num"], tc1_base_from_fe["engine_size_cc_num"],
    tc1_base_from_fe["horse_power_num"], tc1_base_from_fe["torque_num"]
)
# The Pydantic model expects all derived fields, even if some are None from calculate_derived_features
tc1_input = {
    **{k: v for k, v in tc1_base_from_fe.items() if k != 'year_of_manufacture'}, 
    **tc1_derived
}


# --- Test Case 2: Toyota Prado 2024 (from feature-engineered dataset) ---
# Actual price_log: 16.70588231586044 -> price ~ 17,999,999
tc2_actual_price = np.exp(16.70588231586044)
tc2_base_from_fe = {
    "make_name_cleaned": "toyota",
    "model_name_cleaned": "prado",
    "year_of_manufacture": 2024, # car_age will be 1 if test run in 2025
    "mileage_num": 35.0,
    "engine_size_cc_num": 2800.0,
    "fuel_type_cleaned": "diesel",
    "transmission_cleaned": "automatic",
    "horse_power_num": 201.0,
    "torque_num": 500.0,
    "annual_insurance": 720000.0,
    "acceleration_num": 10.0,
    "seats_num": 5.0,
    "drive_type_cleaned": "4wd", # From FE data
    "condition_clean": "Accident free", # From FE data (was 'Excellent' in raw, mapped to 'Accident free' in FE)
    "usage_type_clean": "Foreign Used", # From FE data (was 'Brand New' in raw)
    "body_type_cleaned": "suv",
    "is_luxury_make": 0,
    "make_model_cleaned": "toyota_prado"
}
tc2_derived = calculate_derived_features(
    tc2_base_from_fe["year_of_manufacture"], tc2_base_from_fe["mileage_num"], tc2_base_from_fe["engine_size_cc_num"],
    tc2_base_from_fe["horse_power_num"], tc2_base_from_fe["torque_num"]
)
tc2_input = {
    **{k: v for k, v in tc2_base_from_fe.items() if k != 'year_of_manufacture'}, 
    **tc2_derived
}


# --- Test Case 3: BMW X3 2004 (from feature-engineered dataset) ---
# Actual price_log: 13.25339339419516 -> price ~ 570,000
tc3_actual_price = np.exp(13.25339339419516)
tc3_base_from_fe = {
    "make_name_cleaned": "bmw",
    "model_name_cleaned": "x3",
    "year_of_manufacture": 2004, # car_age will be 21 if test run in 2025
    "mileage_num": 145000.0,
    "engine_size_cc_num": 3000.0,
    "fuel_type_cleaned": "petrol",
    "transmission_cleaned": "automatic",
    "horse_power_num": 225.0,
    "torque_num": 290.0,
    "annual_insurance": 30000.0,
    "acceleration_num": 8.1,
    "seats_num": 5.0,
    "drive_type_cleaned": "4wd", # From FE data
    "condition_clean": "Accident involved", # From FE data
    "usage_type_clean": "Kenyan Used", # From FE data
    "body_type_cleaned": "suv",
    "is_luxury_make": 1,
    "make_model_cleaned": "bmw_x3"
}
tc3_derived = calculate_derived_features(
    tc3_base_from_fe["year_of_manufacture"], tc3_base_from_fe["mileage_num"], tc3_base_from_fe["engine_size_cc_num"],
    tc3_base_from_fe["horse_power_num"], tc3_base_from_fe["torque_num"]
)
tc3_input = {
    **{k: v for k, v in tc3_base_from_fe.items() if k != 'year_of_manufacture'}, 
    **tc3_derived
}

# --- Test Case 4: Generic Toyota Corolla (Original Case 1, adjusted expectations) ---
# This case remains to test a common scenario not directly from the FE snippet
generic_toyota_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "corolla",
    "year_of_manufacture": 2018, # For car_age calculation
    "mileage_num": 60000.0, "engine_size_cc_num": 1800.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 139.0, "torque_num": 171.0,
    "annual_insurance": 50000.0, "acceleration_num": 10.5,
    "seats_num": 5.0, "drive_type_cleaned": "fwd",
    # For condition/usage, ensure these are valid categories the model was trained on.
    # If "Accident free" and "Foreign Used" are standard from your FE process, use those.
    "condition_clean": "Accident free", # Example: Assuming this is a standard clean category
    "usage_type_clean": "Foreign Used",   # Example: Assuming imported used
    "body_type_cleaned": "sedan", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_corolla"
}
generic_toyota_derived = calculate_derived_features(
    generic_toyota_base["year_of_manufacture"], generic_toyota_base["mileage_num"], 
    generic_toyota_base["engine_size_cc_num"], generic_toyota_base["horse_power_num"], 
    generic_toyota_base["torque_num"]
)
generic_toyota_input = {
    **{k:v for k,v in generic_toyota_base.items() if k != 'year_of_manufacture'}, 
    **generic_toyota_derived
}


TEST_CASES = [
    {
        "description": "Toyota Crown 2011 (from FE dataset)",
        "input": tc1_input,
        "expected_price_range": (tc1_actual_price * 0.80, tc1_actual_price * 1.20) # +/- 20%
    },
    {
        "description": "Toyota Prado 2024, Brand New (from FE dataset)",
        "input": tc2_input,
        "expected_price_range": (tc2_actual_price * 0.80, tc2_actual_price * 1.20) # +/- 20%
    },
    {
        "description": "BMW X3 2004, Accident Involved (from FE dataset)",
        "input": tc3_input,
        "expected_price_range": (tc3_actual_price * 0.75, tc3_actual_price * 1.25) # +/- 25% for more variance
    },
    {
        "description": "Generic Test: Common Toyota Sedan (Original Case 1 adjusted)",
        "input": generic_toyota_input,
        "expected_price_range": (800000, 1200000) 
    }
]

def run_tests():
    results = []
    all_passed = True
    for i, case in enumerate(TEST_CASES):
        print(f"--- Running Test Case {i+1}: {case['description']} ---")
        
        try:
            response = requests.post(API_URL, json=case["input"])
            response.raise_for_status()
            
            response_data = response.json()
            predicted_price = response_data.get("predicted_price")
            
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response_data}")
            
            if predicted_price is not None:
                print(f"Predicted Price: KES {predicted_price:,.0f}")
                min_expected, max_expected = case.get("expected_price_range", (None, None))
                
                if min_expected is not None and max_expected is not None:
                    if min_expected <= predicted_price <= max_expected:
                        print(f"Result: PASSED (Expected KES {min_expected:,.0f} - {max_expected:,.0f})\\n")
                        results.append({"description": case["description"], "status": "PASSED", "predicted_price": predicted_price, "details": ""})
                    else:
                        all_passed = False
                        error_detail = f"FAILED: Predicted price KES {predicted_price:,.0f} is outside the expected range KES {min_expected:,.0f} - {max_expected:,.0f}."
                        print(f"Result: {error_detail}\\n")
                        results.append({"description": case["description"], "status": "FAILED", "predicted_price": predicted_price, "details": error_detail})
                else:
                    print("Result: CHECK MANUALLY (No expected price range defined)\\n")
                    results.append({"description": case["description"], "status": "CHECK MANUALLY", "predicted_price": predicted_price, "details": ""})
            else:
                all_passed = False
                error_detail = "FAILED: 'predicted_price' not found in response."
                print(f"Result: {error_detail}\\n")
                results.append({"description": case["description"], "status": "FAILED", "predicted_price": None, "details": error_detail})

        except requests.exceptions.HTTPError as http_err:
            all_passed = False
            error_detail = f"FAILED: HTTP error occurred: {http_err}. Response: {http_err.response.text}"
            print(f"{error_detail}\\n")
            results.append({"description": case["description"], "status": "ERROR", "predicted_price": None, "details": error_detail})
        except requests.exceptions.RequestException as req_err:
            all_passed = False
            error_detail = f"FAILED: Request exception occurred: {req_err}"
            print(f"{error_detail}\\n")
            results.append({"description": case["description"], "status": "ERROR", "predicted_price": None, "details": error_detail})
        except Exception as e:
            all_passed = False
            error_detail = f"FAILED: An unexpected error occurred: {e}"
            print(f"{error_detail}\\n")
            results.append({"description": case["description"], "status": "ERROR", "predicted_price": None, "details": str(e)})

    print("\\n--- Test Summary ---")
    summary_df = pd.DataFrame(results)
    print(summary_df.to_string(index=False))
    
    if all_passed and results:
        print("\\nOverall Result: ALL TESTS PASSED!")
    elif results:
        print("\\nOverall Result: SOME TESTS FAILED OR ERRORED.")
    else:
        print("\\nOverall Result: NO TESTS WERE RUN.")

if __name__ == "__main__":
    run_tests() 