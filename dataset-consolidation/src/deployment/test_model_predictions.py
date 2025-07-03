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

# --- New Custom Test Cases ---

# 1. Custom Toyota Axio 2016 (Foreign Used, 80k km, Ref 1.85M)
axio_2016_ref_price = 1850000.0
axio_2016_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "axio",
    "year_of_manufacture": 2016,
    "mileage_num": 80000.0, "engine_size_cc_num": 1500.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 109.0, "torque_num": 138.0,
    "annual_insurance": 45000.0, "acceleration_num": 11.5,
    "seats_num": 5.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used",
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "sedan", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_axio"
}
axio_2016_derived = calculate_derived_features(
    axio_2016_base["year_of_manufacture"], axio_2016_base["mileage_num"], 
    axio_2016_base["engine_size_cc_num"], axio_2016_base["horse_power_num"], 
    axio_2016_base["torque_num"]
)
axio_2016_input = {
    **{k:v for k,v in axio_2016_base.items() if k != 'year_of_manufacture'}, 
    **axio_2016_derived
}
TEST_CASES.append({
    "description": "Custom Toyota Axio 2016 (Foreign Used, 80k km, Ref 1.85M)",
    "input": axio_2016_input,
    "expected_price_range": (axio_2016_ref_price * 0.80, axio_2016_ref_price * 1.20)
})

# 2. Custom 2017 Axio Hybrid (99k km, Ref 1.69M)
axio_hybrid_ref_price = 1690000.0
axio_hybrid_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "axio",
    "year_of_manufacture": 2017,
    "mileage_num": 99000.0, "engine_size_cc_num": 1500.0,
    "fuel_type_cleaned": "hybrid_petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 100.0, "torque_num": 140.0,
    "annual_insurance": 48000.0, "acceleration_num": 12.0,
    "seats_num": 5.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", 
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "sedan", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_axio"
}
axio_hybrid_derived = calculate_derived_features(
    axio_hybrid_base["year_of_manufacture"], axio_hybrid_base["mileage_num"], 
    axio_hybrid_base["engine_size_cc_num"], axio_hybrid_base["horse_power_num"], 
    axio_hybrid_base["torque_num"]
)
axio_hybrid_input = {
    **{k:v for k,v in axio_hybrid_base.items() if k != 'year_of_manufacture'}, 
    **axio_hybrid_derived
}
TEST_CASES.append({
    "description": "Custom 2017 Axio Hybrid (99k km, Ref 1.69M)",
    "input": axio_hybrid_input,
    "expected_price_range": (axio_hybrid_ref_price * 0.80, axio_hybrid_ref_price * 1.20)
})

# 3. Custom 2015 High-Perf SUV (Assumed BMW X5, 123k km, Ref 4.5M)
hp_suv_ref_price = 4500000.0
hp_suv_base = {
    "make_name_cleaned": "bmw", "model_name_cleaned": "x5",
    "year_of_manufacture": 2015,
    "mileage_num": 122932.0, "engine_size_cc_num": 4400.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 444.0, "torque_num": 650.0,
    "annual_insurance": 180000.0, "acceleration_num": 4.7,
    "seats_num": 5.0, "drive_type_cleaned": "4wd",
    "condition_clean": "used", 
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "bmw_x5"
}
hp_suv_derived = calculate_derived_features(
    hp_suv_base["year_of_manufacture"], hp_suv_base["mileage_num"], 
    hp_suv_base["engine_size_cc_num"], hp_suv_base["horse_power_num"], 
    hp_suv_base["torque_num"]
)
hp_suv_input = {
    **{k:v for k,v in hp_suv_base.items() if k != 'year_of_manufacture'}, 
    **hp_suv_derived
}
TEST_CASES.append({
    "description": "Custom 2015 High-Perf SUV (BMW X5, 123k km, Ref 4.5M)",
    "input": hp_suv_input,
    "expected_price_range": (hp_suv_ref_price * 0.80, hp_suv_ref_price * 1.20)
})

# 4. Custom 2017 Volvo XC60 (80k km, Ref 3.6M)
volvo_xc60_ref_price = 3600000.0
volvo_xc60_base = {
    "make_name_cleaned": "volvo", "model_name_cleaned": "xc60",
    "year_of_manufacture": 2017,
    "mileage_num": 80000.0, "engine_size_cc_num": 1969.0,
    "fuel_type_cleaned": "diesel", "transmission_cleaned": "automatic",
    "horse_power_num": 190.0, "torque_num": 400.0,
    "annual_insurance": 120000.0, "acceleration_num": 8.5,
    "seats_num": 5.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", 
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "volvo_xc60"
}
volvo_xc60_derived = calculate_derived_features(
    volvo_xc60_base["year_of_manufacture"], volvo_xc60_base["mileage_num"], 
    volvo_xc60_base["engine_size_cc_num"], volvo_xc60_base["horse_power_num"], 
    volvo_xc60_base["torque_num"]
)
volvo_xc60_input = {
    **{k:v for k,v in volvo_xc60_base.items() if k != 'year_of_manufacture'}, 
    **volvo_xc60_derived
}
TEST_CASES.append({
    "description": "Custom 2017 Volvo XC60 (80k km, Ref 3.6M)",
    "input": volvo_xc60_input,
    "expected_price_range": (volvo_xc60_ref_price * 0.80, volvo_xc60_ref_price * 1.20)
})

# 5. Custom 2017 Toyota RAV4 AWD (150k km, Ref 2.7M)
rav4_awd_ref_price = 2700000.0
rav4_awd_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "rav4",
    "year_of_manufacture": 2017,
    "mileage_num": 150000.0, "engine_size_cc_num": 2500.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 176.0, "torque_num": 233.0,
    "annual_insurance": 108000.0, "acceleration_num": 9.5,
    "seats_num": 5.0, "drive_type_cleaned": "awd",
    "condition_clean": "used",
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_rav4"
}
rav4_awd_derived = calculate_derived_features(
    rav4_awd_base["year_of_manufacture"], rav4_awd_base["mileage_num"], 
    rav4_awd_base["engine_size_cc_num"], rav4_awd_base["horse_power_num"], 
    rav4_awd_base["torque_num"]
)
rav4_awd_input = {
    **{k:v for k,v in rav4_awd_base.items() if k != 'year_of_manufacture'}, 
    **rav4_awd_derived
}
TEST_CASES.append({
    "description": "Custom 2017 Toyota RAV4 AWD (150k km, Ref 2.7M)",
    "input": rav4_awd_input,
    "expected_price_range": (rav4_awd_ref_price * 0.80, rav4_awd_ref_price * 1.20)
})

# 6. Custom 2018 Toyota Passo 2WD (89k km, Ref 1.05M)
passo_2wd_ref_price = 1050000.0
passo_2wd_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "passo",
    "year_of_manufacture": 2018,
    "mileage_num": 89000.0, "engine_size_cc_num": 1200.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 85.0, "torque_num": 115.0,
    "annual_insurance": 52500.0, "acceleration_num": 13.0,
    "seats_num": 5.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used",
    "usage_type_clean": "Kenyan Used",
    "body_type_cleaned": "hatchback", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_passo"
}
passo_2wd_derived = calculate_derived_features(
    passo_2wd_base["year_of_manufacture"], passo_2wd_base["mileage_num"], 
    passo_2wd_base["engine_size_cc_num"], passo_2wd_base["horse_power_num"], 
    passo_2wd_base["torque_num"]
)
passo_2wd_input = {
    **{k:v for k,v in passo_2wd_base.items() if k != 'year_of_manufacture'}, 
    **passo_2wd_derived
}
TEST_CASES.append({
    "description": "Custom 2018 Toyota Passo 2WD (89k km, Ref 1.05M)",
    "input": passo_2wd_input,
    "expected_price_range": (passo_2wd_ref_price * 0.80, passo_2wd_ref_price * 1.20)
})

# 7. Custom 2017 Land Rover SUV Diesel (Assumed Discovery Sport, 64k km, Ref 6.9M)
lr_suv_ref_price = 6900000.0
lr_suv_base = {
    "make_name_cleaned": "land rover", "model_name_cleaned": "other_model",
    "year_of_manufacture": 2017,
    "mileage_num": 64000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "diesel", "transmission_cleaned": "automatic",
    "horse_power_num": 180.0, "torque_num": 430.0,
    "annual_insurance": 241500.0, "acceleration_num": 9.0,
    "seats_num": 5.0, "drive_type_cleaned": "4wd",
    "condition_clean": "used",
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "land rover_other_model"
}
lr_suv_derived = calculate_derived_features(
    lr_suv_base["year_of_manufacture"], lr_suv_base["mileage_num"], 
    lr_suv_base["engine_size_cc_num"], lr_suv_base["horse_power_num"], 
    lr_suv_base["torque_num"]
)
lr_suv_input = {
    **{k:v for k,v in lr_suv_base.items() if k != 'year_of_manufacture'}, 
    **lr_suv_derived
}
TEST_CASES.append({
    "description": "Custom 2017 Land Rover SUV Diesel (64k km, Ref 6.9M)",
    "input": lr_suv_input,
    "expected_price_range": (lr_suv_ref_price * 0.80, lr_suv_ref_price * 1.20)
})

# --- End New Custom Test Cases ---

# --- Test Cases from Training Data Snippet ---
# Assuming the CSV's car_age was calculated relative to current_year = 2025 for consistency
YEAR_CSV_GENERATED = 2025 

# Training Data Case 1: Mercedes E250 (Row 2 of snippet)
td1_actual_price_api = np.exp(15.424948470398375)
td1_base_api = {
    "make_name_cleaned": "mercedes", "model_name_cleaned": "e250",
    "year_of_manufacture": YEAR_CSV_GENERATED - 9, # car_age = 9
    "mileage_num": 109000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 211.0, "torque_num": 350.0,
    "annual_insurance": 208000.0, "acceleration_num": 7.6,
    "seats_num": 5.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", # Mapping 'Accident free' from CSV
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "sedan", "is_luxury_make": 1,
    "make_model_cleaned": "mercedes_e250"
}
td1_derived_api = calculate_derived_features(
    td1_base_api["year_of_manufacture"], td1_base_api["mileage_num"], 
    td1_base_api["engine_size_cc_num"], td1_base_api["horse_power_num"], 
    td1_base_api["torque_num"]
)
td1_input_api = {
    **{k:v for k,v in td1_base_api.items() if k != 'year_of_manufacture'}, 
    **td1_derived_api
}
TEST_CASES.append({
    "description": "Training Data API: Mercedes E250 (car_age 9, Ref ~5.0M)",
    "input": td1_input_api,
    "expected_price_range": (td1_actual_price_api * 0.95, td1_actual_price_api * 1.05)
})

# Training Data Case 2: Subaru Forester XT (Row 4 of snippet)
td2_actual_price_api = np.exp(13.527828485512494)
td2_base_api = {
    "make_name_cleaned": "subaru", "model_name_cleaned": "forester xt",
    "year_of_manufacture": YEAR_CSV_GENERATED - 22, # car_age = 22
    "mileage_num": 198000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 200.0, "torque_num": 300.0,
    "annual_insurance": 30000.0, "acceleration_num": 7.7,
    "seats_num": 4.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", # Mapping 'Accident free' from CSV
    "usage_type_clean": "Kenyan Used",
    "body_type_cleaned": "suv", "is_luxury_make": 0,
    "make_model_cleaned": "subaru_forester xt"
}
td2_derived_api = calculate_derived_features(
    td2_base_api["year_of_manufacture"], td2_base_api["mileage_num"], 
    td2_base_api["engine_size_cc_num"], td2_base_api["horse_power_num"], 
    td2_base_api["torque_num"]
)
td2_input_api = {
    **{k:v for k,v in td2_base_api.items() if k != 'year_of_manufacture'}, 
    **td2_derived_api
}
TEST_CASES.append({
    "description": "Training Data API: Subaru Forester XT (car_age 22, Ref ~0.75M)",
    "input": td2_input_api,
    "expected_price_range": (td2_actual_price_api * 0.95, td2_actual_price_api * 1.05)
})

# Training Data Case 3: Volvo XC60 (Row 7 of snippet)
td3_actual_price_api = np.exp(14.508657738524219)
td3_base_api = {
    "make_name_cleaned": "volvo", "model_name_cleaned": "xc60",
    "year_of_manufacture": YEAR_CSV_GENERATED - 10, # car_age = 10
    "mileage_num": 136000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 240.0, "torque_num": 320.0,
    "annual_insurance": 80000.0, "acceleration_num": 7.9,
    "seats_num": 5.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", # Mapping 'Accident free' from CSV
    "usage_type_clean": "Kenyan Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "volvo_xc60"
}
td3_derived_api = calculate_derived_features(
    td3_base_api["year_of_manufacture"], td3_base_api["mileage_num"], 
    td3_base_api["engine_size_cc_num"], td3_base_api["horse_power_num"], 
    td3_base_api["torque_num"]
)
td3_input_api = {
    **{k:v for k,v in td3_base_api.items() if k != 'year_of_manufacture'}, 
    **td3_derived_api
}
TEST_CASES.append({
    "description": "Training Data API: Volvo XC60 (car_age 10, Ref ~2.0M)",
    "input": td3_input_api,
    "expected_price_range": (td3_actual_price_api * 0.95, td3_actual_price_api * 1.05)
})

# Training Data Case 4: Land Rover Range Rover Vogue (Row 13 of snippet)
td4_actual_price_api = np.exp(16.314813170178272)
td4_base_api = {
    "make_name_cleaned": "land rover", "model_name_cleaned": "range rover vogue",
    "year_of_manufacture": YEAR_CSV_GENERATED - 7, # car_age = 7
    "mileage_num": 48000.0, "engine_size_cc_num": 3000.0,
    "fuel_type_cleaned": "diesel", "transmission_cleaned": "automatic",
    "horse_power_num": 258.0, "torque_num": 600.0,
    "annual_insurance": 486960.0, "acceleration_num": 7.9,
    "seats_num": 5.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", # Mapping 'Accident free' from CSV
    "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "land rover_range rover vogue"
}
td4_derived_api = calculate_derived_features(
    td4_base_api["year_of_manufacture"], td4_base_api["mileage_num"], 
    td4_base_api["engine_size_cc_num"], td4_base_api["horse_power_num"], 
    td4_base_api["torque_num"]
)
td4_input_api = {
    **{k:v for k,v in td4_base_api.items() if k != 'year_of_manufacture'}, 
    **td4_derived_api
}
TEST_CASES.append({
    "description": "Training Data API: Land Rover Vogue (car_age 7, Ref ~12.1M)",
    "input": td4_input_api,
    "expected_price_range": (td4_actual_price_api * 0.95, td4_actual_price_api * 1.05)
})

# --- End Test Cases from Training Data Snippet ---

# --- Test Cases from cars_modeling_input.csv snippet (lines 420-430) ---
# YEAR_CSV_GENERATED is already defined as 2025 for previous training data cases

# Case 1: Mazda Atenza 2017
csv_api1_ref_price = 2659999.0
csv_api1_base = {
    "make_name_cleaned": "mazda", "model_name_cleaned": "atenza",
    "year_of_manufacture": 2017,
    "mileage_num": 82000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 158.0, "torque_num": 210.0, "acceleration_num": 8.0, "seats_num": 5.0,
    "annual_insurance": 106400.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "sedan", "is_luxury_make": 0,
    "make_model_cleaned": "mazda_atenza"
}
csv_api1_derived = calculate_derived_features(csv_api1_base["year_of_manufacture"], csv_api1_base["mileage_num"], csv_api1_base["engine_size_cc_num"], csv_api1_base["horse_power_num"], csv_api1_base["torque_num"], csv_api1_base["acceleration_num"], csv_api1_base["seats_num"])
csv_api1_input = {**{k:v for k,v in csv_api1_base.items() if k != 'year_of_manufacture'}, **csv_api1_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Mazda Atenza 2017 (Ref ~2.66M)", "input": csv_api1_input,
    "expected_price_range": (csv_api1_ref_price * 0.90, csv_api1_ref_price * 1.10)
})

# Case 2: Porsche Cayenne 2018 (1)
csv_api2_ref_price = 8899999.0
csv_api2_base = {
    "make_name_cleaned": "porsche", "model_name_cleaned": "cayenne",
    "year_of_manufacture": 2018,
    "mileage_num": 44000.0, "engine_size_cc_num": 3000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 340.0, "torque_num": 450.0, "acceleration_num": 6.2, "seats_num": 5.0,
    "annual_insurance": 356000.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "porsche_cayenne"
}
csv_api2_derived = calculate_derived_features(csv_api2_base["year_of_manufacture"], csv_api2_base["mileage_num"], csv_api2_base["engine_size_cc_num"], csv_api2_base["horse_power_num"], csv_api2_base["torque_num"], csv_api2_base["acceleration_num"], csv_api2_base["seats_num"])
csv_api2_input = {**{k:v for k,v in csv_api2_base.items() if k != 'year_of_manufacture'}, **csv_api2_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Porsche Cayenne 2018 (1) (Ref ~8.9M)", "input": csv_api2_input,
    "expected_price_range": (csv_api2_ref_price * 0.90, csv_api2_ref_price * 1.10)
})

# Case 3: Porsche Cayenne 2018 (2)
csv_api3_ref_price = 7999999.0
csv_api3_base = {
    "make_name_cleaned": "porsche", "model_name_cleaned": "cayenne",
    "year_of_manufacture": 2018,
    "mileage_num": 46000.0, "engine_size_cc_num": 3000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 340.0, "torque_num": 450.0, "acceleration_num": 6.2, "seats_num": 5.0,
    "annual_insurance": 320000.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "porsche_cayenne"
}
csv_api3_derived = calculate_derived_features(csv_api3_base["year_of_manufacture"], csv_api3_base["mileage_num"], csv_api3_base["engine_size_cc_num"], csv_api3_base["horse_power_num"], csv_api3_base["torque_num"], csv_api3_base["acceleration_num"], csv_api3_base["seats_num"])
csv_api3_input = {**{k:v for k,v in csv_api3_base.items() if k != 'year_of_manufacture'}, **csv_api3_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Porsche Cayenne 2018 (2) (Ref ~8.0M)", "input": csv_api3_input,
    "expected_price_range": (csv_api3_ref_price * 0.90, csv_api3_ref_price * 1.10)
})

# Case 4: Toyota Landcruiser Zx 2023
csv_api4_ref_price = 18404000.0
csv_api4_base = {
    "make_name_cleaned": "toyota", "model_name_cleaned": "landcruiser",
    "year_of_manufacture": 2023,
    "mileage_num": 7000.0, "engine_size_cc_num": 3300.0,
    "fuel_type_cleaned": "diesel", "transmission_cleaned": "automatic",
    "horse_power_num": 305.0, "torque_num": 700.0, "acceleration_num": 7.0, "seats_num": 4.0,
    "annual_insurance": 736160.0, "drive_type_cleaned": "4wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 0,
    "make_model_cleaned": "toyota_landcruiser"
}
csv_api4_derived = calculate_derived_features(csv_api4_base["year_of_manufacture"], csv_api4_base["mileage_num"], csv_api4_base["engine_size_cc_num"], csv_api4_base["horse_power_num"], csv_api4_base["torque_num"], csv_api4_base["acceleration_num"], csv_api4_base["seats_num"])
csv_api4_input = {**{k:v for k,v in csv_api4_base.items() if k != 'year_of_manufacture'}, **csv_api4_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Toyota Landcruiser Zx 2023 (Ref ~18.4M)", "input": csv_api4_input,
    "expected_price_range": (csv_api4_ref_price * 0.90, csv_api4_ref_price * 1.10)
})

# Case 5: Daihatsu Mira 2017
csv_api5_ref_price = 879999.0
csv_api5_base = {
    "make_name_cleaned": "daihatsu", "model_name_cleaned": "other_model",
    "year_of_manufacture": 2017,
    "mileage_num": 70000.0, "engine_size_cc_num": 660.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 58.0, "torque_num": 68.0, "acceleration_num": 17.5, "seats_num": 4.0,
    "annual_insurance": 35200.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "hatchback", "is_luxury_make": 0,
    "make_model_cleaned": "daihatsu_other_model"
}
csv_api5_derived = calculate_derived_features(csv_api5_base["year_of_manufacture"], csv_api5_base["mileage_num"], csv_api5_base["engine_size_cc_num"], csv_api5_base["horse_power_num"], csv_api5_base["torque_num"], csv_api5_base["acceleration_num"], csv_api5_base["seats_num"])
csv_api5_input = {**{k:v for k,v in csv_api5_base.items() if k != 'year_of_manufacture'}, **csv_api5_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Daihatsu Mira 2017 (Ref ~0.88M)", "input": csv_api5_input,
    "expected_price_range": (csv_api5_ref_price * 0.90, csv_api5_ref_price * 1.10)
})

# Case 6: Nissan Xtrail 2017
csv_api6_ref_price = 2749999.0
csv_api6_base = {
    "make_name_cleaned": "nissan", "model_name_cleaned": "xtrail",
    "year_of_manufacture": 2017,
    "mileage_num": 125000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 155.0, "torque_num": 195.0, "acceleration_num": 10.0, "seats_num": 5.0,
    "annual_insurance": 110000.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 0,
    "make_model_cleaned": "nissan_xtrail"
}
csv_api6_derived = calculate_derived_features(csv_api6_base["year_of_manufacture"], csv_api6_base["mileage_num"], csv_api6_base["engine_size_cc_num"], csv_api6_base["horse_power_num"], csv_api6_base["torque_num"], csv_api6_base["acceleration_num"], csv_api6_base["seats_num"])
csv_api6_input = {**{k:v for k,v in csv_api6_base.items() if k != 'year_of_manufacture'}, **csv_api6_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Nissan Xtrail 2017 (Ref ~2.75M)", "input": csv_api6_input,
    "expected_price_range": (csv_api6_ref_price * 0.90, csv_api6_ref_price * 1.10)
})

# Case 7: Mercedes C180 2017
csv_api7_ref_price = 4050000.0
csv_api7_base = {
    "make_name_cleaned": "mercedes", "model_name_cleaned": "c class",
    "year_of_manufacture": 2017,
    "mileage_num": 32000.0, "engine_size_cc_num": 1600.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 156.0, "torque_num": 250.0, "acceleration_num": 8.5, "seats_num": 3.0,
    "annual_insurance": 162000.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "coupe", "is_luxury_make": 1,
    "make_model_cleaned": "mercedes_c class"
}
csv_api7_derived = calculate_derived_features(csv_api7_base["year_of_manufacture"], csv_api7_base["mileage_num"], csv_api7_base["engine_size_cc_num"], csv_api7_base["horse_power_num"], csv_api7_base["torque_num"], csv_api7_base["acceleration_num"], csv_api7_base["seats_num"])
csv_api7_input = {**{k:v for k,v in csv_api7_base.items() if k != 'year_of_manufacture'}, **csv_api7_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Mercedes C180 2017 (Ref ~4.05M)", "input": csv_api7_input,
    "expected_price_range": (csv_api7_ref_price * 0.90, csv_api7_ref_price * 1.10)
})

# Case 8: Mercedes C200 2017
csv_api8_ref_price = 3950000.0
csv_api8_base = {
    "make_name_cleaned": "mercedes", "model_name_cleaned": "c class",
    "year_of_manufacture": 2017,
    "mileage_num": 36000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 184.0, "torque_num": 300.0, "acceleration_num": 7.2, "seats_num": 6.0,
    "annual_insurance": 158000.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "sedan", "is_luxury_make": 1,
    "make_model_cleaned": "mercedes_c class"
}
csv_api8_derived = calculate_derived_features(csv_api8_base["year_of_manufacture"], csv_api8_base["mileage_num"], csv_api8_base["engine_size_cc_num"], csv_api8_base["horse_power_num"], csv_api8_base["torque_num"], csv_api8_base["acceleration_num"], csv_api8_base["seats_num"])
csv_api8_input = {**{k:v for k,v in csv_api8_base.items() if k != 'year_of_manufacture'}, **csv_api8_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Mercedes C200 2017 (Ref ~3.95M)", "input": csv_api8_input,
    "expected_price_range": (csv_api8_ref_price * 0.90, csv_api8_ref_price * 1.10)
})

# Case 9: Audi Q5 2017
csv_api9_ref_price = 4599999.0
csv_api9_base = {
    "make_name_cleaned": "audi", "model_name_cleaned": "q5",
    "year_of_manufacture": 2017,
    "mileage_num": 23000.0, "engine_size_cc_num": 2000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 220.0, "torque_num": 350.0, "acceleration_num": 7.1, "seats_num": 4.0,
    "annual_insurance": 184000.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "audi_q5"
}
csv_api9_derived = calculate_derived_features(csv_api9_base["year_of_manufacture"], csv_api9_base["mileage_num"], csv_api9_base["engine_size_cc_num"], csv_api9_base["horse_power_num"], csv_api9_base["torque_num"], csv_api9_base["acceleration_num"], csv_api9_base["seats_num"])
csv_api9_input = {**{k:v for k,v in csv_api9_base.items() if k != 'year_of_manufacture'}, **csv_api9_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Audi Q5 2017 (Ref ~4.6M)", "input": csv_api9_input,
    "expected_price_range": (csv_api9_ref_price * 0.90, csv_api9_ref_price * 1.10)
})

# Case 10: Land Rover Range Rover Sport 2019
csv_api10_ref_price = 12100000.0
csv_api10_base = {
    "make_name_cleaned": "land rover", "model_name_cleaned": "range rover sport",
    "year_of_manufacture": 2019,
    "mileage_num": 47000.0, "engine_size_cc_num": 3000.0,
    "fuel_type_cleaned": "petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 306.0, "torque_num": 700.0, "acceleration_num": 7.1, "seats_num": 4.0,
    "annual_insurance": 484000.0, "drive_type_cleaned": "awd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 1,
    "make_model_cleaned": "land rover_range rover sport"
}
csv_api10_derived = calculate_derived_features(csv_api10_base["year_of_manufacture"], csv_api10_base["mileage_num"], csv_api10_base["engine_size_cc_num"], csv_api10_base["horse_power_num"], csv_api10_base["torque_num"], csv_api10_base["acceleration_num"], csv_api10_base["seats_num"])
csv_api10_input = {**{k:v for k,v in csv_api10_base.items() if k != 'year_of_manufacture'}, **csv_api10_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Land Rover Range Rover Sport 2019 (Ref ~12.1M)", "input": csv_api10_input,
    "expected_price_range": (csv_api10_ref_price * 0.90, csv_api10_ref_price * 1.10)
})

# Case 11: Honda Vezel 2017
csv_api11_ref_price = 2649999.0
csv_api11_base = {
    "make_name_cleaned": "honda", "model_name_cleaned": "vezel",
    "year_of_manufacture": 2017,
    "mileage_num": 96000.0, "engine_size_cc_num": 1500.0,
    "fuel_type_cleaned": "hybrid_petrol", "transmission_cleaned": "automatic",
    "horse_power_num": 130.0, "torque_num": 165.0, "acceleration_num": 8.5, "seats_num": 9.0,
    "annual_insurance": 106000.0, "drive_type_cleaned": "2wd",
    "condition_clean": "used", "usage_type_clean": "Foreign Used",
    "body_type_cleaned": "suv", "is_luxury_make": 0,
    "make_model_cleaned": "honda_vezel"
}
csv_api11_derived = calculate_derived_features(csv_api11_base["year_of_manufacture"], csv_api11_base["mileage_num"], csv_api11_base["engine_size_cc_num"], csv_api11_base["horse_power_num"], csv_api11_base["torque_num"], csv_api11_base["acceleration_num"], csv_api11_base["seats_num"])
csv_api11_input = {**{k:v for k,v in csv_api11_base.items() if k != 'year_of_manufacture'}, **csv_api11_derived}
TEST_CASES.append({
    "description": "CSV Snippet API: Honda Vezel 2017 (Ref ~2.65M)", "input": csv_api11_input,
    "expected_price_range": (csv_api11_ref_price * 0.90, csv_api11_ref_price * 1.10)
})

# --- End Test Cases from cars_modeling_input.csv snippet ---

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