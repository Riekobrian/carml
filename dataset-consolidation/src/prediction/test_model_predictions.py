import pandas as pd
import numpy as np
from predict import CarPricePredictor
import logging
from tabulate import tabulate
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_test_case(predictor, test_case, description, expected_range=None):
    """
    Run a single test case and return the results.
    """
    try:
        # Make prediction
        result = predictor.predict(test_case)
        predicted_price = result['predicted_price'][0]
        
        # Check if prediction is within expected range
        status = "PASSED"
        if expected_range:
            min_price, max_price = expected_range
            if predicted_price < min_price or predicted_price > max_price:
                status = "FAILED"
        
        # Format output
        print(f"\n--- Running Test Case: {description} ---")
        print(f"Predicted Price: KES {predicted_price:,.0f}")
        if expected_range:
            print(f"Expected Range: KES {min_price:,.0f} - {max_price:,.0f}")
        print(f"Result: {status}")
        
        return {
            'description': description,
            'status': status,
            'predicted_price': predicted_price,
            'details': f"Expected: KES {min_price:,.0f} - {max_price:,.0f}" if expected_range else ""
        }
        
    except Exception as e:
        logger.error(f"Error running test case {description}: {e}")
        return {
            'description': description,
            'status': 'ERROR',
            'predicted_price': None,
            'details': str(e)
        }

def create_one_hot_features(data):
    """
    Convert categorical features to one-hot encoded format.
    """
    # Initialize all possible categorical features to 0
    one_hot_features = {}
    
    # Fuel type
    fuel_types = ['diesel', 'electric', 'hybrid_diesel', 'hybrid_petrol', 'petrol', 'plugin_hybrid_petrol', 'unknown']
    for ft in fuel_types:
        one_hot_features[f'cat__fuel_type_cleaned_{ft}'] = 0
    one_hot_features[f'cat__fuel_type_cleaned_{data["fuel_type_cleaned"]}'] = 1
    
    # Transmission
    transmissions = ['automated_manual', 'automatic', 'manual', 'unknown']
    for t in transmissions:
        one_hot_features[f'cat__transmission_cleaned_{t}'] = 0
    one_hot_features[f'cat__transmission_cleaned_{data["transmission_cleaned"]}'] = 1
    
    # Drive type
    drive_types = ['2wd', '2wd_front', '2wd_mid_engine', '2wd_rear', '2wd_rear_engine', '4wd', 'awd', 'unknown']
    for dt in drive_types:
        one_hot_features[f'cat__drive_type_cleaned_{dt}'] = 0
    one_hot_features[f'cat__drive_type_cleaned_{data["drive_type_cleaned"]}'] = 1
    
    # Usage type
    usage_types = ['Foreign Used', 'Kenyan Used']
    for ut in usage_types:
        one_hot_features[f'cat__usage_type_clean_{ut}'] = 0
    one_hot_features[f'cat__usage_type_clean_{data["usage_type_clean"]}'] = 1
    
    # Body type
    body_types = ['bus', 'convertible', 'coupe', 'hatchback', 'pickup', 'pickup_truck', 'sedan', 
                 'special_purpose_truck', 'suv', 'unknown', 'van_minivan', 'wagon', 'wagon_estate']
    for bt in body_types:
        one_hot_features[f'cat__body_type_cleaned_{bt}'] = 0
    one_hot_features[f'cat__body_type_cleaned_{data["body_type_cleaned"]}'] = 1
    
    # Make name
    makes = ['amg', 'aston', 'audi', 'bentley', 'bmw', 'chevrolet', 'daihatsu', 'ferrari', 'ford',
            'hino', 'honda', 'isuzu', 'jaguar', 'jeep', 'lamborghini', 'land', 'land rover', 'lexus',
            'maserati', 'mazda', 'mercedes', 'mitsubishi', 'nissan', 'other_make', 'peugeot',
            'porsche', 'renault', 'rolls', 'subaru', 'suzuki', 'toyota', 'unknown_placeholder',
            'volkswagen', 'volvo']
    for m in makes:
        one_hot_features[f'cat__make_name_cleaned_{m}'] = 0
    one_hot_features[f'cat__make_name_cleaned_{data["make_name_cleaned"]}'] = 1
    
    # Condition
    conditions = ['Accident involved', 'used', 'new']
    for c in conditions:
        one_hot_features[f'cat__condition_clean_{c}'] = 0
    one_hot_features[f'cat__condition_clean_{data["condition_clean"]}'] = 1
    
    # Add numeric features with prefixes
    numeric_features = {
        f'num_insurance__annual_insurance': data['annual_insurance'],
        f'num_main__car_age': data['car_age'],
        f'num_main__car_age_squared': data['car_age_squared'],
        f'num_main__mileage_num': data['mileage_num'],
        f'num_main__mileage_log': data['mileage_log'],
        f'num_main__mileage_per_year': data['mileage_per_year'],
        f'num_main__engine_size_cc_num': data['engine_size_cc_num'],
        f'num_main__engine_size_cc_log': data['engine_size_cc_log'],
        f'num_main__horse_power_num': data['horse_power_num'],
        f'num_main__horse_power_log': data['horse_power_log'],
        f'num_main__torque_num': data['torque_num'],
        f'num_main__torque_log': data['torque_log'],
        f'num_main__power_per_cc': data['power_per_cc'],
        f'num_main__acceleration_num': data['acceleration_num'],
        f'num_main__seats_num': data['seats_num'],
        f'num_main__mileage_per_cc': data['mileage_per_cc'],
        f'num_main__is_luxury_make': data['is_luxury_make']
    }
    
    # Initialize all model name features to 0
    model_names = [
        '116i', '208', '3 series', '3008', '360', '458 italia', '911', 'a class', 'a6', 'ad van',
        'allion', 'alphard', 'alphard hybrid', 'alto', 'alto 2018', 'aqua', 'aqua 2018', 'atenza',
        'atenza sedan', 'auris', 'aventador', 'axela', 'axela sport', 'axela sport 2018', 'axio',
        'bb', 'bentayga', 'c class', 'c class 2018', 'canter truck', 'caravan', 'cayenne', 'cayman',
        'coaster', 'continental', 'corolla', 'corolla axio', 'corolla axio 2019', 'corolla fielder',
        'corolla fielder 2018', 'corolla sport 2022', 'corona premio', 'crown', 'crown hybrid 2020',
        'crv', 'cube', 'cullinan', 'cx', 'cx-5', 'cx3 2018', 'cx5', 'cx5 2018', 'cx8 2018', 'dawn',
        'dbs', 'dbx', 'defender', 'demio', 'discovery iv', 'dutro', 'dyna', 'e class', 'e200',
        'elf truck', 'escudo', 'esquire', 'f type', 'f430', 'f430 spider', 'f8 tributo',
        'fairlady z', 'ff', 'fielder', 'fighter', 'fit', 'fit 2018', 'fit hybrid', 'fj cruiser',
        'flying spur', 'forester', 'forester 2018', 'fortuner', 'freed hybrid', 'g class',
        'gallardo', 'gclass', 'ghibli', 'golf', 'golf tsi', 'grand cherokee', 'gt-r', 'gtc 4lusso',
        'harrier', 'harrier 2018', 'hiace', 'hiace van', 'hiace van 2018', 'hiace wagon',
        'hijet truck', 'hilux', 'huracan', 'impreza', 'impreza g4', 'impreza sport', 'isis', 'ist',
        'juke', 'land', 'land cruiser', 'land cruiser 70', 'land cruiser prado', 'landcruiser',
        'landcruiser 200 series', 'legacy b4', 'levante', 'lx', 'lx 600', 'lx600', 'march',
        'mark x', 'mark x 2018', 'mark x zio', 'mc20', 'mini', 'mira es', 'mpv', 'murano', 'noah',
        'note', 'note 2018', 'note 2019', 'nsx', 'nv150 ad 2018', 'nv350caravan', 'nx', 'odyssey',
        'other_model', 'outback', 'outlander', 'passat', 'passo', 'patrol', 'phantom', 'polo',
        'porte', 'premacy', 'premio', 'premio 2018', 'profia', 'q5', 'ractis', 'range rover',
        'range rover evoque', 'range rover sport', 'ranger', 'raum', 'rav4', 'regiusace van',
        'rome', 'rosa', 'rvr', 'rx7', 'serena', 'shuttle 2019', 'sienta', 'skyline', 'spade',
        'stepwagon', 'stepwagon spada', 'super great', 'supra', 'swift', 'swift sport', 'sylphy',
        'teana', 'tiguan', 'touareg', 'toyoace truck', 'unknown_placeholder', 'urus', 'v40',
        'vanguard', 'vellfire', 'vezel', 'vezel 2020', 'vitz', 'voxy', 'x-trail', 'x1', 'x3',
        'x5', 'x6', 'xc60', 'xc90', 'xtrail', 'xv'
    ]
    
    for model in model_names:
        one_hot_features[f'cat__model_name_cleaned_{model}'] = 0
    one_hot_features[f'cat__model_name_cleaned_{data["model_name_cleaned"]}'] = 1
    
    # Initialize all make-model combinations to 0
    make_model_combinations = [
        'amg_gclass', 'amg_other_model', 'aston_dbs', 'aston_dbx', 'aston_other_model', 'audi_a6',
        'audi_other_model', 'audi_q5', 'bentley_bentayga', 'bentley_continental', 'bmw_3 series',
        'bmw_dawn', 'bmw_other_model', 'bmw_x1', 'bmw_x3', 'bmw_x5', 'bmw_x6',
        'daihatsu_hijet truck', 'daihatsu_other_model', 'ferrari_360', 'ferrari_458 italia',
        'ferrari_458 spider', 'ferrari_488 spider', 'ferrari_f430', 'ferrari_f430 spider',
        'ferrari_f8 spider', 'ferrari_f8 tributo', 'ferrari_gtc 4lusso', 'ferrari_other_model',
        'ford_ranger', 'hino_dutro', 'hino_dutro hybrid', 'hino_profia', 'hino_ranger',
        'honda_crv', 'honda_fit', 'honda_fit 2018', 'honda_fit hybrid', 'honda_freed hybrid',
        'honda_nsx', 'honda_odyssey', 'honda_other_model', 'honda_shuttle 2019',
        'honda_stepwagon', 'honda_stepwagon spada', 'honda_vezel', 'honda_vezel 2018',
        'isuzu_elf truck', 'isuzu_forward', 'isuzu_giga', 'jaguar_f type', 'jaguar_other_model',
        'jeep_grand cherokee', 'lamborghini_aventador', 'lamborghini_gallardo',
        'lamborghini_huracan', 'lamborghini_other_model', 'lamborghini_urus',
        'land rover_defender', 'land rover_other_model', 'land rover_range rover evoque',
        'land rover_range rover sport', 'land rover_range rover vogue', 'land_defender',
        'land_other_model', 'land_range rover', 'land_range rover sport', 'lexus_lx',
        'lexus_lx600', 'lexus_nx', 'lexus_nx200t', 'lexus_other_model', 'maserati_ghibli',
        'maserati_mc20', 'mazda_atenza', 'mazda_atenza sedan', 'mazda_axela',
        'mazda_axela sport', 'mazda_axela sport 2018', 'mazda_continental', 'mazda_cx-5',
        'mazda_cx3 2018', 'mazda_cx5', 'mazda_cx5 2018', 'mazda_cx8 2018', 'mazda_demio',
        'mazda_other_model', 'mazda_premacy', 'mazda_rx7', 'mercedes_a class', 'mercedes_c class',
        'mercedes_c class 2018', 'mercedes_e class', 'mercedes_g class', 'mercedes_glc',
        'mercedes_m class', 'mercedes_other_model', 'mercedes_unknown_placeholder',
        'mitsubishi_canter truck', 'mitsubishi_fighter', 'mitsubishi_other_model',
        'mitsubishi_outlander', 'mitsubishi_rosa', 'mitsubishi_rvr', 'mitsubishi_super great',
        'nissan_ad van', 'nissan_caravan', 'nissan_cube', 'nissan_fairlady z', 'nissan_juke',
        'nissan_latio', 'nissan_march', 'nissan_murano', 'nissan_note', 'nissan_note 2018',
        'nissan_nv150 ad 2018', 'nissan_other_model', 'nissan_patrol', 'nissan_serena',
        'nissan_skyline', 'nissan_teana', 'nissan_x-trail', 'nissan_xtrail',
        'other_make_levante', 'other_make_mini', 'other_make_other_model', 'other_make_phantom',
        'peugeot_208', 'porsche_911', 'porsche_cayenne', 'porsche_cayman',
        'porsche_other_model', 'renault_other_model', 'rolls_cullinan', 'rolls_dawn',
        'rolls_other_model', 'subaru_forester', 'subaru_forester 2018', 'subaru_impreza',
        'subaru_impreza g4', 'subaru_impreza sport', 'subaru_other_model', 'subaru_outback',
        'subaru_xv', 'suzuki_alto 2018', 'suzuki_escudo', 'toyota_allion', 'toyota_alphard',
        'toyota_aqua', 'toyota_aqua 2018', 'toyota_auris', 'toyota_axio', 'toyota_bb',
        'toyota_coaster', 'toyota_continental', 'toyota_corolla', 'toyota_corolla axio',
        'toyota_corolla axio 2019', 'toyota_corolla fielder', 'toyota_corolla fielder 2018',
        'toyota_corolla sport 2022', 'toyota_corona premio', 'toyota_crown',
        'toyota_crown hybrid 2020', 'toyota_cullinan', 'toyota_dbs', 'toyota_dyna',
        'toyota_fielder', 'toyota_fj cruiser', 'toyota_fortuner', 'toyota_funcargo',
        'toyota_harrier', 'toyota_harrier 2018', 'toyota_hiace', 'toyota_hiace van',
        'toyota_hiace van 2018', 'toyota_hiace wagon', 'toyota_hilux', 'toyota_isis',
        'toyota_ist', 'toyota_land', 'toyota_land cruiser', 'toyota_land cruiser 70',
        'toyota_land cruiser prado', 'toyota_landcruiser', 'toyota_landcruiser 200 series',
        'toyota_mark x', 'toyota_mark x 2018', 'toyota_mark x zio', 'toyota_noah',
        'toyota_other_model', 'toyota_passo', 'toyota_porte', 'toyota_premio',
        'toyota_premio 2018', 'toyota_probox van', 'toyota_ractis', 'toyota_raize 2020',
        'toyota_raum', 'toyota_rav4', 'toyota_regiusace van', 'toyota_sienta',
        'toyota_sienta 2018', 'toyota_skyline', 'toyota_spade', 'toyota_supra',
        'toyota_toyoace truck', 'toyota_urus', 'toyota_vanguard', 'toyota_vellfire',
        'toyota_vitz', 'toyota_voxy', 'unknown_placeholder_unknown_placeholder',
        'volkswagen_golf', 'volkswagen_other_model', 'volkswagen_passat', 'volkswagen_tiguan',
        'volkswagen_touareg', 'volvo_v40', 'volvo_xc60', 'volvo_xc90'
    ]
    
    for make_model in make_model_combinations:
        one_hot_features[f'cat__make_model_cleaned_{make_model}'] = 0
    one_hot_features[f'cat__make_model_cleaned_{data["make_model_cleaned"]}'] = 1
    
    # Combine all features
    return {**one_hot_features, **numeric_features}

def main():
    """Run test cases and display results."""
    try:
        # Initialize predictor
        predictor = CarPricePredictor()
        
        # Define test cases
        test_cases = [
            {
                'description': "Toyota Crown 2011 (Foreign Used)",
                'data': {
                    'mileage_per_cc': 12.5,
                    'car_age': 13,
                    'car_age_squared': 169,
                    'annual_insurance': 45000,
                    'engine_size_cc_num': 2500,
                    'engine_size_cc_log': np.log(2500),
                    'mileage_num': 150000,
                    'mileage_log': np.log(150000),
                    'mileage_per_year': 11538.46,
                    'horse_power_num': 180,
                    'horse_power_log': np.log(180),
                    'torque_num': 235,
                    'torque_log': np.log(235),
                    'power_per_cc': 0.072,
                    'acceleration_num': 9.5,
                    'seats_num': 5,
                    'is_luxury_make': 0,
                    'usage_type_clean': 'Foreign Used',
                    'body_type_cleaned': 'sedan',
                    'make_name_cleaned': 'toyota',
                    'transmission_cleaned': 'automatic',
                    'drive_type_cleaned': '2wd',
                    'fuel_type_cleaned': 'petrol',
                    'condition_clean': 'used',
                    'model_name_cleaned': 'crown',
                    'make_model_cleaned': 'toyota_crown'
                },
                'expected_range': (863999, 1295999)
            },
            {
                'description': "Toyota Prado 2020 (Foreign Used)",
                'data': {
                    'mileage_per_cc': 8.5,
                    'car_age': 4,
                    'car_age_squared': 16,
                    'annual_insurance': 85000,
                    'engine_size_cc_num': 2800,
                    'engine_size_cc_log': np.log(2800),
                    'mileage_num': 80000,
                    'mileage_log': np.log(80000),
                    'mileage_per_year': 20000,
                    'horse_power_num': 204,
                    'horse_power_log': np.log(204),
                    'torque_num': 500,
                    'torque_log': np.log(500),
                    'power_per_cc': 0.073,
                    'acceleration_num': 9.2,
                    'seats_num': 7,
                    'is_luxury_make': 0,
                    'usage_type_clean': 'Foreign Used',
                    'body_type_cleaned': 'suv',
                    'make_name_cleaned': 'toyota',
                    'transmission_cleaned': 'automatic',
                    'drive_type_cleaned': '4wd',
                    'fuel_type_cleaned': 'diesel',
                    'condition_clean': 'used',
                    'model_name_cleaned': 'land cruiser prado',
                    'make_model_cleaned': 'toyota_land cruiser prado'
                },
                'expected_range': (6000000, 9000000)
            },
            {
                'description': "BMW X3 2004 (Accident Involved)",
                'data': {
                    'mileage_per_cc': 11.2,
                    'car_age': 20,
                    'car_age_squared': 400,
                    'annual_insurance': 55000,
                    'engine_size_cc_num': 2500,
                    'engine_size_cc_log': np.log(2500),
                    'mileage_num': 220000,
                    'mileage_log': np.log(220000),
                    'mileage_per_year': 11000,
                    'horse_power_num': 184,
                    'horse_power_log': np.log(184),
                    'torque_num': 245,
                    'torque_log': np.log(245),
                    'power_per_cc': 0.0736,
                    'acceleration_num': 10.2,
                    'seats_num': 5,
                    'is_luxury_make': 1,
                    'usage_type_clean': 'Foreign Used',
                    'body_type_cleaned': 'suv',
                    'make_name_cleaned': 'bmw',
                    'transmission_cleaned': 'automatic',
                    'drive_type_cleaned': '4wd',
                    'fuel_type_cleaned': 'petrol',
                    'condition_clean': 'Accident involved',
                    'model_name_cleaned': 'x3',
                    'make_model_cleaned': 'bmw_x3'
                },
                'expected_range': (427501, 712501)
            },
            {
                'description': "Toyota Axio 2016 (Foreign Used)",
                'data': {
                    'mileage_per_cc': 14.5,
                    'car_age': 8,
                    'car_age_squared': 64,
                    'annual_insurance': 40000,
                    'engine_size_cc_num': 1500,
                    'engine_size_cc_log': np.log(1500),
                    'mileage_num': 120000,
                    'mileage_log': np.log(120000),
                    'mileage_per_year': 15000,
                    'horse_power_num': 110,
                    'horse_power_log': np.log(110),
                    'torque_num': 145,
                    'torque_log': np.log(145),
                    'power_per_cc': 0.073,
                    'acceleration_num': 11.0,
                    'seats_num': 5,
                    'is_luxury_make': 0,
                    'usage_type_clean': 'Foreign Used',
                    'body_type_cleaned': 'sedan',
                    'make_name_cleaned': 'toyota',
                    'transmission_cleaned': 'automatic',
                    'drive_type_cleaned': '2wd',
                    'fuel_type_cleaned': 'petrol',
                    'condition_clean': 'used',
                    'model_name_cleaned': 'axio',
                    'make_model_cleaned': 'toyota_axio'
                },
                'expected_range': (800000, 1200000)
            }
        ]
        
        # --- New Custom Test Case for Toyota Axio 2016 ---
        # Assuming current_year for age calculation, adjust if tests should be static to a specific year
        current_year_for_calc = datetime.now().year 
        axio_custom_year = 2016
        axio_custom_car_age = float(current_year_for_calc - axio_custom_year)
        axio_custom_mileage = 80000.0
        axio_custom_engine_cc = 1500.0
        axio_custom_hp = 109.0 
        axio_custom_torque = 138.0

        axio_custom_test_data = {
            'make_name_cleaned': 'toyota',
            'model_name_cleaned': 'axio',
            'body_type_cleaned': 'sedan',
            'car_age': axio_custom_car_age,
            'car_age_squared': axio_custom_car_age**2,
            'condition_clean': 'used',
            'mileage_num': axio_custom_mileage,
            'mileage_log': np.log1p(axio_custom_mileage),
            'mileage_per_year': axio_custom_mileage / (axio_custom_car_age + 1e-6) if axio_custom_car_age > 0 else axio_custom_mileage,
            'engine_size_cc_num': axio_custom_engine_cc,
            'engine_size_cc_log': np.log1p(axio_custom_engine_cc),
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '2wd',
            'horse_power_num': axio_custom_hp,
            'horse_power_log': np.log1p(axio_custom_hp) if axio_custom_hp is not None else None,
            'torque_num': axio_custom_torque,
            'torque_log': np.log1p(axio_custom_torque) if axio_custom_torque is not None else None,
            'acceleration_num': 11.5, 
            'seats_num': 5.0, 
            'annual_insurance': 45000.0, 
            'power_per_cc': axio_custom_hp / axio_custom_engine_cc if axio_custom_hp is not None and axio_custom_engine_cc > 0 else 0.0,
            'mileage_per_cc': axio_custom_mileage / axio_custom_engine_cc if axio_custom_engine_cc > 0 else 0.0,
            'is_luxury_make': 0,
            'make_model_cleaned': 'toyota_axio',
            'usage_type_clean': 'Foreign Used'
        }
        # --- End New Custom Test Case ---

        # Append the new test case to the existing list
        test_cases.append({
            'description': "Custom Toyota Axio 2016 (Foreign Used, 80k km, Ref 1.85M)",
            'data': axio_custom_test_data,
            # No expected_range for now, we just want to see the prediction against the reference 1.85M
            'expected_range': None 
        })

        # --- New Custom Test Case for 2017 Toyota Axio Hybrid ---
        # Assuming current_year for age calculation
        # current_year_for_calc is already defined above
        axio_hybrid_year = 2017
        axio_hybrid_car_age = float(current_year_for_calc - axio_hybrid_year)
        axio_hybrid_mileage = 99000.0
        axio_hybrid_engine_cc = 1500.0
        axio_hybrid_hp = 100.0  # Assumed system HP for hybrid
        axio_hybrid_torque = 140.0 # Assumed system torque for hybrid

        axio_hybrid_test_data = {
            'make_name_cleaned': 'toyota',
            'model_name_cleaned': 'axio', # Model is Axio, "Corolla Axio" is common naming
            'body_type_cleaned': 'sedan', # Saloon -> sedan
            'car_age': axio_hybrid_car_age,
            'car_age_squared': axio_hybrid_car_age**2,
            'condition_clean': 'used', # Assuming Foreign Used, good condition
            'mileage_num': axio_hybrid_mileage,
            'mileage_log': np.log1p(axio_hybrid_mileage),
            'mileage_per_year': axio_hybrid_mileage / (axio_hybrid_car_age + 1e-6) if axio_hybrid_car_age > 0 else axio_hybrid_mileage,
            'engine_size_cc_num': axio_hybrid_engine_cc,
            'engine_size_cc_log': np.log1p(axio_hybrid_engine_cc),
            'fuel_type_cleaned': 'hybrid_petrol', # From "Hybrid/P"
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '2wd',
            'horse_power_num': axio_hybrid_hp,
            'horse_power_log': np.log1p(axio_hybrid_hp) if axio_hybrid_hp is not None else None,
            'torque_num': axio_hybrid_torque,
            'torque_log': np.log1p(axio_hybrid_torque) if axio_hybrid_torque is not None else None,
            'acceleration_num': 12.0, # Assumed
            'seats_num': 5.0,
            'annual_insurance': 48000.0, # Assumed placeholder
            'power_per_cc': axio_hybrid_hp / axio_hybrid_engine_cc if axio_hybrid_hp is not None and axio_hybrid_engine_cc > 0 else 0.0,
            'mileage_per_cc': axio_hybrid_mileage / axio_hybrid_engine_cc if axio_hybrid_engine_cc > 0 else 0.0,
            'is_luxury_make': 0,
            'make_model_cleaned': 'toyota_axio', # make_model_cleaned might need adjustment if model saw 'toyota_corolla axio'
            'usage_type_clean': 'Foreign Used' # Assumed
        }
        test_cases.append({
            'description': "Custom 2017 Axio Hybrid (99k km, Ref 1.69M)",
            'data': axio_hybrid_test_data,
            'expected_range': None 
        })
        # --- End New Custom Test Case for 2017 Toyota Axio Hybrid ---

        # --- New Custom Test Case for High-Performance 2015 SUV ---
        # current_year_for_calc is already defined
        hp_suv_year = 2015
        hp_suv_car_age = float(current_year_for_calc - hp_suv_year)
        hp_suv_mileage = 122932.0
        hp_suv_engine_cc = 4400.0
        hp_suv_hp = 444.0
        hp_suv_torque = 650.0
        hp_suv_acceleration = 4.7
        hp_suv_insurance = 180000.0

        hp_suv_test_data = {
            'make_name_cleaned': 'bmw', # ASSUMED - User to confirm/correct
            'model_name_cleaned': 'x5',   # ASSUMED - User to confirm/correct (or a generic model name)
            'body_type_cleaned': 'suv',
            'car_age': hp_suv_car_age,
            'car_age_squared': hp_suv_car_age**2,
            'condition_clean': 'used', # Assuming Foreign Used
            'mileage_num': hp_suv_mileage,
            'mileage_log': np.log1p(hp_suv_mileage),
            'mileage_per_year': hp_suv_mileage / (hp_suv_car_age + 1e-6) if hp_suv_car_age > 0 else hp_suv_mileage,
            'engine_size_cc_num': hp_suv_engine_cc,
            'engine_size_cc_log': np.log1p(hp_suv_engine_cc),
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '4wd',
            'horse_power_num': hp_suv_hp,
            'horse_power_log': np.log1p(hp_suv_hp) if hp_suv_hp is not None else None,
            'torque_num': hp_suv_torque,
            'torque_log': np.log1p(hp_suv_torque) if hp_suv_torque is not None else None,
            'acceleration_num': hp_suv_acceleration,
            'seats_num': 5.0, # Assumed
            'annual_insurance': hp_suv_insurance,
            'power_per_cc': hp_suv_hp / hp_suv_engine_cc if hp_suv_hp is not None and hp_suv_engine_cc > 0 else 0.0,
            'mileage_per_cc': hp_suv_mileage / hp_suv_engine_cc if hp_suv_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, # Assumed BMW is luxury
            'make_model_cleaned': 'bmw_x5', # ASSUMED
            'usage_type_clean': 'Foreign Used' # Assumed
        }
        test_cases.append({
            'description': "Custom 2015 High-Perf SUV (123k km, Ref 4.5M)",
            'data': hp_suv_test_data,
            'expected_range': None 
        })
        # --- End New Custom Test Case for High-Performance 2015 SUV ---

        # --- New Custom Test Case for 2017 Volvo XC60 Diesel ---
        volvo_year = 2017
        volvo_car_age = float(current_year_for_calc - volvo_year)
        volvo_mileage = 80000.0  # Placeholder - USER TO PROVIDE ACTUAL MILEAGE
        volvo_engine_cc = 1969.0 # Assumed 2.0L common diesel engine
        volvo_hp = 190.0      # Assumed for D4/similar
        volvo_torque = 400.0  # Assumed for D4/similar
        volvo_acceleration = 8.5 # Assumed
        volvo_insurance = 120000.0 # Assumed

        volvo_xc60_test_data = {
            'make_name_cleaned': 'volvo',
            'model_name_cleaned': 'xc60',
            'body_type_cleaned': 'suv',
            'car_age': volvo_car_age,
            'car_age_squared': volvo_car_age**2,
            'condition_clean': 'used', # Foreign Used
            'mileage_num': volvo_mileage,
            'mileage_log': np.log1p(volvo_mileage),
            'mileage_per_year': volvo_mileage / (volvo_car_age + 1e-6) if volvo_car_age > 0 else volvo_mileage,
            'engine_size_cc_num': volvo_engine_cc,
            'engine_size_cc_log': np.log1p(volvo_engine_cc),
            'fuel_type_cleaned': 'diesel',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': 'awd', # Assumed, common for XC60
            'horse_power_num': volvo_hp,
            'horse_power_log': np.log1p(volvo_hp) if volvo_hp is not None else None,
            'torque_num': volvo_torque,
            'torque_log': np.log1p(volvo_torque) if volvo_torque is not None else None,
            'acceleration_num': volvo_acceleration,
            'seats_num': 5.0,
            'annual_insurance': volvo_insurance,
            'power_per_cc': volvo_hp / volvo_engine_cc if volvo_hp is not None and volvo_engine_cc > 0 else 0.0,
            'mileage_per_cc': volvo_mileage / volvo_engine_cc if volvo_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, # Volvo considered luxury/near-luxury
            'make_model_cleaned': 'volvo_xc60',
            'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "Custom 2017 Volvo XC60 (Mileage TBD, Ref 3.6M)",
            'data': volvo_xc60_test_data,
            'expected_range': None 
        })
        # --- End New Custom Test Case for 2017 Volvo XC60 ---

        # --- New Custom Test Case 1: 2017 AWD Petrol (Ref KSh 2.7M) ---
        # current_year_for_calc is already defined
        car1_year = 2017
        car1_car_age = float(current_year_for_calc - car1_year)
        car1_mileage = 150000.0
        car1_engine_cc = 2500.0
        car1_hp = 176.0
        car1_torque = 233.0
        car1_acceleration = 9.5
        car1_insurance = 108000.0

        car1_test_data = {
            'make_name_cleaned': 'toyota',
            'model_name_cleaned': 'rav4',
            'body_type_cleaned': 'suv',
            'car_age': car1_car_age,
            'car_age_squared': car1_car_age**2,
            'condition_clean': 'used',
            'mileage_num': car1_mileage,
            'mileage_log': np.log1p(car1_mileage),
            'mileage_per_year': car1_mileage / (car1_car_age + 1e-6) if car1_car_age > 0 else car1_mileage,
            'engine_size_cc_num': car1_engine_cc,
            'engine_size_cc_log': np.log1p(car1_engine_cc),
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': 'awd', # Given as AWD
            'horse_power_num': car1_hp,
            'horse_power_log': np.log1p(car1_hp) if car1_hp is not None else None,
            'torque_num': car1_torque,
            'torque_log': np.log1p(car1_torque) if car1_torque is not None else None,
            'acceleration_num': car1_acceleration,
            'seats_num': 5.0,
            'annual_insurance': car1_insurance,
            'power_per_cc': car1_hp / car1_engine_cc if car1_hp is not None and car1_engine_cc > 0 else 0.0,
            'mileage_per_cc': car1_mileage / car1_engine_cc if car1_engine_cc > 0 else 0.0,
            'is_luxury_make': 0,
            'make_model_cleaned': 'toyota_rav4',
            'usage_type_clean': 'Foreign Used' # From "cars in kenya - Japan"
        }
        test_cases.append({
            'description': "Custom 2017 Toyota RAV4 AWD (150k km, Ref 2.7M)",
            'data': car1_test_data,
            'expected_range': None
        })
        # --- End Custom Test Case 1 ---

        # --- New Custom Test Case 2: 2018 2WD Petrol (Ref KSh 1.05M) ---
        car2_year = 2018
        car2_car_age = float(current_year_for_calc - car2_year)
        car2_mileage = 89000.0
        car2_engine_cc = 1200.0
        car2_hp = 85.0
        car2_torque = 115.0
        car2_acceleration = 13.0
        car2_insurance = 52500.0

        car2_test_data = {
            'make_name_cleaned': 'toyota',
            'model_name_cleaned': 'passo',
            'body_type_cleaned': 'hatchback',
            'car_age': car2_car_age,
            'car_age_squared': car2_car_age**2,
            'condition_clean': 'used',
            'mileage_num': car2_mileage,
            'mileage_log': np.log1p(car2_mileage),
            'mileage_per_year': car2_mileage / (car2_car_age + 1e-6) if car2_car_age > 0 else car2_mileage,
            'engine_size_cc_num': car2_engine_cc,
            'engine_size_cc_log': np.log1p(car2_engine_cc),
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '2wd',
            'horse_power_num': car2_hp,
            'horse_power_log': np.log1p(car2_hp) if car2_hp is not None else None,
            'torque_num': car2_torque,
            'torque_log': np.log1p(car2_torque) if car2_torque is not None else None,
            'acceleration_num': car2_acceleration,
            'seats_num': 5.0,
            'annual_insurance': car2_insurance,
            'power_per_cc': car2_hp / car2_engine_cc if car2_hp is not None and car2_engine_cc > 0 else 0.0,
            'mileage_per_cc': car2_mileage / car2_engine_cc if car2_engine_cc > 0 else 0.0,
            'is_luxury_make': 0,
            'make_model_cleaned': 'toyota_passo',
            'usage_type_clean': 'Kenyan Used' # From "cars in kenya - Kenya"
        }
        test_cases.append({
            'description': "Custom 2018 Toyota Passo 2WD (89k km, Ref 1.05M)",
            'data': car2_test_data,
            'expected_range': None
        })
        # --- End Custom Test Case 2 ---

        # --- New Custom Test Case 3: 2017 4WD Diesel (Ref KSh 6.9M) ---
        car3_year = 2017
        car3_car_age = float(current_year_for_calc - car3_year)
        car3_mileage = 64000.0
        car3_engine_cc = 2000.0
        car3_hp = 180.0
        car3_torque = 430.0
        car3_acceleration = 9.0
        car3_insurance = 241500.0

        car3_test_data = {
            'make_name_cleaned': 'land rover', # Corrected make
            'model_name_cleaned': 'other_model',   # Discovery Sport not in list, using other_model
            'body_type_cleaned': 'suv',
            'car_age': car3_car_age,
            'car_age_squared': car3_car_age**2,
            'condition_clean': 'used',
            'mileage_num': car3_mileage,
            'mileage_log': np.log1p(car3_mileage),
            'mileage_per_year': car3_mileage / (car3_car_age + 1e-6) if car3_car_age > 0 else car3_mileage,
            'engine_size_cc_num': car3_engine_cc,
            'engine_size_cc_log': np.log1p(car3_engine_cc),
            'fuel_type_cleaned': 'diesel',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '4wd',
            'horse_power_num': car3_hp,
            'horse_power_log': np.log1p(car3_hp) if car3_hp is not None else None,
            'torque_num': car3_torque,
            'torque_log': np.log1p(car3_torque) if car3_torque is not None else None,
            'acceleration_num': car3_acceleration,
            'seats_num': 5.0,
            'annual_insurance': car3_insurance,
            'power_per_cc': car3_hp / car3_engine_cc if car3_hp is not None and car3_engine_cc > 0 else 0.0,
            'mileage_per_cc': car3_mileage / car3_engine_cc if car3_engine_cc > 0 else 0.0,
            'is_luxury_make': 1,
            'make_model_cleaned': 'land rover_other_model', # Corrected make_model
            'usage_type_clean': 'Foreign Used' # From "cars in kenya - Japan"
        }
        test_cases.append({
            'description': "Custom 2017 Land Rover SUV Diesel (64k km, Ref 6.9M)",
            'data': car3_test_data,
            'expected_range': None
        })
        # --- End Custom Test Case 3 ---

        # --- Test Cases from Training Data Snippet ---

        # Training Data Case 1: Mercedes E250 (Row 2 of snippet)
        # price_log: 15.424948470398375
        td1_actual_price = np.exp(15.424948470398375)
        td1_car_age = 9.0
        td1_mileage = 109000.0
        td1_engine_cc = 2000.0
        td1_hp = 211.0
        td1_torque = 350.0
        
        td1_data = {
            'make_name_cleaned': 'mercedes',
            'model_name_cleaned': 'e250', # Assuming e250 is a valid model, else map to other_model
            'body_type_cleaned': 'sedan',
            'car_age': td1_car_age,
            'car_age_squared': td1_car_age**2,
            'condition_clean': 'used', # Assuming 'Accident free' from CSV maps to 'used'
            'mileage_num': td1_mileage,
            'mileage_log': np.log1p(td1_mileage), # CSV has 11.599112335481124
            'mileage_per_year': 12111.10976543225,
            'engine_size_cc_num': td1_engine_cc,
            'engine_size_cc_log': np.log1p(td1_engine_cc), # CSV has 7.601402334583733
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '2wd',
            'horse_power_num': td1_hp,
            'horse_power_log': np.log1p(td1_hp), # CSV has 5.356586274672012
            'torque_num': td1_torque,
            'torque_log': np.log1p(td1_torque), # CSV has 5.860786223465865
            'acceleration_num': 7.6,
            'seats_num': 5.0,
            'annual_insurance': 208000.0, # From CSV
            'power_per_cc': 0.10549999994725, # From CSV
            'mileage_per_cc': 54.49999997275, # From CSV
            'is_luxury_make': 1, # Mercedes is luxury
            'make_model_cleaned': 'mercedes_e250', # CSV has mercedes_e250
            'usage_type_clean': 'Foreign Used' # From CSV
        }
        test_cases.append({
            'description': "Training Data: Mercedes E250 (car_age 9, Ref ~5.0M)",
            'data': td1_data,
            'expected_range': (td1_actual_price * 0.95, td1_actual_price * 1.05) # Tighter range for training data
        })

        # Training Data Case 2: Subaru Forester XT (Row 4 of snippet)
        # price_log: 13.527828485512494
        td2_actual_price = np.exp(13.527828485512494)
        td2_car_age = 22.0
        td2_mileage = 198000.0
        td2_engine_cc = 2000.0
        td2_hp = 200.0
        td2_torque = 300.0

        td2_data = {
            'make_name_cleaned': 'subaru',
            'model_name_cleaned': 'forester xt', # Check if 'forester xt' needs mapping
            'body_type_cleaned': 'suv',
            'car_age': td2_car_age,
            'car_age_squared': td2_car_age**2,
            'condition_clean': 'used',
            'mileage_num': td2_mileage,
            'mileage_log': np.log1p(td2_mileage), # CSV has 12.19602736016897
            'mileage_per_year': 8999.999590909109,
            'engine_size_cc_num': td2_engine_cc,
            'engine_size_cc_log': np.log1p(td2_engine_cc), # CSV has 7.601402334583733
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': 'awd',
            'horse_power_num': td2_hp,
            'horse_power_log': np.log1p(td2_hp), # CSV has 5.303304908059076
            'torque_num': td2_torque,
            'torque_log': np.log1p(td2_torque), # CSV has 5.707110264748875
            'acceleration_num': 7.7,
            'seats_num': 4.0, # CSV shows 4
            'annual_insurance': 30000.0, # From CSV
            'power_per_cc': 0.09999999994999999, # From CSV
            'mileage_per_cc': 98.9999999505, # From CSV
            'is_luxury_make': 0, # Subaru not typically luxury
            'make_model_cleaned': 'subaru_forester xt', # From CSV
            'usage_type_clean': 'Kenyan Used' # From CSV
        }
        test_cases.append({
            'description': "Training Data: Subaru Forester XT (car_age 22, Ref ~0.75M)",
            'data': td2_data,
            'expected_range': (td2_actual_price * 0.95, td2_actual_price * 1.05)
        })

        # Training Data Case 3: Volvo XC60 (Row 7 of snippet)
        # price_log: 14.508657738524219
        td3_actual_price = np.exp(14.508657738524219)
        td3_car_age = 10.0
        td3_mileage = 136000.0
        td3_engine_cc = 2000.0
        td3_hp = 240.0
        td3_torque = 320.0
        
        td3_data = {
            'make_name_cleaned': 'volvo',
            'model_name_cleaned': 'xc60',
            'body_type_cleaned': 'suv',
            'car_age': td3_car_age,
            'car_age_squared': td3_car_age**2,
            'condition_clean': 'used',
            'mileage_num': td3_mileage,
            'mileage_log': np.log1p(td3_mileage), # CSV has 11.820417517632333
            'mileage_per_year': 13599.998640000136,
            'engine_size_cc_num': td3_engine_cc,
            'engine_size_cc_log': np.log1p(td3_engine_cc), # CSV has 7.601402334583733
            'fuel_type_cleaned': 'petrol',
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': '2wd',
            'horse_power_num': td3_hp,
            'horse_power_log': np.log1p(td3_hp), # CSV has 5.484796933490655
            'torque_num': td3_torque,
            'torque_log': np.log1p(td3_torque), # CSV has 5.771441123130016
            'acceleration_num': 7.9,
            'seats_num': 5.0,
            'annual_insurance': 80000.0, # From CSV
            'power_per_cc': 0.11999999993999999, # From CSV
            'mileage_per_cc': 67.99999996599999, # From CSV
            'is_luxury_make': 1, # Volvo considered luxury/near-luxury
            'make_model_cleaned': 'volvo_xc60', # From CSV
            'usage_type_clean': 'Kenyan Used' # From CSV
        }
        test_cases.append({
            'description': "Training Data: Volvo XC60 (car_age 10, Ref ~2.0M)",
            'data': td3_data,
            'expected_range': (td3_actual_price * 0.95, td3_actual_price * 1.05)
        })
        
        # Training Data Case 4: Land Rover Range Rover Vogue (Row 13 of snippet)
        # price_log: 16.314813170178272
        td4_actual_price = np.exp(16.314813170178272)
        td4_car_age = 7.0
        td4_mileage = 48000.0
        td4_engine_cc = 3000.0
        td4_hp = 258.0 # From CSV
        td4_torque = 600.0 # From CSV

        td4_data = {
            'make_name_cleaned': 'land rover',
            'model_name_cleaned': 'range rover vogue', # Check mapping
            'body_type_cleaned': 'suv',
            'car_age': td4_car_age,
            'car_age_squared': td4_car_age**2,
            'condition_clean': 'used',
            'mileage_num': td4_mileage,
            'mileage_log': np.log1p(td4_mileage), # CSV has 10.778977123006351
            'mileage_per_year': 6857.14187755116, # From CSV
            'engine_size_cc_num': td4_engine_cc,
            'engine_size_cc_log': np.log1p(td4_engine_cc), # CSV has 8.006700845440367
            'fuel_type_cleaned': 'diesel', # From CSV
            'transmission_cleaned': 'automatic',
            'drive_type_cleaned': 'awd', # From CSV
            'horse_power_num': td4_hp,
            'horse_power_log': np.log1p(td4_hp), # CSV has 5.556828061699537
            'torque_num': td4_torque,
            'torque_log': np.log1p(td4_torque), # CSV has 6.398594934535208
            'acceleration_num': 7.9, # From CSV
            'seats_num': 5.0, # From CSV
            'annual_insurance': 486960.0, # From CSV
            'power_per_cc': 0.08599999997133334, # From CSV
            'mileage_per_cc': 15.999999994666668, # From CSV
            'is_luxury_make': 1, # Land Rover is luxury
            'make_model_cleaned': 'land rover_range rover vogue', # From CSV
            'usage_type_clean': 'Foreign Used' # From CSV
        }
        test_cases.append({
            'description': "Training Data: Land Rover Range Rover Vogue (car_age 7, Ref ~12.1M)",
            'data': td4_data,
            'expected_range': (td4_actual_price * 0.95, td4_actual_price * 1.05)
        })
        # --- End Test Cases from Training Data Snippet ---

        # --- Test Cases from cars_modeling_input.csv snippet (lines 420-430) ---
        # current_year_for_calc is already defined (datetime.now().year)

        # Case 1: Mazda Atenza 2017
        csv1_ref_price = 2659999.0
        csv1_year = 2017
        csv1_car_age = float(current_year_for_calc - csv1_year)
        csv1_mileage = 82000.0
        csv1_engine_cc = 2000.0
        csv1_hp = 158.0 # CSV shows 158.0 as 4th last column
        csv1_torque = 210.0 # CSV shows 210.0 as 5th last column
        csv1_accel = 8.0 # CSV shows 8.0 as 3rd last
        csv1_insurance = 106400.0 # 2nd last from snippet
        
        csv1_data = {
            'make_name_cleaned': 'mazda', 'model_name_cleaned': 'atenza',
            'body_type_cleaned': 'sedan', 'car_age': csv1_car_age, 'car_age_squared': csv1_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv1_mileage, 'mileage_log': np.log1p(csv1_mileage),
            'mileage_per_year': csv1_mileage / (csv1_car_age + 1e-6) if csv1_car_age > 0 else csv1_mileage,
            'engine_size_cc_num': csv1_engine_cc, 'engine_size_cc_log': np.log1p(csv1_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv1_hp, 'horse_power_log': np.log1p(csv1_hp),
            'torque_num': csv1_torque, 'torque_log': np.log1p(csv1_torque),
            'acceleration_num': csv1_accel, 'seats_num': 5.0, # CSV shows 5
            'annual_insurance': csv1_insurance,
            'power_per_cc': csv1_hp / csv1_engine_cc if csv1_hp and csv1_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv1_mileage / csv1_engine_cc if csv1_engine_cc > 0 else 0.0,
            'is_luxury_make': 0, 'make_model_cleaned': 'mazda_atenza', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Mazda Atenza 2017 (Ref ~2.66M)", 'data': csv1_data,
            'expected_range': (csv1_ref_price * 0.90, csv1_ref_price * 1.10)
        })

        # Case 2: Porsche Cayenne 2018 (1)
        csv2_ref_price = 8899999.0
        csv2_year = 2018
        csv2_car_age = float(current_year_for_calc - csv2_year)
        csv2_mileage = 44000.0
        csv2_engine_cc = 3000.0
        csv2_hp = 340.0 # CSV shows 340
        csv2_torque = 450.0 # CSV shows 450
        csv2_accel = 6.2
        csv2_insurance = 356000.0
        csv2_data = {
            'make_name_cleaned': 'porsche', 'model_name_cleaned': 'cayenne',
            'body_type_cleaned': 'suv', 'car_age': csv2_car_age, 'car_age_squared': csv2_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv2_mileage, 'mileage_log': np.log1p(csv2_mileage),
            'mileage_per_year': csv2_mileage / (csv2_car_age + 1e-6) if csv2_car_age > 0 else csv2_mileage,
            'engine_size_cc_num': csv2_engine_cc, 'engine_size_cc_log': np.log1p(csv2_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': 'awd',
            'horse_power_num': csv2_hp, 'horse_power_log': np.log1p(csv2_hp),
            'torque_num': csv2_torque, 'torque_log': np.log1p(csv2_torque),
            'acceleration_num': csv2_accel, 'seats_num': 5.0, 'annual_insurance': csv2_insurance,
            'power_per_cc': csv2_hp / csv2_engine_cc if csv2_hp and csv2_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv2_mileage / csv2_engine_cc if csv2_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'porsche_cayenne', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Porsche Cayenne 2018 (1) (Ref ~8.9M)", 'data': csv2_data,
            'expected_range': (csv2_ref_price * 0.90, csv2_ref_price * 1.10)
        })

        # Case 3: Porsche Cayenne 2018 (2)
        csv3_ref_price = 7999999.0
        csv3_year = 2018
        csv3_car_age = float(current_year_for_calc - csv3_year)
        csv3_mileage = 46000.0
        csv3_engine_cc = 3000.0
        csv3_hp = 340.0
        csv3_torque = 450.0
        csv3_accel = 6.2
        csv3_insurance = 320000.0
        csv3_data = {
            'make_name_cleaned': 'porsche', 'model_name_cleaned': 'cayenne',
            'body_type_cleaned': 'suv', 'car_age': csv3_car_age, 'car_age_squared': csv3_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv3_mileage, 'mileage_log': np.log1p(csv3_mileage),
            'mileage_per_year': csv3_mileage / (csv3_car_age + 1e-6) if csv3_car_age > 0 else csv3_mileage,
            'engine_size_cc_num': csv3_engine_cc, 'engine_size_cc_log': np.log1p(csv3_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': 'awd',
            'horse_power_num': csv3_hp, 'horse_power_log': np.log1p(csv3_hp),
            'torque_num': csv3_torque, 'torque_log': np.log1p(csv3_torque),
            'acceleration_num': csv3_accel, 'seats_num': 5.0, 'annual_insurance': csv3_insurance,
            'power_per_cc': csv3_hp / csv3_engine_cc if csv3_hp and csv3_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv3_mileage / csv3_engine_cc if csv3_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'porsche_cayenne', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Porsche Cayenne 2018 (2) (Ref ~8.0M)", 'data': csv3_data,
            'expected_range': (csv3_ref_price * 0.90, csv3_ref_price * 1.10)
        })

        # Case 4: Toyota Landcruiser Zx 2023
        csv4_ref_price = 18404000.0
        csv4_year = 2023
        csv4_car_age = float(current_year_for_calc - csv4_year)
        csv4_mileage = 7000.0
        csv4_engine_cc = 3300.0
        csv4_hp = 305.0 # CSV has 305
        csv4_torque = 700.0 # CSV has 700
        csv4_accel = 7.0
        csv4_insurance = 736160.0
        csv4_data = {
            'make_name_cleaned': 'toyota', 'model_name_cleaned': 'landcruiser', # ZX likely a trim
            'body_type_cleaned': 'suv', 'car_age': csv4_car_age, 'car_age_squared': csv4_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv4_mileage, 'mileage_log': np.log1p(csv4_mileage),
            'mileage_per_year': csv4_mileage / (csv4_car_age + 1e-6) if csv4_car_age > 0 else csv4_mileage,
            'engine_size_cc_num': csv4_engine_cc, 'engine_size_cc_log': np.log1p(csv4_engine_cc),
            'fuel_type_cleaned': 'diesel', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '4wd',
            'horse_power_num': csv4_hp, 'horse_power_log': np.log1p(csv4_hp),
            'torque_num': csv4_torque, 'torque_log': np.log1p(csv4_torque),
            'acceleration_num': csv4_accel, 'seats_num': 4.0, # CSV shows 4
            'annual_insurance': csv4_insurance,
            'power_per_cc': csv4_hp / csv4_engine_cc if csv4_hp and csv4_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv4_mileage / csv4_engine_cc if csv4_engine_cc > 0 else 0.0,
            'is_luxury_make': 0, # Toyota, though high end model
            'make_model_cleaned': 'toyota_landcruiser', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Toyota Landcruiser Zx 2023 (Ref ~18.4M)", 'data': csv4_data,
            'expected_range': (csv4_ref_price * 0.90, csv4_ref_price * 1.10)
        })

        # Case 5: Daihatsu Mira 2017
        csv5_ref_price = 879999.0
        csv5_year = 2017
        csv5_car_age = float(current_year_for_calc - csv5_year)
        csv5_mileage = 70000.0
        csv5_engine_cc = 660.0
        csv5_hp = 58.0 # CSV has 58
        csv5_torque = 68.0 # CSV has 68
        csv5_accel = 17.5
        csv5_insurance = 35200.0
        csv5_data = {
            'make_name_cleaned': 'daihatsu', 'model_name_cleaned': 'other_model', # Mira likely other_model
            'body_type_cleaned': 'hatchback', 'car_age': csv5_car_age, 'car_age_squared': csv5_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv5_mileage, 'mileage_log': np.log1p(csv5_mileage),
            'mileage_per_year': csv5_mileage / (csv5_car_age + 1e-6) if csv5_car_age > 0 else csv5_mileage,
            'engine_size_cc_num': csv5_engine_cc, 'engine_size_cc_log': np.log1p(csv5_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv5_hp, 'horse_power_log': np.log1p(csv5_hp),
            'torque_num': csv5_torque, 'torque_log': np.log1p(csv5_torque),
            'acceleration_num': csv5_accel, 'seats_num': 4.0, # CSV shows 4
            'annual_insurance': csv5_insurance,
            'power_per_cc': csv5_hp / csv5_engine_cc if csv5_hp and csv5_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv5_mileage / csv5_engine_cc if csv5_engine_cc > 0 else 0.0,
            'is_luxury_make': 0, 'make_model_cleaned': 'daihatsu_other_model', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Daihatsu Mira 2017 (Ref ~0.88M)", 'data': csv5_data,
            'expected_range': (csv5_ref_price * 0.90, csv5_ref_price * 1.10)
        })

        # Case 6: Nissan Xtrail 2017
        csv6_ref_price = 2749999.0
        csv6_year = 2017
        csv6_car_age = float(current_year_for_calc - csv6_year)
        csv6_mileage = 125000.0
        csv6_engine_cc = 2000.0
        csv6_hp = 155.0 # CSV has 155
        csv6_torque = 195.0 # CSV has 195
        csv6_accel = 10.0
        csv6_insurance = 110000.0
        csv6_data = {
            'make_name_cleaned': 'nissan', 'model_name_cleaned': 'xtrail',
            'body_type_cleaned': 'suv', 'car_age': csv6_car_age, 'car_age_squared': csv6_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv6_mileage, 'mileage_log': np.log1p(csv6_mileage),
            'mileage_per_year': csv6_mileage / (csv6_car_age + 1e-6) if csv6_car_age > 0 else csv6_mileage,
            'engine_size_cc_num': csv6_engine_cc, 'engine_size_cc_log': np.log1p(csv6_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv6_hp, 'horse_power_log': np.log1p(csv6_hp),
            'torque_num': csv6_torque, 'torque_log': np.log1p(csv6_torque),
            'acceleration_num': csv6_accel, 'seats_num': 5.0, 'annual_insurance': csv6_insurance,
            'power_per_cc': csv6_hp / csv6_engine_cc if csv6_hp and csv6_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv6_mileage / csv6_engine_cc if csv6_engine_cc > 0 else 0.0,
            'is_luxury_make': 0, 'make_model_cleaned': 'nissan_xtrail', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Nissan Xtrail 2017 (Ref ~2.75M)", 'data': csv6_data,
            'expected_range': (csv6_ref_price * 0.90, csv6_ref_price * 1.10)
        })
        
        # Case 7: Mercedes C180 2017
        csv7_ref_price = 4050000.0
        csv7_year = 2017
        csv7_car_age = float(current_year_for_calc - csv7_year)
        csv7_mileage = 32000.0
        csv7_engine_cc = 1600.0
        csv7_hp = 156.0 # CSV has 156
        csv7_torque = 250.0 # CSV has 250
        csv7_accel = 8.5
        csv7_insurance = 162000.0
        csv7_data = {
            'make_name_cleaned': 'mercedes', 'model_name_cleaned': 'c class', # C180 is a C Class
            'body_type_cleaned': 'coupe', 'car_age': csv7_car_age, 'car_age_squared': csv7_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv7_mileage, 'mileage_log': np.log1p(csv7_mileage),
            'mileage_per_year': csv7_mileage / (csv7_car_age + 1e-6) if csv7_car_age > 0 else csv7_mileage,
            'engine_size_cc_num': csv7_engine_cc, 'engine_size_cc_log': np.log1p(csv7_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv7_hp, 'horse_power_log': np.log1p(csv7_hp),
            'torque_num': csv7_torque, 'torque_log': np.log1p(csv7_torque),
            'acceleration_num': csv7_accel, 'seats_num': 3.0, # Coupe, often 2+2, CSV shows 3
            'annual_insurance': csv7_insurance,
            'power_per_cc': csv7_hp / csv7_engine_cc if csv7_hp and csv7_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv7_mileage / csv7_engine_cc if csv7_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'mercedes_c class', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Mercedes C180 2017 (Ref ~4.05M)", 'data': csv7_data,
            'expected_range': (csv7_ref_price * 0.90, csv7_ref_price * 1.10)
        })

        # Case 8: Mercedes C200 2017
        csv8_ref_price = 3950000.0
        csv8_year = 2017
        csv8_car_age = float(current_year_for_calc - csv8_year)
        csv8_mileage = 36000.0
        csv8_engine_cc = 2000.0
        csv8_hp = 184.0 # CSV has 184
        csv8_torque = 300.0 # CSV has 300
        csv8_accel = 7.2
        csv8_insurance = 158000.0
        csv8_data = {
            'make_name_cleaned': 'mercedes', 'model_name_cleaned': 'c class', # C200 is a C Class
            'body_type_cleaned': 'sedan', 'car_age': csv8_car_age, 'car_age_squared': csv8_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv8_mileage, 'mileage_log': np.log1p(csv8_mileage),
            'mileage_per_year': csv8_mileage / (csv8_car_age + 1e-6) if csv8_car_age > 0 else csv8_mileage,
            'engine_size_cc_num': csv8_engine_cc, 'engine_size_cc_log': np.log1p(csv8_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv8_hp, 'horse_power_log': np.log1p(csv8_hp),
            'torque_num': csv8_torque, 'torque_log': np.log1p(csv8_torque),
            'acceleration_num': csv8_accel, 'seats_num': 6.0, # CSV has 6, unusual for C Class Saloon
            'annual_insurance': csv8_insurance,
            'power_per_cc': csv8_hp / csv8_engine_cc if csv8_hp and csv8_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv8_mileage / csv8_engine_cc if csv8_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'mercedes_c class', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Mercedes C200 2017 (Ref ~3.95M)", 'data': csv8_data,
            'expected_range': (csv8_ref_price * 0.90, csv8_ref_price * 1.10)
        })

        # Case 9: Audi Q5 2017
        csv9_ref_price = 4599999.0
        csv9_year = 2017
        csv9_car_age = float(current_year_for_calc - csv9_year)
        csv9_mileage = 23000.0
        csv9_engine_cc = 2000.0
        csv9_hp = 220.0 # CSV has 220
        csv9_torque = 350.0 # CSV has 350
        csv9_accel = 7.1
        csv9_insurance = 184000.0
        csv9_data = {
            'make_name_cleaned': 'audi', 'model_name_cleaned': 'q5',
            'body_type_cleaned': 'suv', 'car_age': csv9_car_age, 'car_age_squared': csv9_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv9_mileage, 'mileage_log': np.log1p(csv9_mileage),
            'mileage_per_year': csv9_mileage / (csv9_car_age + 1e-6) if csv9_car_age > 0 else csv9_mileage,
            'engine_size_cc_num': csv9_engine_cc, 'engine_size_cc_log': np.log1p(csv9_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': 'awd',
            'horse_power_num': csv9_hp, 'horse_power_log': np.log1p(csv9_hp),
            'torque_num': csv9_torque, 'torque_log': np.log1p(csv9_torque),
            'acceleration_num': csv9_accel, 'seats_num': 4.0, # CSV shows 4
            'annual_insurance': csv9_insurance,
            'power_per_cc': csv9_hp / csv9_engine_cc if csv9_hp and csv9_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv9_mileage / csv9_engine_cc if csv9_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'audi_q5', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Audi Q5 2017 (Ref ~4.6M)", 'data': csv9_data,
            'expected_range': (csv9_ref_price * 0.90, csv9_ref_price * 1.10)
        })

        # Case 10: Land Rover Range Rover Sport 2019
        csv10_ref_price = 12100000.0
        csv10_year = 2019
        csv10_car_age = float(current_year_for_calc - csv10_year)
        csv10_mileage = 47000.0
        csv10_engine_cc = 3000.0
        csv10_hp = 306.0 # CSV has 306
        csv10_torque = 700.0 # CSV has 700
        csv10_accel = 7.1
        csv10_insurance = 484000.0
        csv10_data = {
            'make_name_cleaned': 'land rover', 'model_name_cleaned': 'range rover sport',
            'body_type_cleaned': 'suv', 'car_age': csv10_car_age, 'car_age_squared': csv10_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv10_mileage, 'mileage_log': np.log1p(csv10_mileage),
            'mileage_per_year': csv10_mileage / (csv10_car_age + 1e-6) if csv10_car_age > 0 else csv10_mileage,
            'engine_size_cc_num': csv10_engine_cc, 'engine_size_cc_log': np.log1p(csv10_engine_cc),
            'fuel_type_cleaned': 'petrol', 'transmission_cleaned': 'automatic', 'drive_type_cleaned': 'awd',
            'horse_power_num': csv10_hp, 'horse_power_log': np.log1p(csv10_hp),
            'torque_num': csv10_torque, 'torque_log': np.log1p(csv10_torque),
            'acceleration_num': csv10_accel, 'seats_num': 4.0, # CSV shows 4
            'annual_insurance': csv10_insurance,
            'power_per_cc': csv10_hp / csv10_engine_cc if csv10_hp and csv10_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv10_mileage / csv10_engine_cc if csv10_engine_cc > 0 else 0.0,
            'is_luxury_make': 1, 'make_model_cleaned': 'land rover_range rover sport', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Land Rover Range Rover Sport 2019 (Ref ~12.1M)", 'data': csv10_data,
            'expected_range': (csv10_ref_price * 0.90, csv10_ref_price * 1.10)
        })

        # Case 11: Honda Vezel 2017
        csv11_ref_price = 2649999.0
        csv11_year = 2017
        csv11_car_age = float(current_year_for_calc - csv11_year)
        csv11_mileage = 96000.0
        csv11_engine_cc = 1500.0
        csv11_hp = 130.0 # CSV has 130
        csv11_torque = 165.0 # CSV has 165
        csv11_accel = 8.5
        csv11_insurance = 106000.0
        csv11_data = {
            'make_name_cleaned': 'honda', 'model_name_cleaned': 'vezel',
            'body_type_cleaned': 'suv', 'car_age': csv11_car_age, 'car_age_squared': csv11_car_age**2,
            'condition_clean': 'used', 'mileage_num': csv11_mileage, 'mileage_log': np.log1p(csv11_mileage),
            'mileage_per_year': csv11_mileage / (csv11_car_age + 1e-6) if csv11_car_age > 0 else csv11_mileage,
            'engine_size_cc_num': csv11_engine_cc, 'engine_size_cc_log': np.log1p(csv11_engine_cc),
            'fuel_type_cleaned': 'hybrid_petrol', # CSV has Petrol Hybrid
            'transmission_cleaned': 'automatic', 'drive_type_cleaned': '2wd',
            'horse_power_num': csv11_hp, 'horse_power_log': np.log1p(csv11_hp),
            'torque_num': csv11_torque, 'torque_log': np.log1p(csv11_torque),
            'acceleration_num': csv11_accel, 'seats_num': 9.0, # CSV has 9, very unusual for Vezel
            'annual_insurance': csv11_insurance,
            'power_per_cc': csv11_hp / csv11_engine_cc if csv11_hp and csv11_engine_cc > 0 else 0.0,
            'mileage_per_cc': csv11_mileage / csv11_engine_cc if csv11_engine_cc > 0 else 0.0,
            'is_luxury_make': 0, 'make_model_cleaned': 'honda_vezel', 'usage_type_clean': 'Foreign Used'
        }
        test_cases.append({
            'description': "CSV Snippet: Honda Vezel 2017 (Ref ~2.65M)", 'data': csv11_data,
            'expected_range': (csv11_ref_price * 0.90, csv11_ref_price * 1.10)
        })
        
        # --- End Test Cases from cars_modeling_input.csv snippet ---

        # Run test cases
        results = []
        for i, test_case in enumerate(test_cases):
            # Create DataFrame from test case data
            input_df = pd.DataFrame([test_case['data']])
            
            # Make prediction
            result = predictor.predict(input_df)
            
            # Check if prediction is within expected range
            predicted_price = result['predicted_price'][0]
            status = "PASSED"
            details = ""
            
            if test_case.get('expected_range'):
                min_price, max_price = test_case['expected_range']
                if not (min_price <= predicted_price <= max_price):
                    status = "FAILED"
                    details = f"Expected: KES {min_price:,.0f} - {max_price:,.0f}"
            
            # Format results
            results.append({
                'description': test_case['description'],
                'status': status,
                'predicted_price': predicted_price,
                'details': details
            })
            
            # Print detailed results
            print(f"\n--- Running Test Case: {test_case['description']} ---")
            print(f"Predicted Price: KES {predicted_price:,.0f}")
            if test_case.get('expected_range'):
                min_price, max_price = test_case['expected_range']
                print(f"Expected Range: KES {min_price:,.0f} - {max_price:,.0f}")
        
        # Display summary
        print("\n--- Test Summary ---")
        df_summary = pd.DataFrame(results)
        print(df_summary[['description', 'status', 'predicted_price', 'details']].to_string())
        
        # Check overall result
        all_passed = all(r['status'] == 'PASSED' for r in results)
        print(f"\nOverall Result: {'ALL TESTS PASSED!' if all_passed else 'SOME TESTS FAILED!'}")
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        raise

if __name__ == "__main__":
    main() 