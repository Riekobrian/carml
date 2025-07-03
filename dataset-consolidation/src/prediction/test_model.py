import pandas as pd
import numpy as np
from predict import CarPricePredictor
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from pathlib import Path # Added for cleaner path handling

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_test_data():
    """
    Load the original feature-engineered data, perform a train-test split
    to get the consistent X_test_unprocessed and y_test_series.
    """
    try:
        # Path to the full feature-engineered data
        fe_data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'processed', '20250520_225504_cars_feature_engineered.csv'
        )
        
        logger.info(f"Loading full feature-engineered data from: {fe_data_path}")
        df_fe = pd.read_csv(fe_data_path)
        
        TARGET_COLUMN = 'price_log'
        if TARGET_COLUMN not in df_fe.columns:
            logger.error(f"Target column '{TARGET_COLUMN}' not found in {fe_data_path}. Exiting.")
            raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

        X_original_fe = df_fe.drop(columns=[TARGET_COLUMN])
        y_original_fe = df_fe[TARGET_COLUMN]

        logger.info(f"Original feature set X_original_fe shape: {X_original_fe.shape}")
        logger.info(f"Original target y_original_fe shape: {y_original_fe.shape}")

        # Perform 80/20 train-test split, same as in the notebook
        _, X_test_unprocessed, _, y_test_log_series = train_test_split(
            X_original_fe, y_original_fe, test_size=0.2, random_state=42
        )

        logger.info(f"X_test_unprocessed shape after split: {X_test_unprocessed.shape}")
        logger.info(f"y_test_log_series shape after split: {y_test_log_series.shape}")
        
        # Convert y_test from log scale to actual prices for evaluation
        actual_prices = np.exp(y_test_log_series.values)
        
        # The X_test_unprocessed is what predictor.predict() expects as input_data
        # It should not have prefixes; the predictor's preprocessor will handle that.
        return X_test_unprocessed, actual_prices
        
    except Exception as e:
        logger.error(f"Error loading and splitting feature-engineered data: {e}")
        raise

def analyze_predictions(actual_prices, predicted_prices):
    """
    Analyze the prediction results and calculate error metrics.
    """
    # Calculate metrics
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mse = mean_squared_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_prices, predicted_prices)
    
    # Calculate percentage errors
    percentage_errors = np.abs(actual_prices - predicted_prices) / actual_prices * 100
    
    # Calculate additional statistics
    mean_pe = np.mean(percentage_errors)
    median_pe = np.median(percentage_errors)
    std_pe = np.std(percentage_errors)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'percentage_errors': percentage_errors,
        'mean_percentage_error': mean_pe,
        'median_percentage_error': median_pe,
        'std_percentage_error': std_pe
    }

def plot_results(actual_prices, predicted_prices, save_dir=None):
    """
    Create visualizations of the prediction results.
    """
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Actual vs Predicted Plot
    plt.subplot(2, 2, 1)
    plt.scatter(actual_prices, predicted_prices, alpha=0.5)
    plt.plot([min(actual_prices), max(actual_prices)], 
             [min(actual_prices), max(actual_prices)], 'r--')
    plt.xlabel('Actual Price (KES)')
    plt.ylabel('Predicted Price (KES)')
    plt.title('Actual vs Predicted Prices')
    
    # 2. Error Distribution Plot
    plt.subplot(2, 2, 2)
    errors = predicted_prices - actual_prices
    sns.histplot(errors, kde=True)
    plt.xlabel('Prediction Error (KES)')
    plt.title('Distribution of Prediction Errors')
    
    # 3. Percentage Error by Price Range
    plt.subplot(2, 2, 3)
    percentage_errors = np.abs(actual_prices - predicted_prices) / actual_prices * 100
    plt.scatter(actual_prices, percentage_errors, alpha=0.5)
    plt.xlabel('Actual Price (KES)')
    plt.ylabel('Percentage Error')
    plt.title('Percentage Error by Price Range')
    
    # 4. Residuals Plot
    plt.subplot(2, 2, 4)
    plt.scatter(predicted_prices, errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Price (KES)')
    plt.ylabel('Residuals (KES)')
    plt.title('Residuals vs Predicted Values')
    
    plt.tight_layout()
    
    # Save plots if directory is provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'model_performance_plots.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance plots to: {plot_path}")
    
    plt.close(fig)

def plot_sample_predictions_comparison(csv_path, num_samples=7, save_dir=None):
    """
    Plots a grouped bar chart comparing actual prices with predictions from all models
    for a sample of cars.

    Args:
        csv_path (str): Path to the predictions_comparison_table.csv file.
        num_samples (int): Number of sample cars to plot.
        save_dir (str): Directory to save the plot.
    """
    try:
        logger.info(f"Loading prediction comparison table from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Select a sample of cars
        sample_df = df.head(num_samples).copy() # Use .copy() to avoid SettingWithCopyWarning

        # Identify price columns (Actual and all predictions)
        price_cols = [col for col in sample_df.columns if 'Actual_Price' in col or 'Prediction' in col]
        
        # Clean KES and comma formatting, convert to numeric
        for col in price_cols:
            if sample_df[col].dtype == 'object': # Check if the column needs cleaning
                sample_df[col] = sample_df[col].replace({'KES ': '', ',': ''}, regex=True).astype(float)

        # Create an ID for each car for plotting (e.g., from index or a feature)
        # If 'make_name_cleaned' and 'model_name_cleaned' are in df, use them for better labels
        if 'make_name_cleaned' in df.columns and 'model_name_cleaned' in df.columns:
            sample_df['Car_Identifier'] = df['make_name_cleaned'].astype(str) + " " + \
                                          df['model_name_cleaned'].astype(str) + " (ID: " + sample_df.index.astype(str) + ")"
        else:
            sample_df['Car_Identifier'] = "Car " + sample_df.index.astype(str)
        
        # Melt the DataFrame for easier plotting with seaborn/matplotlib
        plot_df = sample_df.melt(id_vars=['Car_Identifier'], value_vars=price_cols, 
                                 var_name='Model', value_name='Price (KES)')

        plt.figure(figsize=(max(15, num_samples * 2.5), 10))
        sns.barplot(x='Car_Identifier', y='Price (KES)', hue='Model', data=plot_df, palette='viridis')
        
        plt.title(f'Comparison of Predicted Prices for {num_samples} Sample Cars', fontsize=16)
        plt.xlabel('Car', fontsize=14)
        plt.ylabel('Price (KES)', fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(title='Model/Actual', fontsize=10, title_fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plot_path = os.path.join(save_dir, 'sample_predictions_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved sample predictions comparison plot to: {plot_path}")
        # plt.show() # Disabled to prevent blocking
        plt.close()

    except Exception as e:
        logger.error(f"Error generating sample predictions comparison plot: {e}")
        # Optionally re-raise or handle

def main():
    """
    Main function to run model testing.
    """
    try:
        logger.info("Starting model testing...")
        
        # Load actual test data (X_test_unprocessed, actual_prices_for_eval)
        input_data_for_prediction, actual_prices_for_eval = load_test_data()
        logger.info(f"Loaded {len(input_data_for_prediction)} test cases for prediction.")
        
        # Initialize predictor
        predictor = CarPricePredictor()
        
        # Make predictions
        # input_data_for_prediction is already the X_test_unprocessed DataFrame
        results = predictor.predict(input_data_for_prediction)
        predicted_prices = np.array(results['predicted_price'])
        # actual_prices_for_eval is already in the correct (non-log) scale
        
        # Analyze results
        analysis = analyze_predictions(actual_prices_for_eval, predicted_prices)
        
        # Print detailed results
        print("\n=== Model Testing Results ===")
        print(f"\nNumber of test cases: {len(input_data_for_prediction)}")
        print(f"\nOverall Metrics:")
        print(f"R² Score: {analysis['r2']:.4f}")
        print(f"Mean Absolute Error: KES {analysis['mae']:,.2f}")
        print(f"Root Mean Squared Error: KES {analysis['rmse']:,.2f}")
        print(f"Mean Percentage Error: {analysis['mean_percentage_error']:.2f}%")
        print(f"Median Percentage Error: {analysis['median_percentage_error']:.2f}%")
        print(f"Standard Deviation of Percentage Error: {analysis['std_percentage_error']:.2f}%")
        
        # Calculate error percentiles
        percentiles = np.percentile(analysis['percentage_errors'], [25, 50, 75, 90, 95])
        print("\nError Percentiles:")
        print(f"25th percentile: {percentiles[0]:.2f}%")
        print(f"50th percentile: {percentiles[1]:.2f}%")
        print(f"75th percentile: {percentiles[2]:.2f}%")
        print(f"90th percentile: {percentiles[3]:.2f}%")
        print(f"95th percentile: {percentiles[4]:.2f}%")
        
        # Create visualization directory
        viz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
        
        # Plot results
        plot_results(actual_prices_for_eval, predicted_prices, save_dir=viz_dir)
        
        # --- Add predictions from individual base models ---
        logger.info("\n=== Individual Base Model Predictions ===")
        
        base_model_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'models', 'base_models'
        )

        base_models_to_test = {
            "Random Forest": os.path.join(base_model_dir, "20250520_151513_Random_Forest_tuned.joblib"),
            "Gradient Boosting": os.path.join(base_model_dir, "20250520_152843_Gradient_Boosting_tuned.joblib"),
            "K-Nearest Neighbors": os.path.join(base_model_dir, "20250520_153502_K-Nearest_Neighbors_tuned.joblib"),
            "Support Vector Regressor": os.path.join(base_model_dir, "20250520_165502_Support_Vector_Regressor_tuned.joblib")
        }
        
        predictions_table = pd.DataFrame({
            'Actual_Price': actual_prices_for_eval,
            'Stacked_Model_Prediction': predicted_prices
        })

        # Ensure the index is reset if it's not already a simple range index for consistent joining
        input_data_for_prediction_indexed = input_data_for_prediction.reset_index(drop=True)
        predictions_table.index = input_data_for_prediction_indexed.index


        for model_name, model_path_abs in base_models_to_test.items():
            logger.info(f"Predicting with: {model_name} from {model_path_abs}")
            try:
                # The input_data_for_prediction is already X_test_unprocessed
                base_model_results = predictor.predict_with_model(input_data_for_prediction_indexed, model_path_abs)
                base_predicted_prices = np.array(base_model_results['predicted_price'])
                predictions_table[f'{model_name}_Prediction'] = base_predicted_prices
                
                # Optionally, calculate and print metrics for each base model
                base_analysis = analyze_predictions(actual_prices_for_eval, base_predicted_prices)
                print(f"\n--- Metrics for {model_name} ---")
                print(f"R² Score: {base_analysis['r2']:.4f}")
                print(f"Mean Absolute Error: KES {base_analysis['mae']:,.2f}")
                print(f"Median Percentage Error: {base_analysis['median_percentage_error']:.2f}%")

            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                predictions_table[f'{model_name}_Prediction'] = np.nan # Add NaN if prediction fails

        print("\n\n=== Comparative Prediction Table (First 20 Test Cases) ===")
        # Reset index of actual_prices_for_eval if it's a Series with a different index
        # and predicted_prices is a numpy array, to align them correctly in the DataFrame.
        # However, actual_prices_for_eval and predicted_prices are already aligned from earlier.
        # The input_data_for_prediction is used to get original features for context if needed,
        # ensure its index aligns if we were to join with it.
        
        # Display relevant input features alongside predictions for context
        # We use input_data_for_prediction_indexed which has a simple range index
        # Let's pick a few key raw features that are human-readable
        contextual_features = ['make_name_cleaned', 'model_name_cleaned', 'car_age', 'mileage_num', 'engine_size_cc_num']
        display_df = input_data_for_prediction_indexed[contextual_features].copy()
        
        # Merge with predictions_table using the common range index
        display_df = display_df.join(predictions_table)
        
        # Format KES columns
        price_cols_to_format = [col for col in display_df.columns if 'Price' in col or 'Prediction' in col]
        for col in price_cols_to_format:
            # Ensure the column is numeric before attempting to format it
            # This check might be redundant if predictions_table was correctly populated with numbers
            # but good for safety if NaNs were strings or something unexpected.
            if pd.api.types.is_numeric_dtype(display_df[col]):
                 display_df[col] = display_df[col].apply(lambda x: f"KES {x:,.0f}" if pd.notnull(x) else "N/A")
            elif display_df[col].dtype == 'object': # If already string (e.g. due to N/A)
                 # this case might not be needed if formatting is the last step
                 pass # Already formatted or N/A

        print(display_df.head(20).to_string())
        
        # Save the full table for inspection
        csv_save_path = Path(viz_dir) / 'predictions_comparison_table.csv'
        # Save the table with original numeric values for prices before formatting for display_df
        # This means we need to save `predictions_table` joined with contextual features
        
        # Create the full table with numeric predictions for saving
        full_numeric_table = input_data_for_prediction_indexed[contextual_features].copy()
        full_numeric_table = full_numeric_table.join(predictions_table)
        full_numeric_table.to_csv(csv_save_path, index=False)
        logger.info(f"Saved full prediction comparison table (numeric) to: {csv_save_path}")

        # --- Plot sample predictions comparison ---
        logger.info("\n=== Generating Sample Predictions Comparison Plot ===")
        plot_sample_predictions_comparison(csv_path=str(csv_save_path), num_samples=7, save_dir=viz_dir)

        logger.info("Model testing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during model testing: {e}")
        raise

if __name__ == "__main__":
    main() 