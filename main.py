# main.py - Clean version using pre-processed quality data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

from src.data.preprocessing import load_data, create_boulder_angle_dataframe
from src.data.analysis import create_data_profile, analyze_and_clean_data, compare_climbing_dataframes
from src.visualization.plots import plot_grade_distribution
from src.visualization.model_plot import plot_training_history
from src.models.preprocessing import create_train_test_split
from src.models.cnn_model import create_cnn_model

def main():
    DA = False # Set to True to run data analysis

    # Check TensorFlow availability
    print(f"TensorFlow version: {tf.__version__}")

    # Ensure output directories exist
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Load hold data
    hold_data_df = load_data("data/processed/kilter_holds_lookup.csv")
    hold_data_df = hold_data_df.set_index('ledPosition')
  
    if DA is True: #only run the data analysis if 'DA' is set true
        # Load the dataset
        data_path = "data/raw/climbs.csv"
        df = load_data(data_path)

        if df is None:
            print("Failed to load data.")
            return
        
        # Create a data profile
        profile = create_data_profile(df)
        
        # Plot grade distribution
        plot_result = plot_grade_distribution(profile, 
                                            output_file="reports/figures/grade_distribution.png", 
                                            most_popular_only=True)
        
        # Clean the data
        cleaned_df, profile, stats = analyze_and_clean_data(df, 
                                                        upper_percentile=95, 
                                                        save_cleaned=True, 
                                                        output_folder="data/processed/")
        
        # Create boulder angle dataframe for more detailed analysis
        boulder_angles_df = create_boulder_angle_dataframe(cleaned_df, min_ascents=2)
        
        # Compare original and cleaned datasets
        comparison = compare_climbing_dataframes(df, cleaned_df, 
                                                df1_name="Original", 
                                                df2_name="Cleaned", 
                                                plot=True)

    else: #else, load the already cleaned dataset with quality ratings
        print("Skipping data analysis. Loading pre-cleaned dataset with quality ratings.")
        boulder_angles_df = load_data("data/processed/cleaner_quality_data.csv")

        # Verify quality ratings are present
        if 'quality_average' in boulder_angles_df.columns:
            quality_count = boulder_angles_df['quality_average'].notna().sum()
            print(f"Loaded dataset with quality ratings: {quality_count}/{len(boulder_angles_df)} records have quality data")
        else:
            print("Error: Quality ratings not found. Please run add_quality_ratings.py first.")
            return

    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df)
    
    # Create and train CNN model with specified loss function
    try:
        # SPECIFY WHICH LOSS TO USE HERE:
        loss_to_use = "custom_loss"  # Change this to "mean_squared_error", "mean_absolute_error", etc.
        
        print(f"\nTraining CNN model with {loss_to_use} loss...")
        model, history, metrics, loss_display_name = create_cnn_model(
            boulder_angles_df, hold_data_df, X_train, X_test, y_train, y_test, 
            loss_name=loss_to_use
        )
        
        # Create file names that include the loss function name (sanitized for filenames)
        safe_loss_name = loss_display_name.replace(' ', '_').replace('/', '_').lower()
        
        # Plot training history
        history_plot = plot_training_history(history)
        history_plot.savefig(f"reports/figures/model_training_history_{safe_loss_name}.png")
        plt.close()  # Close the plot to free memory
        
        # Save model metrics with loss name in filename
        with open(f"reports/model_metrics_{safe_loss_name}.txt", "w") as f:
            f.write(f"CNN Model Evaluation Metrics - {loss_display_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Loss Function Used: {loss_display_name}\n")
            f.write("-" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
                
        # Save model with loss name in filename
        model.save(f"models/boulder_grade_cnn_{safe_loss_name}.h5")
        
        print(f"\nModel training completed successfully using {loss_display_name}!")
        print("Files saved:")
        print(f"  - models/boulder_grade_cnn_{safe_loss_name}.h5 (trained model)")
        print(f"  - reports/model_metrics_{safe_loss_name}.txt (evaluation metrics)")
        print(f"  - reports/figures/model_training_history_{safe_loss_name}.png (training plots)")
        print(f"  - reports/figures/confusion_matrix_{safe_loss_name}.png (grade confusion matrix)")
        
        # Print a summary of the results
        print(f"\n" + "="*60)
        print(f"RESULTS SUMMARY - {loss_display_name}")
        print(f"="*60)
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"V-grade Exact Accuracy: {metrics['v_grade_accuracy']:.4f} ({metrics['v_grade_accuracy']*100:.1f}%)")
        print(f"V-grade ±1 Accuracy: {metrics['v_grade_accuracy_one']:.4f} ({metrics['v_grade_accuracy_one']*100:.1f}%)")
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping model training. Make sure TensorFlow is installed correctly.")
        return
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()