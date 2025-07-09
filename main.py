# main.py

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from src.data.preprocessing import load_data, create_boulder_angle_dataframe
from src.data.analysis import create_data_profile, analyze_and_clean_data, compare_climbing_dataframes
from src.visualization.plots import plot_grade_distribution
from src.visualization.model_plot import plot_training_history
from src.models.preprocessing import create_train_test_split
from src.models.cnn_model import create_cnn_model
from src.models.weighted_metrics import (
    extract_quality_ratings, 
    update_cnn_model_with_weighted_metrics,
    plot_quality_weight_analysis,
    create_quality_weights )
 
def main():
    DA = False # Set to True to run data analysis

    # Check TensorFlow availability
    print(f"TensorFlow version: {tf.__version__}")

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

    else: #else, load the already cleaned dataset
        print("Skipping data analysis. Loading pre-cleaned dataset.")
        boulder_angles_df = load_data("data/processed/full_clean_dataset.csv")    
    
    
    # new - for weighted accuracy und so 
    print("Extracting quality ratings...")
    quality_ratings = extract_quality_ratings(boulder_angles_df)
    print(f"Quality ratings - Mean: {quality_ratings.mean():.3f}, Std: {quality_ratings.std():.3f}")

    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df)
    
    # print quality ratings 
    train_quality = quality_ratings.loc[X_train.index]
    test_quality = quality_ratings.loc[X_test.index]
    print(f"Train quality mean: {train_quality.mean():.3f}, Test quality mean: {test_quality.mean():.3f}")
    
    # Create and train CNN model if TensorFlow is available
    try:
        model, history, metrics = create_cnn_model(boulder_angles_df, hold_data_df, X_train, X_test, y_train, y_test)
        
        # Plot training history
        history_plot = plot_training_history(history)
        history_plot.savefig("reports/figures/model_training_history.png")
        
                # Perform weighted evaluation
        print("\nPerforming quality-weighted evaluation...")
        
        # Recreate test data for weighted evaluation (mirrors CNN model process)
        from src.models.cnn_model import create_multichannel_grid, create_hold_feature_vector
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = ['angle', 'hold_count', 'ascents']
        scaler = StandardScaler()
        
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])
        
        # Create test data
        test_grids = np.array([create_multichannel_grid(p, hold_data_df) for p in X_test['placements']])
        test_hold_features = np.array([create_hold_feature_vector(p, hold_data_df) for p in X_test['placements']])
        test_features = np.hstack((X_test_scaled[feature_cols].values, test_hold_features))
        
        # Evaluate with weighted metrics
        weighted_results = update_cnn_model_with_weighted_metrics(
            model, test_grids, test_features, y_test, test_quality
        )
        
        # Create quality analysis plots
        predictions = model.predict([test_grids, test_features]).flatten()
        plot_quality_weight_analysis(
            test_quality, 
            create_quality_weights(test_quality, weight_function='exponential'),
            y_test, 
            predictions,
            save_path="reports/figures/quality_weight_analysis.png"
        )
        
        # Save both standard and weighted model metrics
        with open("reports/model_metrics.txt", "w") as f:
            f.write("CNN Model Evaluation Metrics\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        # Save weighted metrics
        with open("reports/weighted_model_metrics.txt", "w") as f:
            f.write("CNN Model Evaluation with Quality Weighting\n")
            f.write("=" * 45 + "\n\n")
            
            f.write("Standard Metrics:\n")
            f.write("-" * 16 + "\n")
            for key in ['mae', 'rmse', 'r2', 'accuracy_exact', 'accuracy_pm1']:
                if key in weighted_results:
                    f.write(f"{key}: {weighted_results[key]:.4f}\n")
            
            f.write("\nQuality-Weighted Metrics:\n")
            f.write("-" * 24 + "\n")
            for key in ['weighted_mae', 'weighted_accuracy_exact', 'weighted_accuracy_pm1']:
                if key in weighted_results:
                    f.write(f"{key}: {weighted_results[key]:.4f}\n")
            
            f.write("\nImprovements:\n")
            f.write("-" * 12 + "\n")
            accuracy_improvement = weighted_results['weighted_accuracy_exact'] - weighted_results['accuracy_exact']
            mae_improvement = weighted_results['mae'] - weighted_results['weighted_mae']
            f.write(f"Accuracy improvement: {accuracy_improvement:+.4f}\n")
            f.write(f"MAE improvement: {mae_improvement:+.4f}\n")
                
        # Save model
        model.save("models/boulder_grade_cnn.h5")
        print("Model saved successfully")
        
        # ADD THIS PRINT STATEMENT
        print(f"\nFiles generated:")
        print(f"  - reports/model_metrics.txt (standard metrics)")
        print(f"  - reports/weighted_model_metrics.txt (quality-weighted metrics)")
        print(f"  - reports/figures/quality_weight_analysis.png (quality analysis plots)")
        
    except Exception as e:
        print(f"Error training model: {e}")
        print("Skipping model training. Make sure TensorFlow is installed correctly.")
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()