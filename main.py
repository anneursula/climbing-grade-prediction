# main.py - Weighted accuracy version (comparison can be easily commented out)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from src.data.preprocessing import load_data, create_boulder_angle_dataframe
from src.data.analysis import create_data_profile, analyze_and_clean_data, compare_climbing_dataframes
from src.visualization.plots import plot_grade_distribution
from src.visualization.model_plot import plot_training_history
from src.models.preprocessing import create_train_test_split
from src.models.cnn_model import create_cnn_model

# Weighted metrics imports
from src.models.weighted_metrics import (
    map_quality_to_clean_data,
    update_cnn_model_with_weighted_metrics,
    plot_quality_weight_analysis,
    create_quality_weights
)

def main():
    DA = False # Set to True to run data analysis
    RUN_COMPARISON = False  # Set to True to compare standard vs weighted training

    # Check TensorFlow availability
    print(f"TensorFlow version: {tf.__version__}")

    hold_data_df = load_data("data/processed/kilter_holds_lookup.csv")
    hold_data_df = hold_data_df.set_index('ledPosition')
    
    # Load original data to get climb_stats
    print("Loading original data for quality extraction...")
    original_df = load_data("data/raw/climbs.csv")  # Adjust path as needed
    climb_stats = original_df['climb_stats']  # Extract the climb_stats column
    print(f"Loaded climb_stats for {len(climb_stats)} original boulders")
  
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
    
    # Extract quality ratings by mapping from original to clean data
    print("Extracting quality ratings...")
    quality_ratings = map_quality_to_clean_data(climb_stats, boulder_angles_df, original_df)
    print(f"Quality ratings - Mean: {quality_ratings.mean():.3f}, Std: {quality_ratings.std():.3f}")

    # Create train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df)
    
    # Extract quality for train/test sets
    train_quality = quality_ratings.loc[X_train.index]
    test_quality = quality_ratings.loc[X_test.index]
    print(f"Train quality mean: {train_quality.mean():.3f}, Test quality mean: {test_quality.mean():.3f}")
    
    # ==========================================================================
    # STANDARD MODEL TRAINING WITH WEIGHTED EVALUATION
    # ==========================================================================
    
    # Create and train CNN model if TensorFlow is available
    try:
        print("\n" + "="*60)
        print("TRAINING STANDARD MODEL WITH WEIGHTED EVALUATION")
        print("="*60)
        
        model, history, metrics = create_cnn_model(boulder_angles_df, hold_data_df, X_train, X_test, y_train, y_test)
        
        # Plot training history
        history_plot = plot_training_history(history)
        history_plot.savefig("reports/figures/model_training_history.png")
        
        # Perform weighted evaluation
        print("\nPerforming quality-weighted evaluation...")
        
        # Recreate test data for weighted evaluation
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
        print("Standard model saved successfully")
        
        print(f"\nStandard Model Files Generated:")
        print(f"  - reports/model_metrics.txt (standard metrics)")
        print(f"  - reports/weighted_model_metrics.txt (quality-weighted metrics)")
        print(f"  - reports/figures/quality_weight_analysis.png (quality analysis plots)")
        print(f"  - models/boulder_grade_cnn.h5 (trained model)")
        
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        print("Skipping model training. Make sure TensorFlow is installed correctly.")
        return
    
    # ==========================================================================
    # OPTIONAL: COMPARISON WITH WEIGHTED TRAINING
    # ==========================================================================
    
    if RUN_COMPARISON:
        try:
            print("\n" + "="*60)
            print("TRAINING WEIGHTED MODEL FOR COMPARISON")
            print("="*60)
            
            # Import weighted training functions
            from src.models.weighted_training import compare_weighted_vs_unweighted_training
            
            # Run comparison (this will train both standard and weighted models)
            comparison_results = compare_weighted_vs_unweighted_training(
                boulder_angles_df, hold_data_df, X_train, X_test, y_train, y_test
            )
            
            # Save comparison results
            comparison_df = pd.DataFrame({
                'Metric': ['MAE', 'RMSE', 'R²', 'Accuracy_Exact', 'Accuracy_PM1', 
                          'Weighted_Accuracy_Exact', 'Weighted_Accuracy_PM1'],
                'Standard_Model': [
                    comparison_results['standard_model']['metrics']['mae'],
                    comparison_results['standard_model']['metrics']['rmse'],
                    comparison_results['standard_model']['metrics']['r2'],
                    comparison_results['standard_model']['metrics']['accuracy_exact'],
                    comparison_results['standard_model']['metrics']['accuracy_pm1'],
                    comparison_results['standard_model']['weighted_metrics']['weighted_accuracy_exact'],
                    comparison_results['standard_model']['weighted_metrics']['weighted_accuracy_pm1']
                ],
                'Weighted_Model': [
                    comparison_results['weighted_model']['metrics']['mae'],
                    comparison_results['weighted_model']['metrics']['rmse'],
                    comparison_results['weighted_model']['metrics']['r2'],
                    comparison_results['weighted_model']['metrics']['accuracy_exact'],
                    comparison_results['weighted_model']['metrics']['accuracy_pm1'],
                    comparison_results['weighted_model']['weighted_metrics']['weighted_accuracy_exact'],
                    comparison_results['weighted_model']['weighted_metrics']['weighted_accuracy_pm1']
                ]
            })
            
            comparison_df['Improvement'] = comparison_df['Weighted_Model'] - comparison_df['Standard_Model']
            comparison_df['Improvement_Percent'] = (comparison_df['Improvement'] / comparison_df['Standard_Model']) * 100
            
            # Save comparison
            comparison_df.to_csv("reports/training_method_comparison.csv", index=False)
            
            # Save both models
            comparison_results['standard_model']['model'].save("models/standard_boulder_cnn.h5")
            comparison_results['weighted_model']['model'].save("models/weighted_trained_boulder_cnn.h5")
            
            print(f"\nComparison Files Generated:")
            print(f"  - reports/training_method_comparison.csv (detailed comparison)")
            print(f"  - models/standard_boulder_cnn.h5 (standard trained model)")
            print(f"  - models/weighted_trained_boulder_cnn.h5 (quality-weighted trained model)")
            
            # Print summary
            print(f"\nCOMPARISON SUMMARY:")
            print(f"Standard vs Weighted Training Impact:")
            for _, row in comparison_df.iterrows():
                if abs(row['Improvement']) > 0.001:  # Only show meaningful differences
                    print(f"  {row['Metric']}: {row['Improvement']:+.4f} ({row['Improvement_Percent']:+.1f}%)")
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            import traceback
            traceback.print_exc()
            print("Comparison failed, but standard model training completed successfully.")
    
    else:
        print(f"\n(Comparison skipped - set RUN_COMPARISON=True to enable)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()