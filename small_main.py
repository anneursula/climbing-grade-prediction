# small_main.py 

import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.abspath('.'))

# Only import TensorFlow if available, otherwise skip CNN
try:
    import tensorflow as tf
    tf_available = True
    print(f"TensorFlow version: {tf.__version__}")
except ImportError:
    tf_available = False
    print("TensorFlow not available - skipping CNN model")

# Try different import methods
try:
    # Method 1: Direct imports
    from src.data.preprocessing import load_data, create_boulder_angle_dataframe
    from src.data.analysis import create_data_profile, analyze_and_clean_data, compare_climbing_dataframes
    from src.visualization.plots import plot_grade_distribution
    print("✓ Imports successful with src.")
except ImportError as e1:
    print(f"Import method 1 failed: {e1}")
    try:
        # Method 2: Add src to path and import directly
        sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src'))
        
        from data.preprocessing import load_data, create_boulder_angle_dataframe
        from data.analysis import create_data_profile, analyze_and_clean_data, compare_climbing_dataframes
        from visualization.plots import plot_grade_distribution
        print("✓ Imports successful with direct path method")
    except ImportError as e2:
        print(f"Import method 2 failed: {e2}")
        print("❌ Could not import required modules")
        print("\nTroubleshooting steps:")
        print("1. Run: python fix_imports.py")
        print("2. Make sure all files exist in src/ subdirectories")
        print("3. Check that you're running from the project root directory")
        sys.exit(1)

# Only import CNN-related modules if TensorFlow is available
if tf_available:
    try:
        from visualization.model_plots import plot_training_history
        from models.preprocessing import create_train_test_split
        from models.cnn_model import create_cnn_model
    except ImportError:
        try:
            from src.visualization.model_plots import plot_training_history
            from src.models.preprocessing import create_train_test_split
            from src.models.cnn_model import create_cnn_model
        except ImportError as e:
            print(f"Warning: Could not import CNN modules: {e}")
            tf_available = False

def main():
    print("Starting Climbing Route Analysis (Small Dataset Version)...")
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)
    
    # Use sample dataset (create it first if it doesn't exist)
    data_path = "data/processed/full_clean_dataset.csv"
    full_data_path = "data/raw/climbs.csv"
    
    # Create sample if it doesn't exist but full dataset does
    if not os.path.exists(data_path) and os.path.exists(full_data_path):
        print("Creating sample dataset...")
        try:
            df_full = pd.read_csv(full_data_path)
            sample_size = min(500, len(df_full))  # Use smaller sample or full dataset if already small
            df_sample = df_full.sample(n=sample_size, random_state=42)
            df_sample.to_csv(data_path, index=False)
            print(f"✓ Created sample with {len(df_sample)} records")
        except Exception as e:
            print(f"Error creating sample: {e}")
            return
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found at {data_path}")
        print("Please make sure you have climbs.csv in data/raw/ directory")
        return
    
    # Load the dataset
    try:
        df = load_data(data_path)
        if df is None:
            print("Failed to load data.")
            return
        print(f"✓ Loaded {len(df)} climbing routes (sample dataset)")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Create a data profile
    print("\nCreating data profile...")
    try:
        profile = create_data_profile(df)
        if profile is None:
            print("Failed to create data profile")
            return
        print("✓ Data profile created")
    except Exception as e:
        print(f"Error creating data profile: {e}")
        return
    
    # Plot grade distribution
    print("\nPlotting grade distribution...")
    try:
        plot_result = plot_grade_distribution(profile, 
                                             output_file="reports/figures/grade_distribution_sample.png", 
                                             most_popular_only=True)
        print("✓ Grade distribution plot saved")
    except Exception as e:
        print(f"Warning: Could not create grade plot: {e}")
    
    # Clean the data
    print("\nCleaning data...")
    try:
        cleaned_df, profile, stats = analyze_and_clean_data(df, 
                                                           upper_percentile=95, 
                                                           save_cleaned=True, 
                                                           output_folder="data/processed/")
        
        if cleaned_df is None:
            print("Failed to clean data")
            return
        print(f"✓ Data cleaned: {len(cleaned_df)} records remain")
    except Exception as e:
        print(f"Error cleaning data: {e}")
        return
    
    # Create boulder angle dataframe for more detailed analysis
    print("\nCreating boulder angle dataframe...")
    try:
        boulder_angles_df = create_boulder_angle_dataframe(cleaned_df, min_ascents=1)  # Lower threshold for small dataset
        print(f"✓ Created {len(boulder_angles_df)} angle-specific entries")
    except Exception as e:
        print(f"Error creating boulder angle dataframe: {e}")
        return
    
    # Compare original and cleaned datasets
    print("\nComparing datasets...")
    try:
        comparison = compare_climbing_dataframes(df, cleaned_df, 
                                                df1_name="Original", 
                                                df2_name="Cleaned", 
                                                plot=True)
        print("✓ Dataset comparison complete")
    except Exception as e:
        print(f"Warning: Could not create comparison plots: {e}")
    
    # Only proceed with CNN if TensorFlow is available and we have enough data
    if tf_available and len(boulder_angles_df) >= 20:  # Need minimum data for train/test split
        print("\nPreparing for CNN training...")
        
        # Create train-test split
        print("Creating train-test split...")
        try:
            X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df, test_size=0.3)  # Larger test split for small data
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            if len(X_train) >= 10 and len(X_test) >= 5:  # Minimum viable sizes
                print("\nTraining CNN model (this may take a few minutes)...")
                
                try:
                    # Train the model
                    model, history, metrics = create_cnn_model(boulder_angles_df, X_train, X_test, y_train, y_test)
                    
                    print("\n" + "="*50)
                    print("🎉 MODEL TRAINING COMPLETE!")
                    print("="*50)
                    
                    # Display metrics
                    print("\nFinal Model Performance:")
                    for metric_name, value in metrics.items():
                        print(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
                    
                    # Plot training history
                    try:
                        history_plot = plot_training_history(history)
                        history_plot.savefig("reports/figures/model_training_history_sample.png", 
                                           dpi=300, bbox_inches='tight')
                        plt.close(history_plot)
                        print("✓ Training history plot saved")
                    except Exception as e:
                        print(f"Warning: Could not save training plot: {e}")
                    
                    # Save model metrics
                    with open("reports/model_metrics_sample.txt", "w") as f:
                        f.write("CNN Model Evaluation Metrics (Sample Dataset)\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Dataset size: {len(boulder_angles_df)} entries\n")
                        f.write(f"Training samples: {len(X_train)}\n")
                        f.write(f"Test samples: {len(X_test)}\n\n")
                        for key, value in metrics.items():
                            f.write(f"{key}: {value:.4f}\n")
                    
                    # Save model
                    model.save("models/boulder_grade_cnn_sample.h5")
                    print("✓ Model saved successfully")
                    
                except Exception as e:
                    print(f"❌ Error training model: {e}")
                    print("This might be normal for very small datasets")
            else:
                print("❌ Dataset too small for meaningful train/test split")
                print("Try increasing the sample size")
        
        except Exception as e:
            print(f"❌ Error in data preparation: {e}")
    
    elif not tf_available:
        print("\n⚠️  TensorFlow not available - CNN training skipped")
        print("Analysis completed without CNN training")
    
    else:
        print(f"\n⚠️  Dataset too small ({len(boulder_angles_df)} entries) for CNN training")
        print("Try increasing the sample size or using the full dataset")
    
    print(f"\nAnalysis complete! Check the reports/ directory for outputs.")

if __name__ == "__main__":
    main()