import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os

# Import necessary functions
from src.models.preprocessing import create_train_test_split
from src.models.cnn_model import create_cnn_model
from src.visualization.model_plot import plot_training_history
from src.data.preprocessing import load_data, create_boulder_angle_dataframe

def main():
    # Check if TensorFlow is available
    print(f"TensorFlow version: {tf.__version__}")
    
    # Verify GPU availability for TensorFlow (if you have a GPU)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) detected: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    else:
        print("No GPU detected. Using CPU for computations (may be slower).")
    
    # Create directories for outputs if they don't exist
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    print("Loading processed data...")
    
    # Option 1: If you've already created the boulder_angles_df and saved it
    try:
        boulder_angles_df = pd.read_csv("data/processed/full_clean_dataset.csv")
        print(f"Loaded boulder angles dataframe with {len(boulder_angles_df)} rows")
    except FileNotFoundError:
        # Option 2: If you need to create the boulder_angles_df from cleaned data
        try:
            cleaned_df = pd.read_csv("data/processed/cleaned_kilterboard_data.csv")
            print(f"Loaded cleaned dataframe with {len(cleaned_df)} rows")
            
            print("Creating boulder angle dataframe...")
            boulder_angles_df = create_boulder_angle_dataframe(cleaned_df, min_ascents=2)
            
            # Save for future use
            boulder_angles_df.to_csv("data/processed/boulder_angles_df.csv", index=False)
            
        except FileNotFoundError:
            # Option 3: If you need to start from raw data
            print("Processed data not found. Starting from raw data...")
            raw_df = load_data("data/raw/climbs.csv")
            
            # You'd need to run the cleaning process here
            # For simplicity, let's assume this is implemented elsewhere
            print("Please run the data processing pipeline first.")
            return
    
    # Create train-test split
    print("Creating train-test split...")
    X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df)
    
    # Create and train CNN model
    print("Training CNN model...")
    try:
        model, history, metrics = create_cnn_model(boulder_angles_df, X_train, X_test, y_train, y_test)
        
        # Plot training history
        print("Plotting training history...")
        history_plot = plot_training_history(history)
        history_plot.savefig("reports/figures/model_training_history.png")
        
        # Save model metrics
        print("Saving model metrics...")
        with open("reports/model_metrics.txt", "w") as f:
            f.write("CNN Model Evaluation Metrics\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
                
        # Save model
        print("Saving model...")
        model.save("models/boulder_grade_cnn.h5")
        print("Model saved successfully to models/boulder_grade_cnn.h5")
        
    except Exception as e:
        print(f"Error training model: {e}")
        print("Make sure TensorFlow is installed correctly and all dependencies are available.")
    
    print("CNN training complete.")

if __name__ == "__main__":
    main()