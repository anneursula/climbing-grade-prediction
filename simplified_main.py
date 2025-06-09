# simplified_main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models.cnn_model import create_cnn_model

def load_sample_data(file_path, sample_size=500):
    """Load and sample data from the cleaned dataset"""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Take random sample
    if len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows from {len(df)} total rows")
    else:
        df_sample = df
        print(f"Using all {len(df)} rows (less than requested sample size)")
    
    return df_sample

def simple_train_test_split(df, test_size=0.2):
    """Simple train/test split without name-based grouping"""
    # Prepare features and target
    X = df.drop('grade', axis=1)
    y = df['grade']
    
    # Simple random split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load sample data
    data_path = "data/processed/full_clean_dataset.csv"
    df = load_sample_data(data_path, sample_size=500)
    
    # Create train/test split
    X_train, X_test, y_train, y_test = simple_train_test_split(df)
    
    # Train CNN model using existing function
    print("\nTraining CNN model...")
    try:
        model, history, metrics = create_cnn_model(df, X_train, X_test, y_train, y_test)
        
        print("\nTraining completed!")
        print("Final metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
            
        # Save model
        model.save("models/simple_boulder_cnn.h5")
        print("\nModel saved to models/simple_boulder_cnn.h5")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()