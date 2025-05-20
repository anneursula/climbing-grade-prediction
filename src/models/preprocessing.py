# src/models/preprocessing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_train_test_split(df, test_size=0.2, random_state=42):
    """
    Create train and test sets for machine learning while ensuring that all entries
    with the same 'name' are always in the same set to prevent data leakage.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the boulder data with angle-specific entries
    test_size : float, optional
        The proportion of the dataset to include in the test split (default: 0.2)
    random_state : int, optional
        Controls the shuffling applied to the data (default: 42)

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - Features and target variables for training and testing
    """
    # Get unique boulder names
    unique_names = df['name'].unique()

    # Randomly select boulder names for train and test sets
    np.random.seed(random_state)
    test_names = np.random.choice(
        unique_names,
        size=int(len(unique_names) * test_size),
        replace=False
    )

    # Create masks for the splits
    test_mask = df['name'].isin(test_names)
    train_mask = ~test_mask

    # Split the data
    train_df = df[train_mask]
    test_df = df[test_mask]

    # Print split information
    print(f"Total dataset size: {len(df)} entries ({len(unique_names)} unique boulder problems)")
    print(f"Training set: {len(train_df)} entries ({train_df['name'].nunique()} unique boulder problems)")
    print(f"Test set: {len(test_df)} entries ({test_df['name'].nunique()} unique boulder problems)")

    # Check if the split matches the desired ratio approximately
    actual_test_size = len(test_df) / len(df)
    print(f"Requested test size: {test_size:.1%}, Actual test size: {actual_test_size:.1%}")

    # Assume 'grade' is the target variable - adjust if needed
    X_train = train_df.drop('grade', axis=1)
    y_train = train_df['grade']
    X_test = test_df.drop('grade', axis=1)
    y_test = test_df['grade']

    return X_train, X_test, y_train, y_test