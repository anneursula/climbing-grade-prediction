#!/usr/bin/env python3
"""
Create a small sample of the climbing data for testing
"""

import pandas as pd
import os

def create_sample_dataset(input_file="data/processed/full_clean_dataset.csv", 
                         output_file="data/raw/climbs_sample.csv", 
                         sample_size=500):
    """
    Create a small sample of the climbing dataset
    
    Parameters:
    -----------
    input_file : str
        Path to the full dataset
    output_file : str
        Path to save the sample
    sample_size : int
        Number of rows to sample
    """
    
    if not os.path.exists(input_file):
        print(f"❌ Input file not found: {input_file}")
        print("Please make sure your climbs.csv file is in the data/raw/ directory")
        return False
    
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"✓ Loaded {len(df)} total records")
    
    # Sample the data
    if len(df) <= sample_size:
        print(f"Dataset is already small ({len(df)} records), using all data")
        sample_df = df
    else:
        print(f"Creating random sample of {sample_size} records...")
        sample_df = df.sample(n=sample_size, random_state=42)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save sample
    sample_df.to_csv(output_file, index=False)
    print(f"✓ Sample saved to {output_file}")
    print(f"Sample contains {len(sample_df)} records")
    
    # Show some basic stats
    if 'climb_stats' in sample_df.columns:
        print(f"\nSample statistics:")
        print(f"- Unique route names: {sample_df['name'].nunique() if 'name' in sample_df.columns else 'N/A'}")
        print(f"- Has climb_stats: {sample_df['climb_stats'].notna().sum()}")
        print(f"- Has placements: {sample_df['placements'].notna().sum() if 'placements' in sample_df.columns else 'N/A'}")
    
    return True

if __name__ == "__main__":
    # Create a small sample for testing
    success = create_sample_dataset(
        input_file="data/raw/climbs.csv",
        output_file="data/raw/climbs_sample.csv",
        sample_size=500  # Adjust this number as needed
    )
    
    if success:
        print("\n✓ Sample dataset created!")
        print("Now you can run your existing code with the sample dataset.")
        print("\nTo use the sample:")
        print("1. Modify main.py to use 'data/raw/climbs_sample.csv'")
        print("2. Or run: python main.py")
    else:
        print("❌ Failed to create sample dataset")