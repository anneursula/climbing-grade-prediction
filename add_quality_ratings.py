# add_quality_ratings.py

import pandas as pd
import ast
import os
from src.data.preprocessing import parse_climb_stats, find_most_popular_setup

def add_quality_ratings_to_dataset(input_file, output_file):
    """
    Add quality ratings to the clean dataset and save the result.
    
    Parameters:
    -----------
    input_file : str
        Path to the input CSV file (e.g., "data/processed/full_clean_dataset.csv")
    output_file : str
        Path to save the output CSV file with quality ratings
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load the clean dataset
    df = pd.read_csv(input_file)
    
    print(f"Loaded {len(df)} records")
    
    # Check if quality_average already exists
    if 'quality_average' in df.columns:
        print("Quality ratings already exist in the dataset.")
        return df
    
    print("Adding quality ratings...")
    
    # We need the original climbs data to get quality ratings
    # First, let's try to load the original data
    original_climbs_path = "data/raw/climbs.csv"
    
    if not os.path.exists(original_climbs_path):
        print(f"Warning: Original climbs file not found at {original_climbs_path}")
        print("Attempting to extract quality from existing climb_stats if available...")
        
        # If we don't have the original file, try to extract from climb_stats in current data
        if 'climb_stats' not in df.columns:
            print("Error: No climb_stats column found. Cannot extract quality ratings.")
            return None
        
        quality_ratings = []
        
        for _, row in df.iterrows():
            # Find quality rating for this specific angle
            angle = row.get('angle')
            climb_stats_str = row.get('climb_stats', '[]')
            
            try:
                stats_list = parse_climb_stats(climb_stats_str)
                quality = None
                
                # Find the stats for this specific angle
                for stats in stats_list:
                    if stats.get('angle') == angle:
                        quality = stats.get('quality_average')
                        break
                
                quality_ratings.append(quality)
                
            except Exception as e:
                print(f"Error parsing climb_stats for row: {e}")
                quality_ratings.append(None)
        
        df['quality_average'] = quality_ratings
        
    else:
        print(f"Loading original climbs data from {original_climbs_path}...")
        original_df = pd.read_csv(original_climbs_path)
        
        # Create a mapping from name to climb_stats
        name_to_stats = {}
        for _, row in original_df.iterrows():
            name = row.get('name')
            climb_stats = row.get('climb_stats', '[]')
            if name:
                name_to_stats[name] = climb_stats
        
        print(f"Created mapping for {len(name_to_stats)} boulder problems")
        
        # Add quality ratings to the clean dataset
        quality_ratings = []
        
        for _, row in df.iterrows():
            name = row.get('name')
            angle = row.get('angle')
            
            if name in name_to_stats:
                climb_stats_str = name_to_stats[name]
                
                try:
                    stats_list = parse_climb_stats(climb_stats_str)
                    quality = None
                    
                    # Find the stats for this specific angle
                    for stats in stats_list:
                        if stats.get('angle') == angle:
                            quality = stats.get('quality_average')
                            break
                    
                    quality_ratings.append(quality)
                    
                except Exception as e:
                    print(f"Error parsing climb_stats for {name}: {e}")
                    quality_ratings.append(None)
            else:
                print(f"Warning: Could not find climb_stats for boulder '{name}'")
                quality_ratings.append(None)
        
        df['quality_average'] = quality_ratings
    
    # Print statistics about quality ratings
    quality_series = pd.Series(quality_ratings).dropna()
    if len(quality_series) > 0:
        print(f"\nQuality Rating Statistics:")
        print(f"  Records with quality ratings: {len(quality_series)} / {len(df)} ({len(quality_series)/len(df)*100:.1f}%)")
        print(f"  Mean quality: {quality_series.mean():.2f}")
        print(f"  Min quality: {quality_series.min():.2f}")
        print(f"  Max quality: {quality_series.max():.2f}")
        print(f"  Quality distribution:")
        
        # Show quality distribution in bins
        quality_bins = [0, 1, 2, 3, 4, 5]
        quality_hist = pd.cut(quality_series, bins=quality_bins, include_lowest=True).value_counts().sort_index()
        for bin_range, count in quality_hist.items():
            if count > 0:
                print(f"    {bin_range}: {count} problems ({count/len(quality_series)*100:.1f}%)")
    else:
        print("Warning: No quality ratings found in the data")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    
    # Save the enhanced dataset
    df.to_csv(output_file, index=False)
    print(f"\nEnhanced dataset saved to {output_file}")
    
    return df

def main():
    """Main function to process the dataset and add quality ratings"""
    
    input_file = "data/processed/full_clean_dataset.csv"
    output_file = "data/processed/clean_quality_data.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        print("Please make sure you have run the data cleaning process first.")
        return
    
    # Process the dataset
    enhanced_df = add_quality_ratings_to_dataset(input_file, output_file)
    
    if enhanced_df is not None:
        print("\nProcessing completed successfully!")
        print(f"Enhanced dataset contains {len(enhanced_df)} records with quality ratings")
        
        # Show sample of the enhanced data
        print("\nSample of enhanced data:")
        sample_cols = ['name', 'angle', 'grade', 'ascents', 'hold_count', 'quality_average']
        available_cols = [col for col in sample_cols if col in enhanced_df.columns]
        print(enhanced_df[available_cols].head(10).to_string(index=False))
    else:
        print("Processing failed. Please check the error messages above.")

if __name__ == "__main__":
    main()