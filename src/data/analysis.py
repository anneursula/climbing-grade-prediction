# src/data/analysis.py

import numpy as np
import pandas as pd
from collections import Counter
from ..features.grade_conversion import difficulty_to_vgrade
from .preprocessing import load_data, parse_climb_stats, find_most_popular_setup, count_holds

def create_data_profile(data_source):
    """Create a comprehensive data profile from the Kilterboard dataset"""
    df = load_data(data_source)
    if df is None:
        return None

    print(f"Dataset contains {len(df)} boulder problems")

    # 1. Count boulders per grade across all angles
    all_grades = []

    # Process each climb and its various setups
    for _, row in df.iterrows():
        stats_list = parse_climb_stats(row.get('climb_stats', []))

        for stats in stats_list:
            difficulty = stats.get('difficulty_average')
            if difficulty is not None:
                vgrade = difficulty_to_vgrade(difficulty)
                all_grades.append(vgrade)

    all_grades_count = Counter(all_grades)

    # Print results for all angles
    print("\n1. Boulder Count Per Grade (All Angles):")
    for grade in sorted(all_grades_count.keys(), key=lambda g: (g[0], int(g[1:]) if g[1:].isdigit() else 0)):
        print(f"  {grade}: {all_grades_count[grade]} problems")

    # 2. Count boulders per grade at most popular angle only
    popular_grades = []

    for _, row in df.iterrows():
        popular_setup = find_most_popular_setup(row.get('climb_stats', []))

        if popular_setup:
            difficulty = popular_setup.get('difficulty_average')
            if difficulty is not None:
                vgrade = difficulty_to_vgrade(difficulty)
                popular_grades.append(vgrade)

    popular_grades_count = Counter(popular_grades)

    # Print results for most popular angles
    print("\n2. Boulder Count Per Grade (Most Popular Angle Only):")
    for grade in sorted(popular_grades_count.keys(), key=lambda g: (g[0], int(g[1:]) if g[1:].isdigit() else 0)):
        print(f"  {grade}: {popular_grades_count[grade]} problems")

    # 3. Average number of holds used
    # Check if 'placements' column exists
    if 'placements' in df.columns:
        # Calculate number of holds for each problem
        df['hold_count'] = df['placements'].apply(count_holds)
        avg_holds = df['hold_count'].mean()

        print(f"\n3. Average Number of Holds Used: {avg_holds:.2f}")

        # Print distribution of hold counts
        print("\nHold Count Distribution:")
        hold_count_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        hold_count_hist = pd.cut(df['hold_count'], bins=hold_count_bins).value_counts().sort_index()
        for bin_range, count in hold_count_hist.items():
            print(f"  {bin_range}: {count} problems")
    else:
        print("\n3. Cannot calculate average number of holds - 'placements' column not found")
        avg_holds = None

    # 4. Bonus: Distribution of angles used
    angles = []

    for _, row in df.iterrows():
        stats_list = parse_climb_stats(row.get('climb_stats', []))

        for stats in stats_list:
            angle = stats.get('angle')
            if angle is not None:
                angles.append(angle)

    angle_count = Counter(angles)

    # Print angle distribution
    print("\n4. Board Angle Distribution:")
    for angle in sorted(angle_count.keys()):
        print(f"  {angle}°: {angle_count[angle]} setups")

    # Return compiled statistics for further use if needed
    return {
        'all_grades_count': all_grades_count,
        'popular_grades_count': popular_grades_count,
        'angles_count': angle_count,
        'avg_holds': avg_holds,
        'dataframe': df  # Return the dataframe with hold_count added if applicable
    }

def remove_hold_count_outliers(data_source, upper_percentile=95, output_file=None):
    """
    Create a new dataframe without outliers in the upper range of hold counts
    """
    df = load_data(data_source)
    if df is None:
        return None, None

    # Check if 'placements' column exists
    if 'placements' not in df.columns:
        print("Cannot analyze hold counts - 'placements' column not found in the data")
        return None, None

    # Calculate number of holds for each problem if not already done
    if 'hold_count' not in df.columns:
        df['hold_count'] = df['placements'].apply(count_holds)

    # Remove any rows with 0 holds (likely parsing errors)
    df = df[df['hold_count'] > 0]

    # Get original statistics
    original_count = len(df)
    original_min = df['hold_count'].min()
    original_max = df['hold_count'].max()
    original_mean = df['hold_count'].mean()
    original_median = df['hold_count'].median()
    original_std = df['hold_count'].std()

    # Calculate the upper threshold
    upper_threshold = df['hold_count'].quantile(upper_percentile/100)

    # Create filtered dataframe
    cleaned_df = df[df['hold_count'] <= upper_threshold]

    # Get new statistics
    cleaned_count = len(cleaned_df)
    cleaned_min = cleaned_df['hold_count'].min()
    cleaned_max = cleaned_df['hold_count'].max()
    cleaned_mean = cleaned_df['hold_count'].mean()
    cleaned_median = cleaned_df['hold_count'].median()
    cleaned_std = cleaned_df['hold_count'].std()

    # Print summary comparison
    print("Hold Count Statistics Comparison:")
    print("--------------------------------")
    print(f"{'Metric':<15} {'Original':<15} {'Cleaned':<15}")
    print(f"{'Count':<15} {original_count:<15} {cleaned_count:<15}")
    print(f"{'Min':<15} {original_min:<15} {cleaned_min:<15}")
    print(f"{'Max':<15} {original_max:<15} {cleaned_max:<15}")
    print(f"{'Mean':<15} {original_mean:<15.2f} {cleaned_mean:<15.2f}")
    print(f"{'Median':<15} {original_median:<15} {cleaned_median:<15}")
    print(f"{'Std Dev':<15} {original_std:<15.2f} {cleaned_std:<15.2f}")
    print(f"{'Upper Threshold':<15} {'-':<15} {upper_threshold:<15}")
    print(f"{'Removed':<15} {'-':<15} {original_count - cleaned_count:<15} ({(original_count - cleaned_count)/original_count*100:.1f}%)")

    # Create distribution table for cleaned data
    hold_count_counter = Counter(cleaned_df['hold_count'])
    sorted_counts = sorted(hold_count_counter.items())

    print("\nCleaned Hold Count Distribution:")
    print("----------------------")
    print("Holds | Count | Percentage")
    print("----------------------")
    total_problems = len(cleaned_df)

    for holds, count in sorted_counts:
        if count > 5:  # Only show if there are more than 5 examples
            percentage = (count / total_problems) * 100
            print(f"{holds:5d} | {count:5d} | {percentage:6.2f}%")

    # Return the cleaned dataframe and statistics
    stats = {
        'original': {
            'count': original_count,
            'min': original_min,
            'max': original_max,
            'mean': original_mean,
            'median': original_median,
            'std': original_std
        },
        'cleaned': {
            'count': cleaned_count,
            'min': cleaned_min,
            'max': cleaned_max,
            'mean': cleaned_mean,
            'median': cleaned_median,
            'std': cleaned_std,
            'upper_threshold': upper_threshold,
            'hold_count_distribution': dict(hold_count_counter)
        }
    }

    return cleaned_df, stats

def analyze_and_clean_data(data_source, upper_percentile=95, save_cleaned=False, output_folder=None):
    """
    Comprehensive function that creates a data profile and removes outliers in one go
    """
    # Create the data profile first
    profile = create_data_profile(data_source)

    if profile is None:
        return None, None, None

    df = profile['dataframe']

    # Determine output path
    if output_folder is None:
        output_folder = ""
    else:
        # Ensure path ends with /
        if not output_folder.endswith('/'):
            output_folder += '/'

    # Generate plot filename
    plot_filename = f"{output_folder}hold_count_comparison.png"

    # Remove outliers
    cleaned_df, stats = remove_hold_count_outliers(df, upper_percentile, output_file=plot_filename)

    if save_cleaned and cleaned_df is not None:
        # Generate output filename
        if isinstance(data_source, str) and (data_source.endswith('.csv') or data_source.endswith('.json')):
            base_name = data_source.split('/')[-1].split('.')[0]
            output_file = f"{output_folder}cleaned_{base_name}.csv"
        else:
            output_file = f"{output_folder}cleaned_kilterboard_data.csv"

        # Save to CSV
        cleaned_df.to_csv(output_file, index=False)
        print(f"Cleaned data saved to {output_file}")

    return cleaned_df, profile, stats

def compare_climbing_dataframes(df1, df2, df1_name="Dataset 1", df2_name="Dataset 2", plot=True):
    """
    Compare two climbing route DataFrames and print a summary of key metrics.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from collections import defaultdict

    # Initialize results dictionary
    results = {
        'datapoints': {df1_name: len(df1), df2_name: len(df2)},
        'holds': defaultdict(dict),
        'ascents': defaultdict(dict)
    }

    # Compare hold counts
    try:
        # For the first dataframe, calculate hold_count from placements
        if 'placements' in df1.columns:
            df1_temp = df1.copy()
            df1_temp['hold_count'] = df1_temp['placements'].apply(count_holds)
            df1_holds = df1_temp['hold_count'].dropna()

            # Calculate statistics
            results['holds'][df1_name]['mean'] = df1_holds.mean()
            results['holds'][df1_name]['min'] = df1_holds.min()
            results['holds'][df1_name]['max'] = df1_holds.max()
            results['holds'][df1_name]['std'] = df1_holds.std()
        else:
            print(f"Warning: Cannot calculate hold statistics for {df1_name} - 'placements' column not found")

        # For the second dataframe, use the existing hold_count column
        if 'hold_count' in df2.columns:
            df2_holds = df2['hold_count'].dropna()

            # Calculate statistics
            results['holds'][df2_name]['mean'] = df2_holds.mean()
            results['holds'][df2_name]['min'] = df2_holds.min()
            results['holds'][df2_name]['max'] = df2_holds.max()
            results['holds'][df2_name]['std'] = df2_holds.std()
        else:
            print(f"Warning: 'hold_count' column not found in {df2_name}")
    except Exception as e:
        print(f"Error processing hold counts: {e}")

    # Compare ascent counts
    try:
        # For the first dataframe, parse climb_stats to get ascensionist_count
        df1_ascents = []

        for _, row in df1.iterrows():
            if 'climb_stats' in row:
                stats_list = parse_climb_stats(row['climb_stats'])

                # Each angle setting is treated as a separate boulder
                for stats in stats_list:
                    if 'ascensionist_count' in stats:
                        df1_ascents.append(stats['ascensionist_count'])

        if df1_ascents:
            # Convert to numpy array for easier statistics
            df1_ascents_array = np.array(df1_ascents)

            # Calculate statistics
            results['ascents'][df1_name]['mean'] = np.mean(df1_ascents_array)
            results['ascents'][df1_name]['min'] = np.min(df1_ascents_array)
            results['ascents'][df1_name]['max'] = np.max(df1_ascents_array)
            results['ascents'][df1_name]['std'] = np.std(df1_ascents_array)

        # For the second dataframe, use the ascents column directly
        if 'ascents' in df2.columns:
            df2_ascents_array = df2['ascents'].dropna().values

            if len(df2_ascents_array) > 0:
                # Calculate statistics
                results['ascents'][df2_name]['mean'] = np.mean(df2_ascents_array)
                results['ascents'][df2_name]['min'] = np.min(df2_ascents_array)
                results['ascents'][df2_name]['max'] = np.max(df2_ascents_array)
                results['ascents'][df2_name]['std'] = np.std(df2_ascents_array)
        else:
            print(f"Warning: 'ascents' column not found in {df2_name}")
    except Exception as e:
        print(f"Error processing ascent counts: {e}")
        import traceback
        traceback.print_exc()

    # Print summary table
    print(f"\nComparison: {df1_name} vs {df2_name}")
    print("=" * 90)
    print(f"{'Metric':<15} {'Dataset':<12} {'Count':<10} {'Mean':<10} {'Min':<10} {'Max':<10} {'Std Dev':<10}")
    print("-" * 90)

    # Datapoints
    print(f"{'Datapoints':<15} {df1_name:<12} {len(df1):<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")
    print(f"{'Datapoints':<15} {df2_name:<12} {len(df2):<10} {'-':<10} {'-':<10} {'-':<10} {'-':<10}")

    # Hold counts
    if df1_name in results['holds']:
        print(f"{'Holds':<15} {df1_name:<12} {'-':<10} "
              f"{results['holds'][df1_name]['mean']:.2f} {' ' * 5} "
              f"{results['holds'][df1_name]['min']:.0f} {' ' * 7} "
              f"{results['holds'][df1_name]['max']:.0f} {' ' * 7} "
              f"{results['holds'][df1_name]['std']:.2f}")

    if df2_name in results['holds']:
        print(f"{'Holds':<15} {df2_name:<12} {'-':<10} "
              f"{results['holds'][df2_name]['mean']:.2f} {' ' * 5} "
              f"{results['holds'][df2_name]['min']:.0f} {' ' * 7} "
              f"{results['holds'][df2_name]['max']:.0f} {' ' * 7} "
              f"{results['holds'][df2_name]['std']:.2f}")

    # Ascent counts
    if df1_name in results['ascents']:
        print(f"{'Ascents':<15} {df1_name:<12} {'-':<10} "
              f"{results['ascents'][df1_name]['mean']:.2f} {' ' * 5} "
              f"{results['ascents'][df1_name]['min']:.0f} {' ' * 7} "
              f"{results['ascents'][df1_name]['max']:.0f} {' ' * 7} "
              f"{results['ascents'][df1_name]['std']:.2f}")

    if df2_name in results['ascents']:
        print(f"{'Ascents':<15} {df2_name:<12} {'-':<10} "
              f"{results['ascents'][df2_name]['mean']:.2f} {' ' * 5} "
              f"{results['ascents'][df2_name]['min']:.0f} {' ' * 7} "
              f"{results['ascents'][df2_name]['max']:.0f} {' ' * 7} "
              f"{results['ascents'][df2_name]['std']:.2f}")

    # Create visualizations
    if plot:
        try:
            # Set a consistent style
            plt.style.use('darkgrid')
            
            # Create figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(10, 12))

            # Colors for bars
            colors = {'original': 'steelblue', 'cleaned': 'forestgreen'}

            # 1. Plot Datapoints
            datasets = [df1_name, df2_name]
            datapoint_counts = [len(df1), len(df2)]

            bars1 = axs[0].bar(datasets, datapoint_counts, color=[colors.get(df1_name.lower(), 'steelblue'),
                                                           colors.get(df2_name.lower(), 'forestgreen')])
            axs[0].set_title('Number of Datapoints', fontsize=14)
            axs[0].set_ylabel('Count', fontsize=12)

            # Add counts on top of bars
            for bar in bars1:
                height = bar.get_height()
                axs[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', fontsize=10)

            # 2. Plot Hold Statistics
            if df1_name in results['holds'] and df2_name in results['holds']:
                x = np.arange(4)  # 4 statistics
                width = 0.35

                df1_hold_stats = [
                    results['holds'][df1_name]['mean'],
                    results['holds'][df1_name]['min'],
                    results['holds'][df1_name]['max'],
                    results['holds'][df1_name]['std']
                ]

                df2_hold_stats = [
                    results['holds'][df2_name]['mean'],
                    results['holds'][df2_name]['min'],
                    results['holds'][df2_name]['max'],
                    results['holds'][df2_name]['std']
                ]

                # Plot bars side by side
                bars2_1 = axs[1].bar(x - width/2, df1_hold_stats, width, label=df1_name,
                                  color=colors.get(df1_name.lower(), 'steelblue'))
                bars2_2 = axs[1].bar(x + width/2, df2_hold_stats, width, label=df2_name,
                                  color=colors.get(df2_name.lower(), 'forestgreen'))

                # Add labels and title
                axs[1].set_title('Hold Statistics', fontsize=14)
                axs[1].set_ylabel('Value', fontsize=12)
                axs[1].set_xticks(x)
                axs[1].set_xticklabels(['Mean', 'Min', 'Max', 'Std Dev'])
                axs[1].legend()

                # Add values on top of bars
                for bars in [bars2_1, bars2_2]:
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        # Format differently based on statistic type
                        if i in [0, 3]:  # Mean and StdDev
                            text = f'{height:.2f}'
                        else:  # Min and Max
                            text = f'{int(height)}'
                        axs[1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                text, ha='center', fontsize=9)

            # 3. Plot Ascent Statistics
            if df1_name in results['ascents'] and df2_name in results['ascents']:
                x = np.arange(4)  # 4 statistics
                width = 0.35

                df1_ascent_stats = [
                    results['ascents'][df1_name]['mean'],
                    results['ascents'][df1_name]['min'],
                    results['ascents'][df1_name]['max'],
                    results['ascents'][df1_name]['std']
                ]

                df2_ascent_stats = [
                    results['ascents'][df2_name]['mean'],
                    results['ascents'][df2_name]['min'],
                    results['ascents'][df2_name]['max'],
                    results['ascents'][df2_name]['std']
                ]

                # Plot bars side by side
                bars3_1 = axs[2].bar(x - width/2, df1_ascent_stats, width, label=df1_name,
                                  color=colors.get(df1_name.lower(), 'steelblue'))
                bars3_2 = axs[2].bar(x + width/2, df2_ascent_stats, width, label=df2_name,
                                  color=colors.get(df2_name.lower(), 'forestgreen'))

                # Add labels and title
                axs[2].set_title('Ascent Statistics', fontsize=14)
                axs[2].set_ylabel('Value', fontsize=12)
                axs[2].set_xticks(x)
                axs[2].set_xticklabels(['Mean', 'Min', 'Max', 'Std Dev'])
                axs[2].legend()

                # Add values on top of bars
                for bars in [bars3_1, bars3_2]:
                    for i, bar in enumerate(bars):
                        height = bar.get_height()
                        # Format differently based on statistic type
                        if i in [0, 3]:  # Mean and StdDev
                            text = f'{height:.2f}'
                        else:  # Min and Max
                            text = f'{int(height)}'
                        axs[2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                text, ha='center', fontsize=9)

            # Add overall title
            plt.suptitle(f'Comparison of {df1_name} vs {df2_name} Datasets', fontsize=16, y=0.98)

            # Adjust layout
            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            # Show the plot
            plt.show()

        except Exception as e:
            print(f"Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()

    return results