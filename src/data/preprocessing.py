# -*- coding: utf-8 -*-
"""preprocessing.ipynb

Original file is located at
    https://colab.research.google.com/drive/1CobYqQgLpRTX2KXeviywfEtIEGN3gblV
"""

# src/data/preprocessing.py
import pandas as pd
import ast

def parse_climb_stats(stats_str):
    """Parse the climb_stats string into a list of dictionaries

    Parameters:
    -----------
    stats_str : str
        String representation of climb stats

    Returns:
    --------
    list
        List of dictionaries containing climb stats
    """
    if not isinstance(stats_str, str):
        return []

    # Clean the string if it's wrapped in quotes and brackets
    if stats_str.startswith('[\"[') and stats_str.endswith(']\"]'):
        stats_str = stats_str[3:-3]  # Remove the [\"[ and ]\"]\n

    try:
        # Try parsing as a list of dictionaries
        return ast.literal_eval(stats_str)
    except (SyntaxError, ValueError):
        print(f"Error parsing: {stats_str[:100]}...")
        return []

def find_setup_by_angle(climb_stats, target_angle):
    """Find the setup for a specific angle

    Parameters:
    -----------
    climb_stats : str
        String representation of climb stats
    target_angle : int
        Angle to find the setup for

    Returns:
    --------
    dict or None
        Setup stats for the target angle, or None if not found
    """
    stats_list = parse_climb_stats(climb_stats)

    for stats in stats_list:
        if stats.get('angle') == target_angle:
            return stats

    return None

def find_most_popular_setup(climb_stats):
    """Find the angle with the most ascents

    Parameters:
    -----------
    climb_stats : str
        String representation of climb stats

    Returns:
    --------
    dict or None
        Setup stats for the most popular angle, or None if no stats found
    """
    stats_list = parse_climb_stats(climb_stats)

    most_ascents = 0
    popular_setup = None

    for stats in stats_list:
        ascents = stats.get('ascensionist_count', 0)
        if ascents > most_ascents:
            most_ascents = ascents
            popular_setup = stats

    return popular_setup


def count_holds(placements_str):
    """Count the number of holds used in a problem from the placements field"""
    if not placements_str or pd.isna(placements_str):
        return 0

    try:
        # Try to parse the placements string
        placements = ast.literal_eval(placements_str)
        return len(placements)
    except (SyntaxError, ValueError):
        print(f"Error parsing placements: {placements_str[:100]}...")
        return 0
    


def load_data(data_source):
    """Load data from a file path or return the DataFrame if already loaded"""
    if isinstance(data_source, pd.DataFrame):
        return data_source.copy()
    elif isinstance(data_source, str):
        if data_source.endswith('.csv'):
            return pd.read_csv(data_source)
        elif data_source.endswith('.json'):
            with open(data_source, 'r') as f:
                data = json.load(f)
            return pd.DataFrame(data)
        else:
            print("Unsupported file format. Please use CSV or JSON.")
            return None
    else:
        print("Unsupported data type. Please provide a file path or pandas DataFrame.")
        return None
    

def create_boulder_angle_dataframe(original_df, min_ascents=2):
    """
    Create a new DataFrame with separate entries for each boulder at each angle,
    filtering out entries with fewer than the specified minimum ascents.
    """
    # Create an empty list to store the new rows
    new_rows = []

    # Iterate through each row in the original DataFrame
    for _, row in original_df.iterrows():
        name = row['name']
        placements = row['placements']
        hold_count = row['hold_count'] if 'hold_count' in row else count_holds(placements)

        # Use the existing parse_climb_stats function to get the stats data
        stats_list = parse_climb_stats(row['climb_stats'])

        # Process each angle's stats
        for stats in stats_list:
            # Extract the required fields
            angle = stats.get('angle')
            grade = stats.get('difficulty_average')
            ascents = stats.get('ascensionist_count', 0)

            # Skip entries with missing critical data or insufficient ascents
            if angle is None or grade is None:
                continue

            # Filter out entries with fewer than min_ascents
            if ascents < min_ascents:
                continue

            # Create a new row
            new_row = {
                'name': name,
                'angle': int(angle),  # Ensure angle is an integer
                'grade': grade,
                'placements': placements,
                'ascents': ascents,
                'hold_count': hold_count
            }

            new_rows.append(new_row)

    # Create the new DataFrame
    new_df = pd.DataFrame(new_rows)

    # Print statistics about filtering
    total_entries = sum(len(parse_climb_stats(row['climb_stats'])) for _, row in original_df.iterrows())
    filtered_entries = len(new_df)
    removed_entries = total_entries - filtered_entries

    print(f"Total angle-specific entries: {total_entries}")
    print(f"Entries with {min_ascents}+ ascents: {filtered_entries}")
    print(f"Removed entries: {removed_entries} ({removed_entries/total_entries*100:.1f}% of total)")

    return new_df