# -*- coding: utf-8 -*-
"""loaders.ipynb


Original file is located at
    https://colab.research.google.com/drive/1CobYqQgLpRTX2KXeviywfEtIEGN3gblV
"""

# src/data/loaders.py
import os
import time
import json
import pandas as pd
from supabase import create_client
from tqdm import tqdm

def initialize_supabase(url, key):
    """Initialize the Supabase client

    Parameters:
    -----------
    url : str
        Supabase project URL
    key : str
        Supabase project API key

    Returns:
    --------
    supabase client
    """
    return create_client(url, key)

def fetch_all_records(supabase, table_name, page_size=1000):
    """Fetch all records from a table with pagination

    Parameters:
    -----------
    supabase : supabase client
        Initialized Supabase client
    table_name : str
        Name of the table to fetch data from
    page_size : int, optional
        Number of records to fetch per request (default: 1000)

    Returns:
    --------
    list
        List of records fetched from the table
    """
    all_records = []
    offset = 0
    has_more = True

    print(f"Fetching data from '{table_name}' table...")

    while has_more:
        try:
            response = supabase.table(table_name).select("*").range(offset, offset + page_size - 1).execute()

            data = response.data
            count = len(data)

            if count > 0:
                all_records.extend(data)
                offset += count
                print(f"  Fetched {len(all_records)} records so far...")

                # If we got fewer records than requested, we're at the end
                has_more = count == page_size
            else:
                has_more = False

            # Be nice to the API and avoid rate limiting
            time.sleep(0.5)

        except Exception as e:
            print(f"Error fetching data from {table_name}: {e}")
            break

    print(f"Total records fetched from '{table_name}': {len(all_records)}")
    return all_records

def discover_tables(supabase):
    """Try to discover available tables in the database

    Parameters:
    -----------
    supabase : supabase client
        Initialized Supabase client

    Returns:
    --------
    list
        List of available table names
    """
    # Common tables we might expect to find in a climbing app
    potential_tables = [
        "climbs", "angles", "setters", "ascents", "users", "grades",
        "boards", "holds", "problems", "ratings", "comments", "favorites",
        "climb_holds", "climb_angles", "climb_grades", "climb_ratings"
    ]

    available_tables = []

    print("Discovering available tables...")
    for table in tqdm(potential_tables):
        try:
            response = supabase.table(table).select("*").limit(1).execute()
            if response:
                available_tables.append(table)
                print(f"  Found table: {table}")
        except Exception:
            pass

    return available_tables

def save_data_to_csv(data, table_name, output_dir="data"):
    """Save data to CSV file

    Parameters:
    -----------
    data : list
        List of records to save
    table_name : str
        Name of the table (used for the filename)
    output_dir : str, optional
        Directory to save the files in (default: "data")
    """
    if not data:
        print(f"No data to save for table '{table_name}'")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(data)
    file_path = os.path.join(output_dir, f"{table_name}.csv")
    df.to_csv(file_path, index=False)
    print(f"Saved {len(df)} records to {file_path}")

    # Also save raw JSON for backup
    json_path = os.path.join(output_dir, f"{table_name}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f)

def download_kilterboard_data(supabase_url, supabase_key, output_dir="data"):
    """Main function to download all Kilterboard data

    Parameters:
    -----------
    supabase_url : str
        Supabase project URL
    supabase_key : str
        Supabase project API key
    output_dir : str, optional
        Directory to save the files in (default: "data")
    """
    supabase = initialize_supabase(supabase_url, supabase_key)

    # First, try to discover available tables
    available_tables = discover_tables(supabase)

    if not available_tables:
        print("No tables discovered. Trying known tables...")
        available_tables = ["climbs"]  # We know this table exists from the code snippet

    # Download data from each available table
    for table in available_tables:
        data = fetch_all_records(supabase, table)
        save_data_to_csv(data, table, output_dir)

    # Additionally, try to get specific climb data with related information
    try:
        print("Attempting to fetch detailed climb information...")
        # This is a more advanced query that might work if the schema allows it
        response = supabase.table("climbs").select("*, setters(*)").execute()
        if response.data:
            save_data_to_csv(response.data, "climbs_with_setters", output_dir)
    except Exception as e:
        print(f"Could not fetch detailed climb information: {e}")

    print(f"\nData download complete! Files saved to '{output_dir}' directory.")

def load_data_from_file(file_path):
    """Load climb data from a file

    Parameters:
    -----------
    file_path : str
        Path to the CSV or JSON file

    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the climb data
    """
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        return pd.DataFrame(json_data)
    else:
        raise ValueError("Unsupported file format. Please use CSV or JSON.")