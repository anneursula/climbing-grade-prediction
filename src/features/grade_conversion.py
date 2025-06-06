# -*- coding: utf-8 -*-
"""grade_conversion.ipynb

Original file is located at
    https://colab.research.google.com/drive/1CobYqQgLpRTX2KXeviywfEtIEGN3gblV
"""

# src/features/grade_conversion.py
import pandas as pd

def difficulty_to_vgrade(difficulty):
    """Convert Kilter Board difficulty_average to V-grade with a refined scale

    Parameters:
    -----------
    difficulty : float
        Kilter Board difficulty average value (typically 0-40)

    Returns:
    --------
    str
        Corresponding V-grade (VB to V16+)
    """
    if difficulty is None or pd.isna(difficulty):
        return "N/A"

    # Detailed and precise conversion
    if difficulty < 8:
        return "VB"
    elif difficulty < 10:
        return "V0"
    elif difficulty < 12:
        return "V1"
    elif difficulty < 14:
        return "V2"
    elif difficulty < 16:
        return "V3"
    elif difficulty < 18:
        return "V4"
    elif difficulty < 20:
        return "V5"
    elif difficulty < 22:
        return "V6"
    elif difficulty < 24:
        return "V7"
    elif difficulty < 26:
        return "V8"
    elif difficulty < 28:
        return "V9"
    elif difficulty < 30:
        return "V10"
    elif difficulty < 32:
        return "V11"
    elif difficulty < 34:
        return "V12"
    elif difficulty < 36:
        return "V13"
    elif difficulty < 38:
        return "V14"
    elif difficulty < 40:
        return "V15"
    else:
        return "V16+"

def vgrade_to_difficulty(vgrade):
    """Convert V-grade to a numerical difficulty value

    Parameters:
    -----------
    vgrade : str
        V-grade (VB to V16+)

    Returns:
    --------
    float
        Corresponding numerical difficulty value
    """
    conversion = {
        "VB": 7,
        "V0": 9,
        "V1": 11,
        "V2": 13,
        "V3": 15,
        "V4": 17,
        "V5": 19,
        "V6": 21,
        "V7": 23,
        "V8": 25,
        "V9": 27,
        "V10": 29,
        "V11": 31,
        "V12": 33,
        "V13": 35,
        "V14": 37,
        "V15": 39,
        "V16+": 41
    }

    return conversion.get(vgrade, float('nan'))

def get_grade_conversion_table():
    """Return a dictionary mapping difficulty ranges to V-grades

    Returns:
    --------
    dict
        Dictionary with V-grades as keys and difficulty ranges as values
    """
    return {
        "VB": "< 8",
        "V0": "8-10",
        "V1": "10-12",
        "V2": "12-14",
        "V3": "14-16",
        "V4": "16-18",
        "V5": "18-20",
        "V6": "20-22",
        "V7": "22-24",
        "V8": "24-26",
        "V9": "26-28",
        "V10": "28-30",
        "V11": "30-32",
        "V12": "32-34",
        "V13": "34-36",
        "V14": "36-38",
        "V15": "38-40",
        "V16+": "> 40"
    }