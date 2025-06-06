#!/usr/bin/env python3
"""
Script to create the CNN training notebook
"""

import json

# Save the notebook content from the artifact above
notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CNN Model Training for Climbing Grade Prediction\n",
    "This notebook trains a Convolutional Neural Network to predict climbing grades from hold placements and route features."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Setup and Imports\n",
    "import sys\n",
    "import os\n",
    "sys.path.append('.')  # Add current directory to Python path\n",
    "\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "from IPython.display import display\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
        "# Simple matplotlib settings\n"
       ]
      }
     ],
     "metadata": {},
     "nbformat": 4,
     "nbformat_minor": 2
    }