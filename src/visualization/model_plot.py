# src/visualization/model_plots.py

import matplotlib.pyplot as plt
import numpy as np

def plot_training_history(history):
    """
    Plot training history for a keras model
    
    Parameters:
    -----------
    history : tensorflow.keras.callbacks.History
        History object returned from model.fit()
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the plots
    """
    fig = plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.legend()
    
    plt.tight_layout()
    
    return fig