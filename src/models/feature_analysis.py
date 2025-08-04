#src/models/feature_analysis.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_numerical_feature_importance(model, X_test_grids, X_test_features, y_test, feature_names):
    """
    Analyze importance of numerical features using gradient-based attribution
    """
    
    # Convert to tensors
    grid_tensor = tf.constant(X_test_grids, dtype=tf.float32)
    feature_tensor = tf.constant(X_test_features, dtype=tf.float32)
    
    # Calculate gradients with respect to numerical features
    with tf.GradientTape() as tape:
        tape.watch(feature_tensor)
        predictions = model([grid_tensor, feature_tensor])
        loss = tf.reduce_mean(predictions)  # Or use your specific loss
    
    # Get gradients
    gradients = tape.gradient(loss, feature_tensor)
    
    # Calculate importance as mean absolute gradient
    feature_importance = tf.reduce_mean(tf.abs(gradients), axis=0)
    
    # Create results dictionary
    importance_dict = {}
    for i, name in enumerate(feature_names):
        importance_dict[name] = float(feature_importance[i])
    
    return importance_dict

def analyze_channel_importance(model, X_test_grids, X_test_features, y_test):
    """
    Analyze importance of different grid channels by zeroing them out
    """
    
    channel_names = [
        'START', 'MIDDLE', 'FINISH', 'FEET-ONLY', 
        'orientation', 'depth', 
        'footchip', 'jug', 'sloper', 'crimp', 'pinch'
    ]
    
    # Get baseline predictions
    baseline_preds = model.predict([X_test_grids, X_test_features])
    baseline_mse = np.mean((baseline_preds.flatten() - y_test) ** 2)
    
    channel_importance = {}
    
    for i, channel_name in enumerate(channel_names):
        # Create modified input with channel i zeroed out
        modified_grids = X_test_grids.copy()
        modified_grids[:, :, :, i] = 0  # Zero out channel i
        
        # Get predictions with modified input
        modified_preds = model.predict([modified_grids, X_test_features])
        modified_mse = np.mean((modified_preds.flatten() - y_test) ** 2)
        
        # Importance = performance degradation when channel is removed
        importance = modified_mse - baseline_mse
        channel_importance[channel_name] = importance
        
        print(f"Channel {channel_name}: MSE increase = {importance:.4f}")
    
    return channel_importance

def plot_feature_importance(importance_dict, title="Feature Importance"):
    """
    Plot feature importance results
    """
    features = list(importance_dict.keys())
    importance = list(importance_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(features, importance)
    plt.title(title)
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.xticks(rotation=45, ha='right')
    
    # Color bars by importance
    colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def comprehensive_feature_analysis(model, X_test_grids, X_test_features, y_test):
    """
    Complete feature importance analysis
    """
    
    # Define feature names (adjust based on your actual features)
    numerical_features = [
        'angle', 'hold_count', 'ascents', 'quality_average',
        'avg_orientation', 'avg_depth', 'std_orientation', 'std_depth',
        'start_holds', 'middle_holds', 'finish_holds', 'feet_only',
        'footchip_count', 'jug_count', 'sloper_count', 'crimp_count', 'pinch_count'
    ]
    
    print("Analyzing numerical feature importance...")
    numerical_importance = analyze_numerical_feature_importance(
        model, X_test_grids, X_test_features, y_test, numerical_features
    )
    
    print("\nAnalyzing channel importance...")
    channel_importance = analyze_channel_importance(
        model, X_test_grids, X_test_features, y_test
    )
    
    # Plot results
    fig1 = plot_feature_importance(numerical_importance, "Numerical Feature Importance")
    fig2 = plot_feature_importance(channel_importance, "Grid Channel Importance")
    
    # Print top features
    print("\nTop 5 Numerical Features:")
    sorted_numerical = sorted(numerical_importance.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_numerical[:5]:
        print(f"  {name}: {importance:.4f}")
    
    print("\nTop 5 Grid Channels:")
    sorted_channels = sorted(channel_importance.items(), key=lambda x: x[1], reverse=True)
    for name, importance in sorted_channels[:5]:
        print(f"  {name}: {importance:.4f}")
    
    return numerical_importance, channel_importance

