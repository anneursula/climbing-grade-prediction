# src/models/weighted_metrics.py

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from ..features.grade_conversion import difficulty_to_vgrade

def extract_quality_ratings_from_stats(climb_stats_series, target_angles=None, use_most_popular=True):
    """
    Extract quality ratings directly from climb_stats column
    
    Parameters:
    -----------
    climb_stats_series : pandas.Series
        Series containing climb_stats data (from original unclean data)
    target_angles : pandas.Series or list, optional
        Specific angles to extract quality for (if data is angle-specific)
        Should align with climb_stats_series index
    use_most_popular : bool
        If True and no target_angles provided, use most popular setup for each boulder
        If False, average quality across all angles
        
    Returns:
    --------
    pandas.Series
        Quality ratings aligned with the input series
    """
    from ..data.preprocessing import parse_climb_stats, find_most_popular_setup
    
    quality_ratings = []
    
    print(f"Extracting quality from {len(climb_stats_series)} climb_stats entries...")
    
    successful_extractions = 0
    default_assignments = 0
    
    for idx, stats_data in climb_stats_series.items():
        extracted_quality = None
        
        try:
            # Parse the climb_stats
            stats_list = parse_climb_stats(stats_data)
            
            if not stats_list:
                # No stats found
                extracted_quality = None
            elif target_angles is not None:
                # Look for specific angle
                target_angle = target_angles.loc[idx] if hasattr(target_angles, 'loc') else target_angles[idx]
                
                for stats in stats_list:
                    if stats.get('angle') == target_angle:
                        extracted_quality = stats.get('quality_average')
                        break
            elif use_most_popular:
                # Use most popular setup
                popular_setup = find_most_popular_setup(stats_data)
                if popular_setup:
                    extracted_quality = popular_setup.get('quality_average')
            else:
                # Average quality across all angles
                qualities = [stats.get('quality_average') for stats in stats_list 
                           if stats.get('quality_average') is not None]
                if qualities:
                    extracted_quality = sum(qualities) / len(qualities)
        
        except Exception as e:
            print(f"Error parsing stats for index {idx}: {e}")
            extracted_quality = None
        
        # Use extracted quality or default
        if extracted_quality is not None and pd.notna(extracted_quality):
            quality_ratings.append(float(extracted_quality))
            successful_extractions += 1
        else:
            quality_ratings.append(3.0)  # Default neutral quality
            default_assignments += 1
    
    print(f"Quality extraction complete:")
    print(f"  Successful extractions: {successful_extractions}")
    print(f"  Default assignments: {default_assignments}")
    print(f"  Success rate: {successful_extractions/len(climb_stats_series)*100:.1f}%")
    
    quality_series = pd.Series(quality_ratings, index=climb_stats_series.index)
    
    print(f"Quality statistics:")
    print(f"  Mean: {quality_series.mean():.3f}")
    print(f"  Std: {quality_series.std():.3f}")
    print(f"  Min: {quality_series.min():.3f}")
    print(f"  Max: {quality_series.max():.3f}")
    
    return quality_series


def create_quality_weights(quality_ratings, weight_function='exponential', min_weight=0.1, max_weight=2.0):
    """
    Create weights based on quality ratings
    
    Parameters:
    -----------
    quality_ratings : pandas.Series or list
        Quality ratings (typically 1-5 scale)
    weight_function : str
        Type of weighting function ('linear', 'exponential', 'sigmoid')
    min_weight : float
        Minimum weight for lowest quality boulders
    max_weight : float
        Maximum weight for highest quality boulders
        
    Returns:
    --------
    numpy.ndarray
        Array of weights corresponding to each boulder
    """
    quality_array = np.array(quality_ratings)
    
    # Handle missing values
    quality_array = np.where(np.isnan(quality_array), 3.0, quality_array)
    
    # Normalize quality to 0-1 range (assuming 1-5 scale)
    min_quality = np.min(quality_array)
    max_quality = np.max(quality_array)
    
    if max_quality == min_quality:
        # All qualities are the same, return equal weights
        return np.ones(len(quality_array))
    
    normalized_quality = (quality_array - min_quality) / (max_quality - min_quality)
    
    if weight_function == 'linear':
        weights = min_weight + (max_weight - min_weight) * normalized_quality
        
    elif weight_function == 'exponential':
        # Exponential curve: low quality gets exponentially lower weight
        weights = min_weight * np.exp(np.log(max_weight / min_weight) * normalized_quality)
        
    elif weight_function == 'sigmoid':
        # Sigmoid function: smooth transition with steeper middle
        x = (normalized_quality - 0.5) * 6  # Scale to reasonable sigmoid range
        sigmoid = 1 / (1 + np.exp(-x))
        weights = min_weight + (max_weight - min_weight) * sigmoid
        
    else:
        raise ValueError("weight_function must be 'linear', 'exponential', or 'sigmoid'")
    
    return weights

def map_quality_to_clean_data(climb_stats_series, clean_df, original_df):
    """
    Map quality ratings from original data to clean processed data
    
    Parameters:
    -----------
    climb_stats_series : pandas.Series
        climb_stats column from original data
    clean_df : pandas.DataFrame
        Your processed clean dataset (angle-specific rows)
    original_df : pandas.DataFrame
        Original unclean dataset
        
    Returns:
    --------
    pandas.Series
        Quality ratings aligned with clean_df
    """
    from ..data.preprocessing import parse_climb_stats
    
    print("Mapping quality ratings from original to clean data...")
    
    quality_ratings = []
    
    for idx, row in clean_df.iterrows():
        boulder_name = row['name']
        target_angle = row['angle']
        
        # Find this boulder in the original data
        original_boulder = original_df[original_df['name'] == boulder_name]
        
        if len(original_boulder) == 0:
            print(f"Warning: Boulder '{boulder_name}' not found in original data")
            quality_ratings.append(3.0)
            continue
        
        # Get the climb_stats for this boulder
        original_idx = original_boulder.index[0]
        stats_data = climb_stats_series.loc[original_idx]
        
        # Parse and find quality for target angle
        extracted_quality = None
        try:
            stats_list = parse_climb_stats(stats_data)
            
            for stats in stats_list:
                if stats.get('angle') == target_angle:
                    extracted_quality = stats.get('quality_average')
                    break
        except Exception as e:
            print(f"Error parsing stats for boulder '{boulder_name}': {e}")
        
        # Use extracted quality or default
        if extracted_quality is not None and pd.notna(extracted_quality):
            quality_ratings.append(float(extracted_quality))
        else:
            quality_ratings.append(3.0)
    
    quality_series = pd.Series(quality_ratings, index=clean_df.index)
    
    print(f"Quality mapping complete:")
    print(f"  Mean: {quality_series.mean():.3f}")
    print(f"  Std: {quality_series.std():.3f}")
    print(f"  Non-default values: {sum(1 for x in quality_ratings if x != 3.0)}")
    
    return quality_series

def weighted_accuracy_score(y_true, y_pred, weights, exact_match=True):
    """
    Calculate weighted accuracy score
    
    Parameters:
    -----------
    y_true : array-like
        True target values (grades)
    y_pred : array-like  
        Predicted target values (grades)
    weights : array-like
        Weight for each sample
    exact_match : bool
        If True, only exact matches count. If False, allows ±1 grade tolerance
        
    Returns:
    --------
    float
        Weighted accuracy score
    """
    if exact_match:
        matches = np.array(y_true) == np.array(y_pred)
    else:
        # Allow ±1 grade tolerance using V-grade distance
        def v_grade_distance(v1, v2):
            v_scale = ["VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16+"]
            try:
                return abs(v_scale.index(str(v1)) - v_scale.index(str(v2)))
            except ValueError:
                return float('inf')
        
        # Convert numeric grades to V-grades if needed
        if isinstance(y_true[0], (int, float)):
            y_true_vgrades = [difficulty_to_vgrade(grade) for grade in y_true]
            y_pred_vgrades = [difficulty_to_vgrade(grade) for grade in y_pred]
        else:
            y_true_vgrades = y_true
            y_pred_vgrades = y_pred
            
        matches = np.array([v_grade_distance(true, pred) <= 1 
                           for true, pred in zip(y_true_vgrades, y_pred_vgrades)])
    
    # Calculate weighted accuracy
    weighted_correct = np.sum(matches * weights)
    total_weight = np.sum(weights)
    
    return weighted_correct / total_weight if total_weight > 0 else 0.0

def weighted_mean_absolute_error(y_true, y_pred, weights):
    """
    Calculate weighted mean absolute error
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values  
    weights : array-like
        Weight for each sample
        
    Returns:
    --------
    float
        Weighted MAE
    """
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    weighted_errors = errors * weights
    
    return np.sum(weighted_errors) / np.sum(weights) if np.sum(weights) > 0 else 0.0

def comprehensive_weighted_evaluation(y_true, y_pred, quality_ratings, weight_function='exponential'):
    """
    Perform comprehensive evaluation with quality-weighted metrics
    
    Parameters:
    -----------
    y_true : array-like
        True target values (numeric grades)
    y_pred : array-like
        Predicted target values (numeric grades)
    quality_ratings : array-like
        Quality ratings for each prediction
    weight_function : str
        Type of weighting function to use
        
    Returns:
    --------
    dict
        Dictionary containing all evaluation metrics
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    # Create weights
    weights = create_quality_weights(quality_ratings, weight_function=weight_function)
    
    # Convert to V-grades for grade-specific metrics
    y_true_vgrades = [difficulty_to_vgrade(grade) for grade in y_true]
    y_pred_vgrades = [difficulty_to_vgrade(grade) for grade in y_pred]
    
    # Calculate standard metrics
    standard_metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'accuracy_exact': weighted_accuracy_score(y_true_vgrades, y_pred_vgrades, 
                                                 np.ones(len(y_true)), exact_match=True),
        'accuracy_pm1': weighted_accuracy_score(y_true_vgrades, y_pred_vgrades, 
                                               np.ones(len(y_true)), exact_match=False)
    }
    
    # Calculate weighted metrics
    weighted_metrics = {
        'weighted_mae': weighted_mean_absolute_error(y_true, y_pred, weights),
        'weighted_accuracy_exact': weighted_accuracy_score(y_true_vgrades, y_pred_vgrades, 
                                                          weights, exact_match=True),
        'weighted_accuracy_pm1': weighted_accuracy_score(y_true_vgrades, y_pred_vgrades, 
                                                        weights, exact_match=False)
    }
    
    # Quality distribution analysis
    quality_stats = {
        'quality_mean': np.mean(quality_ratings),
        'quality_std': np.std(quality_ratings),
        'quality_min': np.min(quality_ratings),
        'quality_max': np.max(quality_ratings),
        'weight_mean': np.mean(weights),
        'weight_std': np.std(weights)
    }
    
    return {**standard_metrics, **weighted_metrics, **quality_stats}

def plot_quality_weight_analysis(quality_ratings, weights, y_true, y_pred, save_path=None):
    """
    Create visualizations for quality-weight analysis
    
    Parameters:
    -----------
    quality_ratings : array-like
        Quality ratings
    weights : array-like
        Computed weights
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    save_path : str, optional
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Quality distribution
    axes[0, 0].hist(quality_ratings, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Quality Ratings')
    axes[0, 0].set_xlabel('Quality Rating')
    axes[0, 0].set_ylabel('Frequency')
    
    # 2. Weight vs Quality relationship
    axes[0, 1].scatter(quality_ratings, weights, alpha=0.6, color='orange')
    axes[0, 1].set_title('Weights vs Quality Ratings')
    axes[0, 1].set_xlabel('Quality Rating')
    axes[0, 1].set_ylabel('Weight')
    
    # 3. Prediction errors by quality
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    
    # Bin by quality for better visualization
    quality_bins = np.digitize(quality_ratings, bins=np.linspace(np.min(quality_ratings), 
                                                                np.max(quality_ratings), 6))
    
    quality_bin_centers = []
    mean_errors = []
    
    for bin_idx in range(1, 7):
        mask = quality_bins == bin_idx
        if np.sum(mask) > 0:
            quality_bin_centers.append(np.mean(quality_ratings[mask]))
            mean_errors.append(np.mean(errors[mask]))
    
    axes[1, 0].plot(quality_bin_centers, mean_errors, 'ro-', linewidth=2, markersize=8)
    axes[1, 0].set_title('Mean Prediction Error by Quality')
    axes[1, 0].set_xlabel('Quality Rating')
    axes[1, 0].set_ylabel('Mean Absolute Error')
    
    # 4. Weighted vs Unweighted accuracy comparison
    from sklearn.metrics import accuracy_score
    
    # Calculate accuracy for different quality buckets
    buckets = ['Low (1-2)', 'Medium (2-3)', 'High (3-4)', 'Very High (4-5)']
    bucket_masks = [
        (quality_ratings >= 1) & (quality_ratings < 2),
        (quality_ratings >= 2) & (quality_ratings < 3),
        (quality_ratings >= 3) & (quality_ratings < 4),
        (quality_ratings >= 4) & (quality_ratings <= 5)
    ]
    
    unweighted_acc = []
    weighted_acc = []
    
    for mask in bucket_masks:
        if np.sum(mask) > 0:
            bucket_y_true = [difficulty_to_vgrade(y) for y in np.array(y_true)[mask]]
            bucket_y_pred = [difficulty_to_vgrade(y) for y in np.array(y_pred)[mask]]
            bucket_weights = weights[mask]
            
            # Unweighted accuracy
            matches = np.array(bucket_y_true) == np.array(bucket_y_pred)
            unweighted_acc.append(np.mean(matches))
            
            # Weighted accuracy
            weighted_acc.append(weighted_accuracy_score(bucket_y_true, bucket_y_pred, 
                                                       bucket_weights, exact_match=True))
        else:
            unweighted_acc.append(0)
            weighted_acc.append(0)
    
    x = np.arange(len(buckets))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, unweighted_acc, width, label='Unweighted', alpha=0.8)
    axes[1, 1].bar(x + width/2, weighted_acc, width, label='Weighted', alpha=0.8)
    axes[1, 1].set_title('Accuracy by Quality Bucket')
    axes[1, 1].set_xlabel('Quality Bucket')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(buckets, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Quality analysis plot saved to {save_path}")
    
    plt.show()

def update_cnn_model_with_weighted_metrics(model, X_test_grids, X_test_features, y_test, quality_ratings):
    """
    Evaluate trained CNN model with weighted metrics
    
    Parameters:
    -----------
    model : tensorflow.keras.Model
        Trained CNN model
    X_test_grids : numpy.ndarray
        Test grid data
    X_test_features : numpy.ndarray
        Test feature data
    y_test : array-like
        Test target values
    quality_ratings : array-like
        Quality ratings for test samples
        
    Returns:
    --------
    dict
        Comprehensive evaluation results
    """
    # Make predictions
    predictions = model.predict([X_test_grids, X_test_features]).flatten()
    
    # Perform comprehensive weighted evaluation
    results = comprehensive_weighted_evaluation(y_test, predictions, quality_ratings)
    
    # Print results
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION WITH QUALITY WEIGHTING")
    print("="*60)
    
    print("\nStandard Metrics:")
    print("-" * 20)
    print(f"MAE: {results['mae']:.4f}")
    print(f"RMSE: {results['rmse']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"Accuracy (exact): {results['accuracy_exact']:.4f}")
    print(f"Accuracy (±1 grade): {results['accuracy_pm1']:.4f}")
    
    print("\nQuality-Weighted Metrics:")
    print("-" * 25)
    print(f"Weighted MAE: {results['weighted_mae']:.4f}")
    print(f"Weighted Accuracy (exact): {results['weighted_accuracy_exact']:.4f}")
    print(f"Weighted Accuracy (±1 grade): {results['weighted_accuracy_pm1']:.4f}")
    
    print("\nQuality Statistics:")
    print("-" * 18)
    print(f"Quality Mean: {results['quality_mean']:.2f}")
    print(f"Quality Std: {results['quality_std']:.2f}")
    print(f"Quality Range: {results['quality_min']:.2f} - {results['quality_max']:.2f}")
    print(f"Weight Mean: {results['weight_mean']:.2f}")
    
    # Calculate improvement
    accuracy_improvement = results['weighted_accuracy_exact'] - results['accuracy_exact']
    mae_improvement = results['mae'] - results['weighted_mae']
    
    print("\nImprovements from Quality Weighting:")
    print("-" * 35)
    print(f"Accuracy improvement: {accuracy_improvement:+.4f}")
    print(f"MAE improvement: {mae_improvement:+.4f}")
    
    return results

def analyze_quality_distribution(boulder_angles_df):
    """
    Analyze the quality distribution in the dataset
    """
    print("Analyzing quality distribution...")
    
    quality_ratings = extract_quality_ratings(boulder_angles_df)
    
    # Basic statistics
    print(f"\nQuality Rating Statistics:")
    print(f"Count: {len(quality_ratings)}")
    print(f"Mean: {quality_ratings.mean():.3f}")
    print(f"Std: {quality_ratings.std():.3f}")
    print(f"Min: {quality_ratings.min():.3f}")
    print(f"Max: {quality_ratings.max():.3f}")
    
    # Distribution
    quality_counts = pd.cut(quality_ratings, bins=10).value_counts().sort_index()
    print(f"\nQuality Distribution (10 bins):")
    for interval, count in quality_counts.items():
        percentage = (count / len(quality_ratings)) * 100
        print(f"{interval}: {count} ({percentage:.1f}%)")
    
    return quality_ratings