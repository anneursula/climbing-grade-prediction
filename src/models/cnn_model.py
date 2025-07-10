# src/models/cnn_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from ..features.grade_conversion import difficulty_to_vgrade

def encode_hold_function(placements_str):
    """
    Create new feature encoding the number of each hold type
    
    Parameters:
    -----------
    placements_str : str or list
        String representation of holds placements or the parsed list
        
    Returns:
    --------
    list
        Counts of start, middle, finish, and feet-only holds
    """
    try:
        placements = ast.literal_eval(placements_str) if isinstance(placements_str, str) else placements_str

        # Count different hold types
        start_holds = sum(1 for hold in placements if hold.get('type') == 'START')
        middle_holds = sum(1 for hold in placements if hold.get('type') == 'MIDDLE')
        finish_holds = sum(1 for hold in placements if hold.get('type') == 'FINISH')
        feet_only = sum(1 for hold in placements if hold.get('type') == 'FEET-ONLY')

        return [start_holds, middle_holds, finish_holds, feet_only]
    except:
        return [0, 0, 0, 0]


def create_multichannel_grid(placements_str, hold_data_df, grid_width=24, grid_height=18):
    """
    Create a multi-channel 2D grid with enhanced hold information
    
    Parameters:
    -----------
    placements_str : str or list
        String representation of holds placements or the parsed list
    hold_data_df : pandas.DataFrame
        DataFrame containing hold characteristics indexed by ledPosition
    grid_width : int
        Width of the grid (default: 24)
    grid_height : int
        Height of the grid (default: 18)
        
    Returns:
    --------
    numpy.ndarray
        Multi-channel grid representation (shape: grid_height, grid_width, 11)
        Channels: [START, MIDDLE, FINISH, FEET-ONLY, orientation, depth, type_encoded]
    """
    # Initialize grid with 7 channels
    # 0-3: hold types (START, MIDDLE, FINISH, FEET-ONLY)
    # 4: orientation (normalized)
    # 5: depth (normalized) 
    # 6: type encoded (footchip=0, other types can be added)
    grid = np.zeros((grid_height, grid_width, 11))

    try:
        # Parse placements
        placements = ast.literal_eval(placements_str) if isinstance(placements_str, str) else placements_str

        # Channel mappings for hold types
        type_to_channel = {
            'START': 0,
            'MIDDLE': 1,
            'FINISH': 2,
            'FEET-ONLY': 3
        }
        
        # Normalize orientation and depth values for better learning
        max_orientation = 360.0
        max_depth = hold_data_df['depth'].max() if 'depth' in hold_data_df.columns else 5.0

        # Fill grid with hold placements
        for hold in placements:
            x = hold.get('x', 0)
            y = hold.get('y', 0)
            led_position = hold.get('ledPosition')

            # Fix negative coordinates by taking absolute values (hammer-nail solution, but works for now)
            x = abs(x)
            y = abs(y)
                
            # Normalize to grid indices with proper bounds checking
            # Assuming coordinates are in range [0, max_x] and [0, max_y]
            x_idx = int(x * (grid_width - 1) / 24) if x <= 24 else grid_width - 1
            y_idx = int(y * (grid_height - 1) / 36) if y <= 36 else grid_height - 1
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, grid_width - 1))
            y_idx = max(0, min(y_idx, grid_height - 1))

            # Get hold type and map to channel
            hold_type = hold.get('type', '')
            type_channel = type_to_channel.get(hold_type, 1)  # Default to MIDDLE
            grid[y_idx, x_idx, type_channel] = 1

            # Add enhanced hold information if ledPosition exists in hold_data
            if led_position is not None and led_position in hold_data_df.index:
                hold_info = hold_data_df.loc[led_position]
                
                # Channel 4: Normalized orientation (0-1)
                orientation = hold_info.get('orientation', 0)
                grid[y_idx, x_idx, 4] = orientation / max_orientation
                
                # Channel 5: Normalized depth (0-1)
                depth = hold_info.get('depth', 0)
                grid[y_idx, x_idx, 5] = depth / max_depth
                
                # Channel 6-10: Hold physical types (one-hot encoding)
                physical_type = hold_info.get('type', 'footchip').lower()
                
                # Map each hold type to its own channel
                hold_type_channels = {
                    'footchip': 6,
                    'jug': 7,
                    'sloper': 8,
                    'crimp': 9,
                    'pinch': 10
                }
                
                # Set the appropriate channel to 1
                if physical_type in hold_type_channels:
                    channel_idx = hold_type_channels[physical_type]
                    grid[y_idx, x_idx, channel_idx] = 1
                else:
                    # Default to footchip for unknown types
                    grid[y_idx, x_idx, 6] = 1

        return grid
    except Exception as e:
        print(f"Error creating enhanced grid: {e}")
        return np.zeros((grid_height, grid_width, 11))

def create_hold_feature_vector(placements_str, hold_data_df):
    """
    Create aggregated features from hold characteristics
    
    Parameters:
    -----------
    placements_str : str or list
        String representation of holds placements
    hold_data_df : pandas.DataFrame
        DataFrame containing hold characteristics
        
    Returns:
    --------
    list
        Aggregated hold features: [avg_orientation, avg_depth, std_orientation, std_depth, 
                                  start_holds, middle_holds, finish_holds, feet_only,
                                  footchip_count, jug_count, sloper_count, crimp_count, pinch_count]
    """
    try:
        placements = ast.literal_eval(placements_str) if isinstance(placements_str, str) else placements_str
        
        orientations = []
        depths = []
        hold_counts = {'START': 0, 'MIDDLE': 0, 'FINISH': 0, 'FEET-ONLY': 0}
        physical_type_counts = {'footchip': 0, 'jug': 0, 'sloper': 0, 'crimp': 0, 'pinch': 0}
        
        for hold in placements:
            led_position = hold.get('ledPosition')
            hold_type = hold.get('type', 'MIDDLE')
            
            # Count hold usage types
            hold_counts[hold_type] = hold_counts.get(hold_type, 0) + 1
            
            # Get physical characteristics
            if led_position is not None and led_position in hold_data_df.index:
                hold_info = hold_data_df.loc[led_position]
                orientations.append(hold_info.get('orientation', 0))
                depths.append(hold_info.get('depth', 0))
                
                # Count physical hold types
                physical_type = hold_info.get('type', 'footchip').lower()
                if physical_type in physical_type_counts:
                    physical_type_counts[physical_type] += 1
                else:
                    physical_type_counts['footchip'] += 1  # Default unknown to footchip
        
        # Calculate aggregated features
        avg_orientation = np.mean(orientations) if orientations else 0
        avg_depth = np.mean(depths) if depths else 0
        std_orientation = np.std(orientations) if len(orientations) > 1 else 0
        std_depth = np.std(depths) if len(depths) > 1 else 0
        
        return [
            avg_orientation / 360.0,  # Normalize
            avg_depth / 3.0,          # Normalize (assuming max depth ~3)
            std_orientation / 360.0,
            std_depth / 5.0,
            hold_counts['START'],
            hold_counts['MIDDLE'], 
            hold_counts['FINISH'],
            hold_counts['FEET-ONLY'],
            physical_type_counts['footchip'],
            physical_type_counts['jug'],
            physical_type_counts['sloper'],
            physical_type_counts['crimp'],
            physical_type_counts['pinch']
        ]
        
    except Exception as e:
        print(f"Error creating hold features: {e}")
        return [0] * 13  # Now 13 features instead of 8

def v_grade_distance(v1, v2):
    """
    Calculate distance between V-grades
    
    Parameters:
    -----------
    v1, v2 : str
        V-grade strings (e.g., 'V4', 'V6')
        
    Returns:
    --------
    int or float
        Distance between grades, or infinity if grades are invalid
    """
    v_scale = ["VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16+"]
    if v1 not in v_scale or v2 not in v_scale:
        return float('inf')
    return abs(v_scale.index(v1) - v_scale.index(v2))

def create_v_grade_confusion_matrix(actual, predicted):
    """
    Create and visualize confusion matrix for V-grades
    
    Parameters:
    -----------
    actual : list
        List of actual V-grades
    predicted : list
        List of predicted V-grades
        
    Returns:
    --------
    numpy.ndarray
        Confusion matrix
    """
    from sklearn.metrics import confusion_matrix

    v_scale = ["VB", "V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16+"]

    # Filter to only include grades that appear in the data
    unique_grades = sorted(list(set(actual + predicted)), key=lambda x: v_scale.index(x) if x in v_scale else -1)

    # Create confusion matrix
    cm = confusion_matrix(
        [unique_grades.index(a) if a in unique_grades else -1 for a in actual],
        [unique_grades.index(p) if p in unique_grades else -1 for p in predicted],
        labels=range(len(unique_grades))
    )

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=unique_grades,
               yticklabels=unique_grades)
    plt.xlabel('Predicted Grade')
    plt.ylabel('Actual Grade')
    plt.title('V-Grade Confusion Matrix')
    plt.tight_layout()

    plt.savefig("reports/figures/confusion_matrix.png", dpi=300, bbox_inches='tight')

    return cm

def create_cnn_model(df, hold_data_df, X_train, X_test, y_train, y_test):
    """
    Drop-in replacement for create_cnn_model that uses quality weighting during training
    Same signature as original function for easy replacement
    """
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    
    # Extract quality ratings
    from ..models.weighted_metrics import map_quality_to_clean_data, create_quality_weights
    from ..models.cnn_model import create_multichannel_grid, create_hold_feature_vector

    # Load original data to get climb_stats (you might need to adjust this path)
    from ..data.preprocessing import load_data
    try:
        original_df = load_data("data/raw/climbs.csv")
        climb_stats = original_df['climb_stats']
        
        # Extract quality for train/test sets
        all_quality = map_quality_to_clean_data(climb_stats, df, original_df)
        train_quality = all_quality.loc[X_train.index]
        test_quality = all_quality.loc[X_test.index]
        
        print(f"Quality weighting enabled - Train mean: {train_quality.mean():.3f}, Test mean: {test_quality.mean():.3f}")
        
    except Exception as e:
        print(f"Could not load quality data: {e}")
        print("Training without quality weighting...")
        train_quality = None
        test_quality = None
    
    # Create quality weights for training if available
    if train_quality is not None:
        train_weights = create_quality_weights(train_quality, 
                                             weight_function='exponential',
                                             min_weight=0.1, 
                                             max_weight=3.0)
        use_weights = True
        print(f"Using quality weights - Range: {train_weights.min():.3f} to {train_weights.max():.3f}")
    else:
        train_weights = None
        use_weights = False
        print("No quality weights - using standard training")
    
    # Process features (same as original)
    feature_cols = ['angle', 'hold_count', 'ascents']
    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    # Create multichannel grids and features
    train_grids = np.array([create_multichannel_grid(p, hold_data_df) for p in X_train['placements']])
    test_grids = np.array([create_multichannel_grid(p, hold_data_df) for p in X_test['placements']])

    train_hold_features = np.array([create_hold_feature_vector(p, hold_data_df) for p in X_train['placements']])
    test_hold_features = np.array([create_hold_feature_vector(p, hold_data_df) for p in X_test['placements']])

    train_features = np.hstack((X_train_scaled[feature_cols].values, train_hold_features))
    test_features = np.hstack((X_test_scaled[feature_cols].values, test_hold_features))

    # Build the CNN model (exact same architecture as original)
    grid_input = layers.Input(shape=train_grids.shape[1:])

    # CNN layers for the grid
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Flatten()(x)

    # Input for numerical features
    numerical_input = layers.Input(shape=(len(feature_cols) + 13,))

    # Dense layers for numerical features
    y = layers.Dense(64, activation='relu')(numerical_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.2)(y)
    y = layers.Dense(32, activation='relu')(y)

    # Combine the CNN output with numerical features
    combined = layers.concatenate([x, y])

    # Dense layers for combined data
    z = layers.Dense(512, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(256, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation='relu')(z)

    # Output layer
    output = layers.Dense(1)(z)

    # Create model
    model = models.Model(inputs=[grid_input, numerical_input], outputs=output)

    # Create Optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=800,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Display model summary
    print("CNN Model Architecture (with quality weighting):")
    model.summary()

    # Callbacks for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True)

    # Train the model - WITH OR WITHOUT WEIGHTS
    if use_weights:
        print("Training with quality-based sample weights...")
        history = model.fit(
            [train_grids, train_features],
            y_train,
            sample_weight=train_weights,  # THE KEY DIFFERENCE!
            epochs=15,  # Slightly more epochs since we're using weights
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping]
        )
    else:
        print("Training without sample weights (standard training)...")
        history = model.fit(
            [train_grids, train_features],
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            verbose=1,
            callbacks=[early_stopping]
        )

    # Evaluate on test set
    test_results = model.evaluate([test_grids, test_features], y_test, verbose=1)

    # Make predictions
    predictions = model.predict([test_grids, test_features])

    # Calculate additional metrics (same as original)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print("\nModel Evaluation:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Convert predictions to V-grade scale for easier interpretation
    v_grade_predictions = [difficulty_to_vgrade(p[0]) for p in predictions]
    v_grade_actual = [difficulty_to_vgrade(g) for g in y_test]

    # Calculate accuracy in terms of exact V-grade matches
    exact_matches = sum(p == a for p, a in zip(v_grade_predictions, v_grade_actual))
    v_grade_accuracy = exact_matches / len(v_grade_predictions)

    print(f"V-grade Exact Match Accuracy: {v_grade_accuracy:.4f}")

    # Calculate accuracy within ±1 V-grade
    within_one = sum(v_grade_distance(p, a) <= 1 for p, a in zip(v_grade_predictions, v_grade_actual))
    v_grade_accuracy_one = within_one / len(v_grade_predictions)

    print(f"V-grade ±1 Accuracy: {v_grade_accuracy_one:.4f}")

    # Create confusion matrix
    try:
        cm = create_v_grade_confusion_matrix(v_grade_actual, v_grade_predictions)
        print("Confusion matrix generated successfully")
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")

    # If we have quality data, also print weighted metrics
    if test_quality is not None:
        try:
            from ..models.weighted_metrics import comprehensive_weighted_evaluation
            weighted_results = comprehensive_weighted_evaluation(y_test, predictions.flatten(), test_quality)
            
            print(f"\nQuality-Weighted Metrics:")
            print(f"Weighted Accuracy (exact): {weighted_results['weighted_accuracy_exact']:.4f}")
            print(f"Weighted Accuracy (±1): {weighted_results['weighted_accuracy_pm1']:.4f}")
            print(f"Weighted MAE: {weighted_results['weighted_mae']:.4f}")
            
            # Show improvement
            acc_improvement = weighted_results['weighted_accuracy_exact'] - v_grade_accuracy
            print(f"Quality weighting accuracy improvement: {acc_improvement:+.4f}")
            
        except Exception as e:
            print(f"Could not calculate weighted metrics: {e}")

    # Return metrics (same format as original function)
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'v_grade_accuracy': v_grade_accuracy,
        'v_grade_accuracy_one': v_grade_accuracy_one
    }

    return model, history, metrics