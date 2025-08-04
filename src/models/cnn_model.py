# src/models/cnn_model.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from ..features.grade_conversion import difficulty_to_vgrade, vgrade_to_difficulty
from collections import Counter
from src.models.feature_analysis import comprehensive_feature_analysis


def create_simple_balanced_loss(grade_counts_dict):
    """custom weighting that gets more aggressive for rarer grades"""
    
    total_samples = sum(grade_counts_dict.values())
    n_classes = len(grade_counts_dict)
    
    difficulties = []
    weights = []
    
    for difficulty in range(0, 45):
        vgrade = difficulty_to_vgrade(difficulty)
        count = grade_counts_dict.get(vgrade, 1)
        
        # Calculate base weight
        base_weight = total_samples / (n_classes * count)
        
        # Progressive scaling: more aggressive for rarer grades
        if base_weight <= 2.0:
            # Common grades: minimal adjustment
            weight = base_weight
        elif base_weight <= 5.0:
            # Uncommon grades: moderate boost
            weight = base_weight * 1.2
        else:
            # Rare grades: more aggressive boost
            weight = base_weight * 1.5
        
        # Higher cap but still reasonable
        weight = min(weight, 10.0)
        
        difficulties.append(float(difficulty))
        weights.append(weight)
    
    difficulty_tensor = tf.constant(difficulties, dtype=tf.float32)
    weight_tensor = tf.constant(weights, dtype=tf.float32)
    
    def progressive_weighted_mse_loss(y_true, y_pred):
        base_mse = tf.square(y_true - y_pred)
        y_true_rounded = tf.round(tf.clip_by_value(y_true, 0.0, 44.0))
        indices = tf.cast(y_true_rounded, tf.int32)
        sample_weights = tf.gather(weight_tensor, indices)
        weighted_mse = base_mse * tf.expand_dims(sample_weights, -1)
        return tf.reduce_mean(weighted_mse)
    
    return progressive_weighted_mse_loss


def encode_hold_types(placements_str):
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
    # Initialize grid with 11 channels
    # 0-3: hold types (START, MIDDLE, FINISH, FEET-ONLY)
    # 4: orientation (normalized)
    # 5: depth (normalized) 
    # 6-10: physical hold types (footchip, jug, sloper, crimp, pinch)
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

            # Fix negative coordinates by taking absolute values
            x = abs(x)
            y = abs(y)
                
            # Normalize to grid indices with proper bounds checking
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
        return [0] * 13


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


def create_cnn_model(df, hold_data_df, X_train, X_test, y_train, y_test, loss_name="mean_squared_error"):
    """
    Create and train CNN model with configurable loss function
    """
    
    # Process features
    feature_cols = ['angle', 'hold_count', 'ascents', 'quality_average']
    
    X_train_clipped = X_train.copy()
    X_test_clipped = X_test.copy()
    
    # Clip ascents to 99th percentile to handle extreme outliers
    ascents_cap = X_train['ascents'].quantile(0.99)
    affected_train = (X_train['ascents'] > ascents_cap).sum()
    affected_test = (X_test['ascents'] > ascents_cap).sum()
    
    print(f"Clipping ascents above {ascents_cap:.1f}")
    print(f"  Affects {affected_train} training samples ({affected_train/len(X_train)*100:.1f}%)")
    print(f"  Affects {affected_test} test samples ({affected_test/len(X_test)*100:.1f}%)")
    
    X_train_clipped['ascents'] = X_train['ascents'].clip(upper=ascents_cap)
    X_test_clipped['ascents'] = X_test['ascents'].clip(upper=ascents_cap)
    
    
    scaler = StandardScaler()

    # Normalize numerical features
    
    X_train_scaled = X_train_clipped.copy()
    X_test_scaled = X_test_clipped.copy()

    # Scale numerical features to put them all on the same scale (standardization)
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train_clipped[feature_cols])
    # Use SAME parameters for standardization on test data
    X_test_scaled[feature_cols] = scaler.transform(X_test_clipped[feature_cols])

    # Verify scaling worked properly
    print("Feature ranges after clipping + scaling:")
    for col in feature_cols:
        col_min = X_train_scaled[col].min()
        col_max = X_train_scaled[col].max()
        print(f"  {col}: {col_min:.2f} to {col_max:.2f}")
        if abs(col_max) > 10 or abs(col_min) > 10:
            print(f"    ⚠️  WARNING: {col} still has extreme values!")
        else:
            print(f"    ✓ {col} looks good!")
   

    # Create multichannel grids from placements
    train_grids = np.array([create_multichannel_grid(p, hold_data_df) for p in X_train['placements']])
    test_grids = np.array([create_multichannel_grid(p, hold_data_df) for p in X_test['placements']])

    # Create enhanced hold feature vectors
    train_hold_features = np.array([create_hold_feature_vector(p, hold_data_df) for p in X_train['placements']])
    test_hold_features = np.array([create_hold_feature_vector(p, hold_data_df) for p in X_test['placements']])
    
    # Combine with basic numerical features
    train_features = np.hstack((X_train_scaled[feature_cols].values, train_hold_features))
    test_features = np.hstack((X_test_scaled[feature_cols].values, test_hold_features))

    # Build the CNN model with both image and numerical inputs
    # Input for the grid
    grid_input = layers.Input(shape=train_grids.shape[1:])

    # CNN layers for the grid
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.05)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.05)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)

    # Input for numerical features
    numerical_input = layers.Input(shape=(len(feature_cols) + 13,))

    # Dense layers for numerical features
    y = layers.Dense(64, activation='relu')(numerical_input)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.1)(y)
    y = layers.Dense(32, activation='relu')(y)

    # Combine the CNN output with numerical features
    combined = layers.concatenate([x, y])

    # Dense layers for combined data
    z = layers.Dense(1024, activation='relu')(combined)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(512, activation='relu')(z)
    z = layers.BatchNormalization()(z)
    z = layers.Dropout(0.2)(z)

    z = layers.Dense(256, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation='relu')(z)

    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1)(z)  
    
    # Create model
    model = models.Model(inputs=[grid_input, numerical_input], outputs=output)

    # Create Optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=800,
        decay_rate=0.95)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # Choose the loss function 
    if loss_name == "custom_loss":
        # Create custom weighted loss function
        # Calculate grade distribution for weighted loss
        grade_distribution = Counter([difficulty_to_vgrade(g) for g in y_train])
        print(f"Grade distribution: {dict(grade_distribution)}")

        loss_function = create_simple_balanced_loss(grade_distribution)
        loss_display_name = "Custom Weighted MSE"
    else:
        # Use standard loss
        loss_function = loss_name
        loss_display_name = loss_name.replace('_', ' ').title()

    # Compile model with chosen loss
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['mean_absolute_error'])

    # Display model summary with correct loss name
    print(f"CNN Model Architecture using {loss_display_name} loss:")
    model.summary()

    # Callbacks for early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True)

    # Train with correct loss name display
    print(f"Training with {loss_display_name} loss")
    history = model.fit(
        [train_grids, train_features],
        y_train,
        epochs=25,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
        )
    
    # analyze feature importance
    numerical_imp, channel_imp = comprehensive_feature_analysis(
    model, test_grids, test_features, y_test
    )

    # Evaluate on test set
    test_results = model.evaluate([test_grids, test_features], y_test, verbose=1)

    # Make predictions
    predictions = model.predict([test_grids, test_features])
    print(f"Predictions shape: {predictions.shape}")
    
    # Calculate additional metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"\nModel Evaluation using {loss_display_name}:")
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

    # Return model, history and metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'v_grade_accuracy': v_grade_accuracy,
        'v_grade_accuracy_one': v_grade_accuracy_one
    }

    return model, history, metrics, loss_display_name  # Return the display name