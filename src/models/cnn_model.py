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

def create_multichannel_grid(placements_str, grid_width=24, grid_height=18):
    """
    Create a multi-channel 2D grid representing different hold types
    
    Parameters:
    -----------
    placements_str : str or list
        String representation of holds placements or the parsed list
    grid_width : int
        Width of the grid (default: 24)
    grid_height : int
        Height of the grid (default: 18)
        
    Returns:
    --------
    numpy.ndarray
        Multi-channel grid representation of holds (shape: grid_height, grid_width, 4)
    """
    # Initialize empty grid with 4 channels (START, MIDDLE, FINISH, FEET-ONLY)
    grid = np.zeros((grid_height, grid_width, 4))

    try:
        # Parse placements
        placements = ast.literal_eval(placements_str) if isinstance(placements_str, str) else placements_str

        # Channel mappings
        type_to_channel = {
            'START': 0,
            'MIDDLE': 1,
            'FINISH': 2,
            'FEET-ONLY': 3
        }

        # Fill grid with hold placements
        for hold in placements:
            x = hold.get('x', 0)
            y = hold.get('y', 0)

            # Normalize to grid indices
            x_idx = min(int(x * (grid_width - 1) / 24), grid_width - 1)
            y_idx = min(int(y * (grid_height - 1) / 36), grid_height - 1)

            # Get hold type and map to channel
            hold_type = hold.get('type', '')
            channel = type_to_channel.get(hold_type, 1)  # Default to MIDDLE

            # Mark hold position in appropriate channel
            grid[y_idx, x_idx, channel] = 1

        return grid
    except:
        return np.zeros((grid_height, grid_width, 4))

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

    return cm

def create_cnn_model(df, X_train, X_test, y_train, y_test):
    """
    Create and train an improved CNN model to predict boulder grades using LED positions.

    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame with all data
    X_train, X_test, y_train, y_test : pandas.DataFrame/Series
        Train/test split data

    Returns:
    --------
    tuple
        (model, history, evaluation_metrics)
    """
    # Process features
    # Create both standard grid and multichannel grid
    feature_cols = ['angle', 'hold_count', 'ascents']
    scaler = StandardScaler()

    # Normalize numerical features
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    #scale numerical features to put them all on the same scale (standardization)
    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    #use SAME parameters for standardization on test data
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    # Create multichannel grids from placements
    train_grids = np.array([create_multichannel_grid(p) for p in X_train['placements']])
    test_grids = np.array([create_multichannel_grid(p) for p in X_test['placements']])

    # Create hold type encodings
    train_hold_types = np.array([encode_hold_types(p) for p in X_train['placements']])
    test_hold_types = np.array([encode_hold_types(p) for p in X_test['placements']])

    # Create input for numerical features
    train_features = np.array(X_train_scaled[feature_cols])
    test_features = np.array(X_test_scaled[feature_cols])

    # Combine with hold type encodings
    train_features = np.hstack((train_features, train_hold_types))
    test_features = np.hstack((test_features, test_hold_types))

    # Build the CNN model with both image and numerical inputs
    # Input for the grid
    grid_input = layers.Input(shape=train_grids.shape[1:])

    # CNN layers for the grid
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(grid_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)

    # Input for numerical features
    numerical_input = layers.Input(shape=(len(feature_cols) + 4,))  # +4 for hold type encodings

    # Dense layers for numerical features
    y = layers.Dense(32, activation='relu')(numerical_input)

    # Combine the CNN output with numerical features
    combined = layers.concatenate([x, y])

    # Dense layers for combined data
    z = layers.Dense(256, activation='relu')(combined)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dropout(0.2)(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dropout(0.2)(z)

    # Output layer (predicting a single continuous value)
    output = layers.Dense(1)(z)

    # Create model
    model = models.Model(inputs=[grid_input, numerical_input], outputs=output)

    # Create Optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Display model summary
    model.summary()

    # Callbacks for early stopping and model checkpoint
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True)

    # Train the model
    history = model.fit(
        [train_grids, train_features],
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[early_stopping]
    )

    # Evaluate on test set
    test_results = model.evaluate(
        [test_grids, test_features],
        y_test,
        verbose=1
    )

    # Make predictions
    predictions = model.predict([test_grids, test_features])

    # Calculate additional metrics
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

    # Return model, history and metrics
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'v_grade_accuracy': v_grade_accuracy,
        'v_grade_accuracy_one': v_grade_accuracy_one
    }

    return model, history, metrics