# src/models/weighted_training.py

import tensorflow as tf
import numpy as np
from src.models.weighted_metrics import create_quality_weights, extract_quality_ratings

class WeightedMeanSquaredError(tf.keras.losses.Loss):
    """
    Custom weighted MSE loss that gives more importance to high-quality boulders
    """
    def __init__(self, sample_weights=None, name="weighted_mse"):
        super().__init__(name=name)
        self.sample_weights = sample_weights
    
    def call(self, y_true, y_pred):
        mse = tf.square(y_true - y_pred)
        
        if self.sample_weights is not None:
            # Apply sample weights
            weighted_mse = mse * self.sample_weights
            return tf.reduce_mean(weighted_mse)
        else:
            return tf.reduce_mean(mse)
        
        
# Add this function to src/models/cnn_model.py or create as separate file

def create_weighted_cnn_model(df, hold_data_df, X_train, X_test, y_train, y_test):
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