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
        
        
def create_weighted_cnn_model(df, hold_data_df, X_train, X_test, y_train, y_test, 
                             use_weighted_loss=True, weight_function='exponential'):
    """
    Enhanced version of create_cnn_model that incorporates quality weighting during training
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The original DataFrame with all data
    hold_data_df : pandas.DataFrame
        DataFrame containing hold characteristics
    X_train, X_test, y_train, y_test : pandas.DataFrame/Series
        Train/test split data
    use_weighted_loss : bool
        Whether to use quality-weighted loss during training
    weight_function : str
        Type of weighting function ('linear', 'exponential', 'sigmoid')
        
    Returns:
    --------
    tuple
        (model, history, metrics, weighted_metrics)
    """


    from src.models.cnn_model import (
        create_multichannel_grid, create_hold_feature_vector,
        create_v_grade_confusion_matrix, difficulty_to_vgrade
    )
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from tensorflow.keras import layers, models
    import matplotlib.pyplot as plt
    
    # Extract quality ratings for train and test sets
    train_quality = extract_quality_ratings(df.loc[X_train.index])
    test_quality = extract_quality_ratings(df.loc[X_test.index])
    
    # Create quality weights
    train_weights = create_quality_weights(train_quality, weight_function=weight_function)
    test_weights = create_quality_weights(test_quality, weight_function=weight_function)
    
    print(f"Quality weighting - Train mean: {np.mean(train_weights):.3f}, Test mean: {np.mean(test_weights):.3f}")
    
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
    
    # Build the CNN model (same architecture as original)
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
    
    # Choose loss function based on weighting preference
    if use_weighted_loss:
        # Use sample weights during training
        loss_function = 'mean_squared_error'  # We'll pass sample_weight to fit()
        print("Using quality-weighted training with sample weights")
    else:
        loss_function = 'mean_squared_error'
        print("Using standard unweighted training")
    
    # Create optimizer with learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=800,
        decay_rate=0.95
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compile model
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['mean_absolute_error'])
    
    print("Enhanced CNN Model Architecture:")
    model.summary()
    
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )
    
    # Train the model
    if use_weighted_loss:
        # Train with sample weights
        history = model.fit(
            [train_grids, train_features],
            y_train,
            epochs=10,
            batch_size=32,
            validation_split=0.2,
            sample_weight=train_weights,  # This is the key difference!
            verbose=1,
            callbacks=[early_stopping]
        )
    else:
        # Train without sample weights
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
    predictions = model.predict([test_grids, test_features]).flatten()
    
    # Calculate standard metrics
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    # Calculate weighted metrics
    from src.models.weighted_metrics import comprehensive_weighted_evaluation
    weighted_results = comprehensive_weighted_evaluation(
        y_test, predictions, test_quality, weight_function=weight_function
    )
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("ENHANCED CNN MODEL EVALUATION")
    print("="*60)
    print(f"Training Method: {'Quality-Weighted' if use_weighted_loss else 'Standard'}")
    print(f"Weight Function: {weight_function}")
    
    print("\nStandard Metrics:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    print("\nV-Grade Accuracies:")
    print(f"  Exact Match: {weighted_results['accuracy_exact']:.4f}")
    print(f"  ±1 Grade: {weighted_results['accuracy_pm1']:.4f}")
    
    print("\nQuality-Weighted Metrics:")
    print(f"  Weighted MAE: {weighted_results['weighted_mae']:.4f}")
    print(f"  Weighted Accuracy (exact): {weighted_results['weighted_accuracy_exact']:.4f}")
    print(f"  Weighted Accuracy (±1): {weighted_results['weighted_accuracy_pm1']:.4f}")
    
    # Calculate improvements
    accuracy_improvement = weighted_results['weighted_accuracy_exact'] - weighted_results['accuracy_exact']
    mae_improvement = weighted_results['mae'] - weighted_results['weighted_mae']
    
    print("\nQuality Weighting Impact:")
    print(f"  Accuracy Improvement: {accuracy_improvement:+.4f}")
    print(f"  MAE Improvement: {mae_improvement:+.4f}")
    
    # Create confusion matrix for V-grades
    v_grade_predictions = [difficulty_to_vgrade(p) for p in predictions]
    v_grade_actual = [difficulty_to_vgrade(g) for g in y_test]
    
    try:
        cm = create_v_grade_confusion_matrix(v_grade_actual, v_grade_predictions)
        print("\nConfusion matrix generated successfully")
    except Exception as e:
        print(f"Could not generate confusion matrix: {e}")
    
    # Compile all metrics
    standard_metrics = {
        'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2,
        'accuracy_exact': weighted_results['accuracy_exact'],
        'accuracy_pm1': weighted_results['accuracy_pm1']
    }
    
    return model, history, standard_metrics, weighted_results

def compare_weighted_vs_unweighted_training(df, hold_data_df, X_train, X_test, y_train, y_test):
    """
    Compare models trained with and without quality weighting
    
    Returns:
    --------
    dict
        Comparison results
    """
    print("Training Standard Model...")
    model_std, history_std, metrics_std, weighted_metrics_std = create_weighted_cnn_model(
        df, hold_data_df, X_train, X_test, y_train, y_test, 
        use_weighted_loss=False, weight_function='exponential'
    )
    
    print("\n" + "="*60)
    print("Training Quality-Weighted Model...")
    model_weighted, history_weighted, metrics_weighted, weighted_metrics_weighted = create_weighted_cnn_model(
        df, hold_data_df, X_train, X_test, y_train, y_test, 
        use_weighted_loss=True, weight_function='exponential'
    )
    
    # Create comparison
    comparison = {
        'standard_model': {
            'model': model_std,
            'history': history_std,
            'metrics': metrics_std,
            'weighted_metrics': weighted_metrics_std
        },
        'weighted_model': {
            'model': model_weighted,
            'history': history_weighted,
            'metrics': metrics_weighted,
            'weighted_metrics': weighted_metrics_weighted
        }
    }
    
    # Print comparison
    print("\n" + "="*80)
    print("FINAL COMPARISON: STANDARD vs QUALITY-WEIGHTED TRAINING")
    print("="*80)
    
    print(f"{'Metric':<25} {'Standard':<15} {'Weighted':<15} {'Improvement':<15}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('MAE', 'mae'),
        ('RMSE', 'rmse'),
        ('R²', 'r2'),
        ('Accuracy (exact)', 'accuracy_exact'),
        ('Accuracy (±1)', 'accuracy_pm1'),
        ('Weighted Accuracy (exact)', 'weighted_accuracy_exact'),
        ('Weighted Accuracy (±1)', 'weighted_accuracy_pm1')
    ]
    
    for display_name, metric_key in metrics_to_compare:
        if metric_key in metrics_std:
            std_val = metrics_std[metric_key]
            weighted_val = metrics_weighted[metric_key]
        else:
            std_val = weighted_metrics_std[metric_key]
            weighted_val = weighted_metrics_weighted[metric_key]
        
        improvement = weighted_val - std_val
        print(f"{display_name:<25} {std_val:<15.4f} {weighted_val:<15.4f} {improvement:<15.4f}")
    
    return comparison