# test_clipping_fix.py - Test the clipping solution before implementing

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
from src.data.preprocessing import load_data
from src.models.preprocessing import create_train_test_split

def test_clipping_solution():
    """Test that clipping fixes the scaling issue and improves model training"""
    
    print("="*60)
    print("TESTING CLIPPING FIX")
    print("="*60)
    
    # Load data
    boulder_angles_df = load_data("data/processed/cleaner_quality_data.csv")
    X_train, X_test, y_train, y_test = create_train_test_split(boulder_angles_df)
    
    feature_cols = ['angle', 'hold_count', 'ascents', 'quality_average']
    
    # Test 1: WITHOUT clipping (current broken state)
    print("TEST 1: WITHOUT CLIPPING (current broken state)")
    print("-" * 40)
    
    scaler_broken = StandardScaler()
    X_train_broken = X_train[feature_cols].copy()
    X_train_broken_scaled = pd.DataFrame(
        scaler_broken.fit_transform(X_train_broken),
        columns=feature_cols
    )
    
    print("Scaling results WITHOUT clipping:")
    for col in feature_cols:
        col_min = X_train_broken_scaled[col].min()
        col_max = X_train_broken_scaled[col].max()
        print(f"  {col}: {col_min:.2f} to {col_max:.2f}")
        if abs(col_max) > 10:
            print(f"    ⚠️  {col} has extreme values!")
    
    # Test 2: WITH clipping (proposed fix)
    print(f"\nTEST 2: WITH CLIPPING (proposed fix)")
    print("-" * 40)
    
    # Apply clipping
    X_train_clipped = X_train[feature_cols].copy()
    X_test_clipped = X_test[feature_cols].copy()
    
    # Clip ascents to 99th percentile
    ascents_cap = X_train['ascents'].quantile(0.99)
    affected_train = (X_train['ascents'] > ascents_cap).sum()
    affected_test = (X_test['ascents'] > ascents_cap).sum()
    
    print(f"Clipping ascents above {ascents_cap:.1f}")
    print(f"  Affects {affected_train} training samples ({affected_train/len(X_train)*100:.1f}%)")
    print(f"  Affects {affected_test} test samples ({affected_test/len(X_test)*100:.1f}%)")
    
    X_train_clipped['ascents'] = X_train['ascents'].clip(upper=ascents_cap)
    X_test_clipped['ascents'] = X_test['ascents'].clip(upper=ascents_cap)
    
    # Scale the clipped data
    scaler_fixed = StandardScaler()
    X_train_fixed_scaled = pd.DataFrame(
        scaler_fixed.fit_transform(X_train_clipped),
        columns=feature_cols
    )
    
    print(f"\nScaling results WITH clipping:")
    all_good = True
    for col in feature_cols:
        col_min = X_train_fixed_scaled[col].min()
        col_max = X_train_fixed_scaled[col].max()
        print(f"  {col}: {col_min:.2f} to {col_max:.2f}")
        if abs(col_max) > 10 or abs(col_min) > 10:
            print(f"    ⚠️  {col} still has extreme values!")
            all_good = False
        else:
            print(f"    ✓ {col} looks good!")
    
    if all_good:
        print(f"\n🎉 SUCCESS: All features now have reasonable ranges!")
    else:
        print(f"\n❌ PROBLEM: Some features still have extreme values")
        return
    
    # Test 3: Train simple models with both approaches
    print(f"\nTEST 3: TRAINING COMPARISON")
    print("-" * 40)
    
    # Prepare data for simple models (use subset for speed)
    subset_size = 1000
    
    # Broken model (without clipping)
    X_broken_subset = X_train_broken_scaled.iloc[:subset_size].values
    y_broken_subset = y_train.iloc[:subset_size].values
    
    # Fixed model (with clipping)  
    X_fixed_subset = X_train_fixed_scaled.iloc[:subset_size].values
    y_fixed_subset = y_train.iloc[:subset_size].values
    
    # Create identical simple models
    def create_simple_model():
        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(len(feature_cols),)),
            layers.BatchNormalization(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        return model
    
    print("Training model WITHOUT clipping...")
    model_broken = create_simple_model()
    
    # Test initial predictions for broken model
    initial_preds_broken = model_broken.predict(X_broken_subset[:10])
    initial_loss_broken = np.mean((initial_preds_broken.flatten() - y_broken_subset[:10])**2)
    print(f"  Initial MSE (broken): {initial_loss_broken:.2f}")
    
    # Train broken model
    try:
        history_broken = model_broken.fit(
            X_broken_subset, y_broken_subset,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        final_loss_broken = history_broken.history['loss'][-1]
        final_mae_broken = history_broken.history['mean_absolute_error'][-1]
        print(f"  Final loss (broken): {final_loss_broken:.2f}")
        print(f"  Final MAE (broken): {final_mae_broken:.2f}")
    except Exception as e:
        print(f"  Broken model training failed: {e}")
        final_loss_broken = float('inf')
        final_mae_broken = float('inf')
    
    print(f"\nTraining model WITH clipping...")
    model_fixed = create_simple_model()
    
    # Test initial predictions for fixed model
    initial_preds_fixed = model_fixed.predict(X_fixed_subset[:10])
    initial_loss_fixed = np.mean((initial_preds_fixed.flatten() - y_fixed_subset[:10])**2)
    print(f"  Initial MSE (fixed): {initial_loss_fixed:.2f}")
    
    # Train fixed model
    try:
        history_fixed = model_fixed.fit(
            X_fixed_subset, y_fixed_subset,
            epochs=5,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        final_loss_fixed = history_fixed.history['loss'][-1]
        final_mae_fixed = history_fixed.history['mean_absolute_error'][-1]
        print(f"  Final loss (fixed): {final_loss_fixed:.2f}")
        print(f"  Final MAE (fixed): {final_mae_fixed:.2f}")
    except Exception as e:
        print(f"  Fixed model training failed: {e}")
        final_loss_fixed = float('inf')
        final_mae_fixed = float('inf')
    
    # Compare results
    print(f"\nCOMPARISON RESULTS:")
    print("=" * 40)
    print(f"Initial MSE:")
    print(f"  Without clipping: {initial_loss_broken:.2f}")
    print(f"  With clipping:    {initial_loss_fixed:.2f}")
    print(f"  Improvement:      {initial_loss_broken - initial_loss_fixed:+.2f}")
    
    print(f"\nFinal MAE after 5 epochs:")
    print(f"  Without clipping: {final_mae_broken:.2f}")
    print(f"  With clipping:    {final_mae_fixed:.2f}")
    print(f"  Improvement:      {final_mae_broken - final_mae_fixed:+.2f}")
    
    if initial_loss_fixed < initial_loss_broken / 2:
        print(f"\n🎉 CLIPPING FIX WORKS! Initial loss improved significantly")
    else:
        print(f"\n⚠️  Clipping helped but may not be sufficient")
    
    if final_mae_fixed < final_mae_broken * 0.8:
        print(f"🎉 TRAINING IMPROVED! Final MAE is much better with clipping")
    else:
        print(f"⚠️  Training improvement is marginal")
    
    # Show expected ranges for your main model
    print(f"\nEXPECTED RESULTS FOR YOUR MAIN CNN MODEL:")
    print("=" * 50)
    print(f"With this clipping fix, you should see:")
    print(f"  ✓ Initial loss: ~8-15 (instead of ~190)")
    print(f"  ✓ Smooth training curve (no massive drops)")
    print(f"  ✓ Better final performance")
    print(f"  ✓ All features in reasonable -5 to +5 range")
    
    print(f"\nREADY TO IMPLEMENT:")
    print("=" * 20)
    print("If the results above look good, you can safely add the clipping")
    print("code to your main create_cnn_model() function!")

if __name__ == "__main__":
    test_clipping_solution()