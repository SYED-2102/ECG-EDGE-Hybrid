import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def load_and_split_data(csv_path='data/mitbih_train.csv'):
    print(f"Loading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path, header=None)
    except FileNotFoundError:
        print(f"\n[ERROR] File not found: {csv_path}")
        return None, None, None, None

    # Shuffle to distribute classes evenly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    X = df.iloc[:, :187].values
    y = df.iloc[:, 187].values
    
    # Binary Conversion: 0 is Normal, >0 is Anomaly
    y = (y > 0).astype(int)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # CRITICAL FIX: You MUST have a hold-out test set for valid IEEE benchmarks
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def build_model(architecture_name, input_shape, pool_size=None):
    inputs = tf.keras.Input(shape=input_shape)
    
    if architecture_name == "Baseline_CNN":
        x = tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
    elif architecture_name == "Baseline_LSTM":
        x = tf.keras.layers.LSTM(16)(inputs)
        
    elif architecture_name == "Baseline_GRU":
        x = tf.keras.layers.GRU(16)(inputs)
        
    elif architecture_name == "Hybrid_Conv1D_GRU":
        # Your proposed architecture
        x = tf.keras.layers.Conv1D(filters=16, kernel_size=5, activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=pool_size)(x)
        x = tf.keras.layers.GRU(16)(x)
    else:
        raise ValueError("Unknown architecture")

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=architecture_name)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print(f" TRAINING & EVALUATING: {model.name}")
    print(f"{'='*60}")
    
    model.summary()

    # CRITICAL FIX: Early Stopping to prove convergence, not hardcoded epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=10, 
        restore_best_weights=True,
        verbose=1
    )
    
    start_time = time.time()
    history = model.fit(
        X_train, y_train, 
        epochs=100, # High max epochs, early stopping will halt it naturally
        validation_split=0.2, 
        batch_size=64, 
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - start_time
    
    # Test Set Evaluation
    print(f"\nEvaluating {model.name} on unseen test data...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    # F1 Score Calculation
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)
    
    # Inference Time Latency (Simulating real-time edge processing)
    # Measure time for 1000 samples to get a stable average
    inf_samples = X_test[:1000]
    inf_start = time.time()
    model.predict(inf_samples, verbose=0)
    inf_end = time.time()
    ms_per_inference = ((inf_end - inf_start) / len(inf_samples)) * 1000

    print(f"\n--- RESULTS: {model.name} ---")
    print(f"Test Accuracy      : {test_acc * 100:.2f}%")
    print(f"Test F1-Score      : {f1 * 100:.2f}%")
    print(f"Total Params       : {model.count_params()}")
    print(f"Inference Latency  : {ms_per_inference:.4f} ms / sample")
    
    return {
        "Model": model.name,
        "Accuracy": test_acc,
        "F1_Score": f1,
        "Params": model.count_params(),
        "Inference_ms": ms_per_inference,
        "Total_Epochs_Run": len(history.epoch)
    }

if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)

    X_train, X_test, y_train, y_test = load_and_split_data('data/mitbih_train.csv')

    if X_train is not None:
        results = []
        input_shape = (X_train.shape[1], X_train.shape[2])

        # 1. Evaluate Core Baselines
        baselines = ["Baseline_CNN", "Baseline_LSTM", "Baseline_GRU"]
        for b in baselines:
            model = build_model(b, input_shape)
            res = train_and_evaluate(model, X_train, y_train, X_test, y_test)
            results.append(res)
            
        # 2. Evaluate Proposed Architecture & Ablation Study (P=4, 8, 32, 128)
        pool_sizes = [4, 8, 32, 128]
        for p in pool_sizes:
            model = build_model("Hybrid_Conv1D_GRU", input_shape, pool_size=p)
            model._name = f"Hybrid_Conv1D_GRU_P{p}"
            res = train_and_evaluate(model, X_train, y_train, X_test, y_test)
            results.append(res)
            
        # Print Final Markdown Table for Paper
        print("\n" + "#"*80)
        print(" FINAL BENCHMARK METRICS FOR IEEE TABLE")
        print("#"*80)
        print(f"{'Model':<25} | {'Accuracy (%)':<12} | {'F1-Score (%)':<12} | {'Params':<8} | {'Inference (ms)':<15}")
        print("-" * 80)
        for r in results:
            print(f"{r['Model']:<25} | {r['Accuracy']*100:<12.2f} | {r['F1_Score']*100:<12.2f} | {r['Params']:<8} | {r['Inference_ms']:<15.4f}")
