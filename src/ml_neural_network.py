import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from src.ml_base import (preprocess_data, evaluate_regression,
                         plot_predictions, plot_residuals)

def train_neural_network(df):
    print("\n" + " "*80)
    print(" DEEP NEURAL NETWORK MODEL")
    print(" "*80)
    
    # Preprocess
    X_train, X_test, y_train, y_test, scaler, features = preprocess_data(df)
    
    # Get input dimension
    input_dim = X_train.shape[1]
    
    print(f"\nBuilding Neural Network...")
    print(f"Input features: {input_dim}")
    
    # Build model
    model = keras.Sequential([
        # Input layer + First hidden layer
        layers.Dense(
            64,                                    # 64 neurons
            activation='relu',                     # ReLU activation
            input_shape=(input_dim,),
            kernel_regularizer=regularizers.l2(0.001)  # L2 regularization
        ),
        layers.Dropout(0.2),                      # Drop 20% to prevent overfitting
        
        # Second hidden layer
        layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.Dropout(0.2),
        
        # Third hidden layer
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer
        layers.Dense(1, activation='linear')      # Linear for regression
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',                    # Mean Squared Error
        metrics=['mae']                # Mean Absolute Error
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print("\nTraining Neural Network (100 epochs)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
        ]
    )
    print(" Model saved: models/neural_network_model.h5")
    
    # Make predictions
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)
    
    # Evaluate
    train_results = evaluate_regression(y_train, y_pred_train.flatten(), "Neural Network (Train)")
    test_results = evaluate_regression(y_test, y_pred_test.flatten(), "Neural Network (Test)")
    
    # Plot
    plot_predictions(y_test, y_pred_test.flatten(), "Neural Network")
    plot_residuals(y_test, y_pred_test.flatten(), "Neural Network")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('Neural Network: Training History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MAE', fontsize=12)
    plt.title('Neural Network: MAE History', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/training_history_neural_network.png', dpi=300)
    print(" Model saved: outputs/training_history_neural_network.png")
    plt.close()
    
    # Save model
    model.save('models/neural_network_model.h5')
    print(" Model saved: models/neural_network_model.h5")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_results': train_results,
        'test_results': test_results,
        'history': history,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


if __name__ == "__main__":
    from src.data_loader import load_data
    
    file_path = r"C:\Users\utkar\Adenosine_Receptor_Lingand\data\raw\AR_all_unique_parents_with_smiles.csv"
    df = load_data(file_path)
    
    if df is not None:
        results = train_neural_network(df)