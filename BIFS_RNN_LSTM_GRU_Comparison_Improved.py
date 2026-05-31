"""
BIFS RNN/LSTM/GRU Comparison - Improved Version
MPhil Project: Stock Price Prediction using Recurrent Neural Networks

Key Improvements:
- Fixed data leakage (scaler fit on train data only)
- Added validation set and early stopping
- Standardized model architectures for fair comparison
- Comprehensive metrics (RMSE, MAE, R², MAPE)
- Refactored code with functions to avoid duplication
- Proper error handling and logging
- Extended training with dropout regularization
- Visualization of training history
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

TICKER = 'AAPL'
START_DATE = '2021-01-01'
END_DATE = '2026-01-01'
LOOK_BACK = 15
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
BATCH_SIZE = 16
EPOCHS = 100
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
RANDOM_SEED = 42

# Set random seeds for reproducibility
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)

# ============================================================================
# IMPORTS
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

print("All libraries imported successfully!")

# ============================================================================
# DATA DOWNLOAD & PREPARATION
# ============================================================================

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock price data from Yahoo Finance
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL')
        start_date: Start date (YYYY-MM-DD format)
        end_date: End date (YYYY-MM-DD format)
    
    Returns:
        DataFrame with stock data
    """
    print(f"\nDownloading {ticker} data from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✓ Successfully downloaded {len(data)} records")
        return data
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        raise

def create_sequences(data, look_back):
    """
    Create time series sequences for LSTM/RNN/GRU models
    
    Args:
        data: Scaled data array
        look_back: Number of previous time steps to use for prediction
    
    Returns:
        X: Input sequences (samples, time steps, features)
        y: Target values
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y

def prepare_and_split_data(df, look_back, train_split, val_split):
    """
    Prepare data with proper scaling and splitting to avoid data leakage
    
    CRITICAL: Scaler is fit ONLY on training data!
    
    Args:
        df: DataFrame with stock data
        look_back: Lookback window size
        train_split: Training data fraction (e.g., 0.70 for 70%)
        val_split: Validation data fraction (e.g., 0.15 for 15%)
    
    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    print("\n" + "="*60)
    print("DATA PREPARATION & SPLITTING")
    print("="*60)
    
    # Extract closing price
    close_price = df[['Close']].values
    
    # Initialize scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Split indices
    total_len = len(close_price)
    train_size = int(total_len * train_split)
    val_size = int(total_len * val_split)
    
    print(f"Total records: {total_len}")
    print(f"Train size: {train_size} ({train_split*100:.0f}%)")
    print(f"Val size: {val_size} ({val_split*100:.0f}%)")
    print(f"Test size: {total_len - train_size - val_size} ({(1-train_split-val_split)*100:.0f}%)")
    
    # CRITICAL FIX: Fit scaler ONLY on training data
    train_data = close_price[:train_size]
    scaler.fit(train_data)
    
    # Scale all data using the scaler fit on training data
    scaled_data = scaler.transform(close_price)
    
    # Split into train, validation, and test
    train_scaled = scaled_data[:train_size]
    val_scaled = scaled_data[train_size:train_size+val_size]
    test_scaled = scaled_data[train_size+val_size:]
    
    # Create sequences
    X_train, y_train = create_sequences(train_scaled, look_back)
    X_val, y_val = create_sequences(val_scaled, look_back)
    X_test, y_test = create_sequences(test_scaled, look_back)
    
    print(f"\nSequence shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("="*60)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# ============================================================================
# MODEL ARCHITECTURES (STANDARDIZED)
# ============================================================================

def build_lstm_model(look_back, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """
    Build standardized LSTM model with dropout regularization
    """
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(dropout),
        LSTM(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_gru_model(look_back, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """
    Build standardized GRU model with dropout regularization
    """
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=(look_back, 1)),
        Dropout(dropout),
        GRU(units, return_sequences=False),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_rnn_model(look_back, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """
    Build standardized Simple RNN model with dropout regularization
    """
    model = Sequential([
        SimpleRNN(units, activation='tanh', return_sequences=True, input_shape=(look_back, 1)),
        Dropout(dropout),
        SimpleRNN(units, activation='tanh'),
        Dropout(dropout),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(model, name, X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train model with early stopping based on validation loss
    
    Args:
        model: Keras model to train
        name: Model name (for logging)
        X_train, y_train: Training data
        X_val, y_val: Validation data
        epochs: Maximum number of epochs
        batch_size: Batch size
    
    Returns:
        Trained model and history object
    """
    print(f"\n{'='*60}")
    print(f"TRAINING {name}")
    print(f"{'='*60}")
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )
    
    print(f"✓ {name} training completed!")
    return model, history

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_predictions(y_true, y_pred):
    """
    Calculate comprehensive evaluation metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
    
    Returns:
        Dictionary with metrics
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def print_model_comparison(metrics_dict):
    """
    Print formatted comparison of model metrics
    """
    print(f"\n{'='*60}")
    print("MODEL PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    metrics_df = pd.DataFrame(metrics_dict).T
    print(metrics_df.to_string())
    print("="*60)
    
    # Find best models
    print("\nBest Models by Metric:")
    print(f"  Best RMSE: {metrics_df['RMSE'].idxmin()} ({metrics_df['RMSE'].min():.4f})")
    print(f"  Best MAE:  {metrics_df['MAE'].idxmin()} ({metrics_df['MAE'].min():.4f})")
    print(f"  Best R²:   {metrics_df['R²'].idxmax()} ({metrics_df['R²'].max():.4f})")
    print(f"  Best MAPE: {metrics_df['MAPE'].idxmin()} ({metrics_df['MAPE'].min():.4f}%)")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(histories_dict):
    """
    Plot training and validation loss for all models
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, history) in enumerate(histories_dict.items()):
        axes[idx].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[idx].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[idx].set_title(f'{name} - Training History', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss (MSE)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved as 'training_history.png'")
    plt.show()

def plot_predictions(actual, predictions_dict, scaler):
    """
    Plot actual vs predicted prices for all models
    """
    # Inverse transform to get real prices
    actual_real = scaler.inverse_transform(actual.reshape(-1, 1))
    
    predictions_real = {}
    for name, pred in predictions_dict.items():
        predictions_real[name] = scaler.inverse_transform(pred.reshape(-1, 1))
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(actual_real, color='black', linewidth=2.5, label='Actual Prices (Ground Truth)', zorder=5)
    
    colors = {'LSTM': 'red', 'GRU': 'green', 'Simple RNN': 'orange'}
    linestyles = {'LSTM': '--', 'GRU': '-.', 'Simple RNN': ':'}
    
    for name, pred in predictions_real.items():
        plt.plot(pred, color=colors[name], linestyle=linestyles[name], 
                linewidth=2, label=f'{name} Predictions', alpha=0.8)
    
    plt.title("Stock Price Predictions: RNN vs LSTM vs GRU Comparison", fontsize=14, fontweight='bold')
    plt.xlabel("Timeline (Testing Days)", fontsize=12)
    plt.ylabel("Stock Price (USD)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Predictions comparison plot saved as 'predictions_comparison.png'")
    plt.show()

def plot_residuals(actual, predictions_dict, scaler):
    """
    Plot prediction residuals (errors)
    """
    actual_real = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(predictions_dict.keys())
    
    for idx, name in enumerate(names):
        pred_real = scaler.inverse_transform(predictions_dict[name].reshape(-1, 1)).flatten()
        residuals = actual_real - pred_real
        
        axes[idx].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=['red', 'green', 'orange'][idx])
        axes[idx].set_title(f'{name} - Residuals Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Residual Value')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Residuals analysis plot saved as 'residuals_analysis.png'")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Download and prepare data
    print("\n" + "="*60)
    print("BIFS RNN/LSTM/GRU COMPARISON - IMPROVED VERSION")
    print("="*60)
    
    raw_data = download_stock_data(TICKER, START_DATE, END_DATE)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_and_split_data(
        raw_data, LOOK_BACK, TRAIN_SPLIT, VAL_SPLIT
    )
    
    # Step 2: Build and train models
    print("\nBuilding models...")
    model_lstm, hist_lstm = train_model(
        build_lstm_model(LOOK_BACK), "LSTM",
        X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE
    )
    
    model_gru, hist_gru = train_model(
        build_gru_model(LOOK_BACK), "GRU",
        X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE
    )
    
    model_rnn, hist_rnn = train_model(
        build_rnn_model(LOOK_BACK), "Simple RNN",
        X_train, y_train, X_val, y_val, EPOCHS, BATCH_SIZE
    )
    
    # Step 3: Make predictions
    print(f"\n{'='*60}")
    print("GENERATING PREDICTIONS ON TEST SET")
    print(f"{'='*60}")
    
    pred_lstm = model_lstm.predict(X_test, verbose=0)
    pred_gru = model_gru.predict(X_test, verbose=0)
    pred_rnn = model_rnn.predict(X_test, verbose=0)
    
    # Step 4: Evaluate models
    metrics = {
        'LSTM': evaluate_predictions(y_test, pred_lstm.flatten()),
        'GRU': evaluate_predictions(y_test, pred_gru.flatten()),
        'Simple RNN': evaluate_predictions(y_test, pred_rnn.flatten())
    }
    
    print_model_comparison(metrics)
    
    # Step 5: Visualizations
    plot_training_history({
        'LSTM': hist_lstm,
        'GRU': hist_gru,
        'Simple RNN': hist_rnn
    })
    
    plot_predictions(y_test, {
        'LSTM': pred_lstm,
        'GRU': pred_gru,
        'Simple RNN': pred_rnn
    }, scaler)
    
    plot_residuals(y_test, {
        'LSTM': pred_lstm,
        'GRU': pred_gru,
        'Simple RNN': pred_rnn
    }, scaler)
    
    # Step 6: Save models
    print(f"\n{'='*60}")
    print("SAVING TRAINED MODELS")
    print(f"{'='*60}")
    
    model_lstm.save('lstm_model.keras')
    print("✓ LSTM model saved as 'lstm_model.keras'")
    
    model_gru.save('gru_model.keras')
    print("✓ GRU model saved as 'gru_model.keras'")
    
    model_rnn.save('rnn_model.keras')
    print("✓ Simple RNN model saved as 'rnn_model.keras'")
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_df.to_csv('model_metrics.csv')
    print("✓ Metrics saved as 'model_metrics.csv'")
    
    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - lstm_model.keras")
    print("  - gru_model.keras")
    print("  - rnn_model.keras")
    print("  - model_metrics.csv")
    print("  - training_history.png")
    print("  - predictions_comparison.png")
    print("  - residuals_analysis.png")
