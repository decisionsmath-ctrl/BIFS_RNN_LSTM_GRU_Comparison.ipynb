"""
BIFS RNN/LSTM/GRU Comparison - Google Colab Ready Version
MPhil Project: Stock Price Prediction using Recurrent Neural Networks

This version is optimized for Google Colab and runs completely in the cloud!
No local installation needed.

To use:
1. Go to https://colab.research.google.com/
2. Click "File" → "Open notebook"
3. Click "GitHub" tab
4. Paste this URL: https://github.com/decisionsmath-ctrl/BIFS_RNN_LSTM_GRU_Comparison.ipynb
5. Select this file and open
6. Click "Run all" button (▶️ Play icon in toolbar)
"""

# ============================================================================
# SETUP FOR GOOGLE COLAB (Run this first!)
# ============================================================================

print("="*70)
print("BIFS RNN/LSTM/GRU COMPARISON - GOOGLE COLAB VERSION")
print("="*70)
print("\n⏳ Installing required libraries...")

import sys
import subprocess

# Install/upgrade required packages
packages = [
    'tensorflow>=2.13.0',
    'keras>=2.13.0',
    'scikit-learn>=1.3.0',
    'yfinance>=0.2.0',
    'pandas>=2.0.0',
    'matplotlib>=3.7.0',
    'numpy>=1.24.0'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("✓ All libraries installed successfully!\n")

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

print("✓ All imports completed!\n")

# ============================================================================
# DATA DOWNLOAD & PREPARATION
# ============================================================================

def download_stock_data(ticker, start_date, end_date):
    """
    Download stock price data from Yahoo Finance
    """
    print(f"📥 Downloading {ticker} data from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"✓ Successfully downloaded {len(data)} records\n")
        return data
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        raise

def create_sequences(data, look_back):
    """
    Create time series sequences for LSTM/RNN/GRU models
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
    """
    print("="*70)
    print("📊 DATA PREPARATION & SPLITTING")
    print("="*70)
    
    close_price = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    total_len = len(close_price)
    train_size = int(total_len * train_split)
    val_size = int(total_len * val_split)
    
    print(f"📈 Total records: {total_len}")
    print(f"🔵 Train size: {train_size} ({train_split*100:.0f}%)")
    print(f"🟡 Val size: {val_size} ({val_split*100:.0f}%)")
    print(f"🔴 Test size: {total_len - train_size - val_size} ({(1-train_split-val_split)*100:.0f}%)")
    
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
    
    print(f"\n📐 Sequence shapes:")
    print(f"   X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"   X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("="*70 + "\n")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

# ============================================================================
# MODEL ARCHITECTURES (STANDARDIZED)
# ============================================================================

def build_lstm_model(look_back, units=LSTM_UNITS, dropout=DROPOUT_RATE):
    """Build standardized LSTM model with dropout regularization"""
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
    """Build standardized GRU model with dropout regularization"""
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
    """Build standardized Simple RNN model with dropout regularization"""
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
    """Train model with early stopping based on validation loss"""
    print(f"\n{'='*70}")
    print(f"🤖 TRAINING {name.upper()}")
    print(f"{'='*70}")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
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
    """Calculate comprehensive evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'MAPE': mape
    }

def print_model_comparison(metrics_dict):
    """Print formatted comparison of model metrics"""
    print(f"\n{'='*70}")
    print("🏆 MODEL PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    metrics_df = pd.DataFrame(metrics_dict).T
    print(metrics_df.to_string())
    print("="*70)
    
    print("\n🥇 Best Models by Metric:")
    print(f"  RMSE:  {metrics_df['RMSE'].idxmin()} ({metrics_df['RMSE'].min():.4f}) ⭐")
    print(f"  MAE:   {metrics_df['MAE'].idxmin()} ({metrics_df['MAE'].min():.4f}) ⭐")
    print(f"  R²:    {metrics_df['R²'].idxmax()} ({metrics_df['R²'].max():.4f}) ⭐")
    print(f"  MAPE:  {metrics_df['MAPE'].idxmin()} ({metrics_df['MAPE'].min():.4f}%) ⭐\n")

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(histories_dict):
    """Plot training and validation loss for all models"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, history) in enumerate(histories_dict.items()):
        axes[idx].plot(history.history['loss'], label='Training Loss', linewidth=2, color='blue')
        axes[idx].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='red')
        axes[idx].set_title(f'{name} - Training History', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss (MSE)')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print("\n✓ Training history plot saved!")
    plt.show()

def plot_predictions(actual, predictions_dict, scaler):
    """Plot actual vs predicted prices for all models"""
    actual_real = scaler.inverse_transform(actual.reshape(-1, 1))
    
    predictions_real = {}
    for name, pred in predictions_dict.items():
        predictions_real[name] = scaler.inverse_transform(pred.reshape(-1, 1))
    
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
    print("✓ Predictions comparison plot saved!")
    plt.show()

def plot_residuals(actual, predictions_dict, scaler):
    """Plot prediction residuals (errors)"""
    actual_real = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    names = list(predictions_dict.keys())
    colors_hist = ['red', 'green', 'orange']
    
    for idx, name in enumerate(names):
        pred_real = scaler.inverse_transform(predictions_dict[name].reshape(-1, 1)).flatten()
        residuals = actual_real - pred_real
        
        axes[idx].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color=colors_hist[idx])
        axes[idx].set_title(f'{name} - Residuals Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Residual Value (USD)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    plt.savefig('residuals_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Residuals analysis plot saved!")
    plt.show()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Step 1: Download and prepare data
    print("\n" + "="*70)
    print("📊 BIFS RNN/LSTM/GRU COMPARISON - IMPROVED VERSION")
    print("="*70)
    
    raw_data = download_stock_data(TICKER, START_DATE, END_DATE)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_and_split_data(
        raw_data, LOOK_BACK, TRAIN_SPLIT, VAL_SPLIT
    )
    
    # Step 2: Build and train models
    print("\n🔨 Building and training models...\n")
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
    print(f"\n{'='*70}")
    print("🎯 GENERATING PREDICTIONS ON TEST SET")
    print(f"{'='*70}\n")
    
    pred_lstm = model_lstm.predict(X_test, verbose=0)
    pred_gru = model_gru.predict(X_test, verbose=0)
    pred_rnn = model_rnn.predict(X_test, verbose=0)
    
    print("✓ Predictions generated!")
    
    # Step 4: Evaluate models
    metrics = {
        'LSTM': evaluate_predictions(y_test, pred_lstm.flatten()),
        'GRU': evaluate_predictions(y_test, pred_gru.flatten()),
        'Simple RNN': evaluate_predictions(y_test, pred_rnn.flatten())
    }
    
    print_model_comparison(metrics)
    
    # Step 5: Visualizations
    print("\n📈 Creating visualizations...")
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
    print(f"\n{'='*70}")
    print("💾 SAVING TRAINED MODELS")
    print(f"{'='*70}\n")
    
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
    
    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE!")
    print("="*70)
    print("\n📁 Generated files:")
    print("  ✓ lstm_model.keras")
    print("  ✓ gru_model.keras")
    print("  ✓ rnn_model.keras")
    print("  ✓ model_metrics.csv")
    print("  ✓ training_history.png")
    print("  ✓ predictions_comparison.png")
    print("  ✓ residuals_analysis.png")
    print("\n🎓 Ready for your MPhil Project!\n")
