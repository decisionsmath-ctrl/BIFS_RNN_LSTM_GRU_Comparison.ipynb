

# Real financial/market data API install karna
!pip install yfinance -q

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Real data download karna (Example: Apple Inc. ya koi bhi market index ka 5 saal ka data)
# Aap 'AAPL' ki jagah kisi bhi company ka ticker likh sakti hain
real_data = yf.download('AAPL', start='2021-01-01', end='2026-01-01')

# Hamein sirf Closing Price par prediction karni hai
df_real = pd.DataFrame(real_data['Close'])
df_real.columns = ['Close']

print("Real Dataset successfully download ho gaya hai!")
print(f"Total records (Days): {len(df_real)}")
print(df_real.head())

# 1. Data ko 0 aur 1 ke darmiyan scale (chota) karna
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_real)

# 2. Chronological Split (Waqt ke hisab se 70% Train aur 30% Test)
train_size = int(len(scaled_data) * 0.70)
train_data = scaled_data[0:train_size]
test_data = scaled_data[train_size:len(scaled_data)]

# 3. Look-back Window (Pichle 15 dino ka data dekh kar 16th day predict karna)
look_back = 15

# Train Sequences banana
X_train, y_train = [], []
for i in range(look_back, len(train_data)):
    X_train.append(train_data[i-look_back:i, 0])
    y_train.append(train_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Test Sequences banana
X_test, y_test = [], []
for i in range(look_back, len(test_data)):
    X_test.append(test_data[i-look_back:i, 0])
    y_test.append(test_data[i, 0])
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("Data successfully split aur transform ho gaya hai!")
print(f"Train shapes: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"Test shapes: X_test={X_test.shape}, y_test={y_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 1. LSTM Architecture design karna
model_lstm = Sequential([
    LSTM(50, return_sequences=False, input_shape=(look_back, 1)),
    Dense(25),
    Dense(1)
])

model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# 2. Model ko sirf Train data par sikhana (Epochs = 10)
print("LSTM model Train Data par training shuru kar raha hai...")
model_lstm.fit(X_train, y_train, batch_size=16, epochs=10, verbose=1)

# 3. Trained model ko unseen Test Data par test karna
print("\nModel unseen Test Data par predictions nikaal raha hai...")
predictions = model_lstm.predict(X_test)

# 4. Scale hue numbers ko wapas real stock market prices mein badalna
predictions_real = scaler.inverse_transform(predictions)
actual_y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# 5. Testing ka Final Graph visual dekhna
plt.figure(figsize=(12, 6))
plt.plot(actual_y_test_real, color='black', label='Actual Real Prices (Ground Truth)')
plt.plot(predictions_real, color='red', linestyle='--', label='LSTM Predicted Prices')
plt.title("MPhil Project: Unseen Testing Data Validation Results")
plt.xlabel("Timeline (Testing Days)")
plt.ylabel("Stock Price / Units Value")
plt.legend()
plt.grid(True)
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, GRU, Dense, LSTM

# 1. Simple RNN Model Build aur Train karna
print("Training Simple RNN...")
model_rnn = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(look_back, 1)),
    Dense(1)
])
model_rnn.compile(optimizer='adam', loss='mean_squared_error')
model_rnn.fit(X_train, y_train, batch_size=16, epochs=10, verbose=0)

# 2. GRU Model Build aur Train karna
print("Training GRU Engine...")
model_gru = Sequential([
    GRU(50, return_sequences=False, input_shape=(look_back, 1)),
    Dense(25),
    Dense(1)
])
model_gru.compile(optimizer='adam', loss='mean_squared_error')
model_gru.fit(X_train, y_train, batch_size=16, epochs=10, verbose=0)

print("Simple RNN aur GRU dono train ho chuke hain!")

# 3. Teeno models se unseen Test Data par predictions nikalna
print("\nUnseen Test Data par teeno models apply ho rahe hain...")
pred_rnn = model_rnn.predict(X_test)
pred_gru = model_gru.predict(X_test)

# 4. Scale hue numbers ko wapas real prices mein badalna
predictions_rnn_real = scaler.inverse_transform(pred_rnn)
predictions_gru_real = scaler.inverse_transform(pred_gru)

# 5. Final Master Comparison Graph Visual
plt.figure(figsize=(14, 7))
plt.plot(actual_y_test_real, color='black', linewidth=2.5, label='Actual Real Prices (Ground Truth)')
plt.plot(predictions_real, color='red', linestyle='--', label='LSTM Predicted (Gate Optimized)')
plt.plot(predictions_gru_real, color='green', linestyle='-.', label='GRU Predicted (Lightweight Engine)')
plt.plot(predictions_rnn_real, color='orange', linestyle=':', label='Simple RNN Predicted (Vanishing Gradient)')

plt.title("MPhil Project Master Evaluation: RNN vs LSTM vs GRU", fontsize=14)
plt.xlabel("Timeline (Testing Days)", fontsize=12)
plt.ylabel("Stock Price / Units Value", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True)
plt.show()

"""### Model Performance Comparison using RMSE

To quantitatively compare the performance, we can calculate the Root Mean Squared Error (RMSE) for both the LSTM and GRU models. RMSE measures the average magnitude of the errors and is a common metric for evaluating regression models.
"""

from sklearn.metrics import mean_squared_error

# Calculate RMSE for LSTM
rmse_lstm = np.sqrt(mean_squared_error(actual_y_test_real, predictions_real))
print(f"LSTM Model RMSE: {rmse_lstm:.4f}")

# Calculate RMSE for GRU
rmse_gru = np.sqrt(mean_squared_error(actual_y_test_real, predictions_gru_real))
print(f"GRU Model RMSE: {rmse_gru:.4f}")

# Calculate RMSE for Simple RNN
rmse_rnn = np.sqrt(mean_squared_error(actual_y_test_real, predictions_rnn_real))
print(f"Simple RNN Model RMSE: {rmse_rnn:.4f}")

# You can also compare them directly
if rmse_lstm < rmse_gru and rmse_lstm < rmse_rnn:
    print("\nLSTM model performed best!")
elif rmse_gru < rmse_lstm and rmse_gru < rmse_rnn:
    print("\nGRU model performed best!")
elif rmse_rnn < rmse_lstm and rmse_rnn < rmse_gru:
    print("\nSimple RNN model performed best!")
else:
    print("\nModels performed similarly or there's a tie.")

"""### Saving the Trained Models

It's a good practice to save trained models so they can be reloaded later without needing to retrain them. We will save both the LSTM and GRU models using the `.save()` method in the `.keras` format.
"""

# Save the LSTM model
model_lstm.save('lstm_model.keras')
print("LSTM model saved as 'lstm_model.keras'")

# Save the GRU model
model_gru.save('gru_model.keras')
print("GRU model saved as 'gru_model.keras'")

# Save the Simple RNN model
model_rnn.save('rnn_model.keras')
print("Simple RNN model saved as 'rnn_model.keras'")