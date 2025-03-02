import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("menstrual_data/FedCycleData071012.csv")  # Update with your file path

# Features for cycle prediction
features = ["CycleNumber", "LengthofCycle", "MeanCycleLength", 
            "EstimatedDayofOvulation", "LengthofLutealPhase"]

# Ensure the dataset contains only relevant columns
df = df[features]

# 🔴 FIX: Convert non-numeric values to NaN and replace missing values
df[features] = df[features].apply(pd.to_numeric, errors='coerce')
df.fillna(df.median(), inplace=True)  # Replace NaNs with median values

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predicting 'CycleNumber'
    return np.array(X), np.array(y)

SEQ_LENGTH = 10  # Number of past cycles to consider
X, y = create_sequences(df_scaled, SEQ_LENGTH)

# Split into training and testing sets (80% train, 20% test)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM input
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], len(features)))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], len(features)))

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LENGTH, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation="relu"),
    Dense(1)  # Output layer for CycleNumber prediction
])

model.compile(optimizer="adam", loss="mse")

# Train model with early stopping
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions
y_test_actual = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))), axis=1))[:, 0]
y_pred_actual = scaler.inverse_transform(np.concatenate((y_pred, np.zeros((len(y_pred), len(features) - 1))), axis=1))[:, 0]

# Evaluate performance
mae = mean_absolute_error(y_test_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_actual, label="Actual Cycle", linestyle="-", color="blue")
plt.plot(y_pred_actual, label="Predicted Cycle", linestyle="--", color="orange")
plt.legend()
plt.show()
model.save("lstm_model.h5")