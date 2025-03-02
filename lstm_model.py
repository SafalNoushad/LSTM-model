import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# ✅ Load dataset
df = pd.read_csv("menstrual_data/FedCycleData071012.csv")  # Update path if needed
print(df.head())  # Show first few rows

# ✅ Select relevant features (adjust column names as needed)
df = df[['CycleNumber']]  # Modify if needed

# ✅ Normalize the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# ✅ Function to create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predicting 'CycleNumber'
    return np.array(X), np.array(y)

seq_length = 10  # Use past 10 cycles for prediction
X, y = create_sequences(scaled_data, seq_length)

# ✅ Split into training and testing sets (80% train, 20% test)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ✅ Reshape input to fit LSTM (samples, timesteps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# ✅ Build the LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)  # Output layer for predicting CycleNumber
])

# ✅ Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# ✅ Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# ✅ Evaluate model
loss = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")

# ✅ Predict future cycle numbers
predictions = model.predict(X_test)

# ✅ Reshape predictions for inverse scaling
predictions = predictions.reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# ✅ Plot the results
plt.figure(figsize=(10, 5))
plt.plot(range(len(y_test)), scaler.inverse_transform(y_test.reshape(-1, 1)), label="Actual Cycle")
plt.plot(range(len(predictions)), predictions, label="Predicted Cycle", linestyle='dashed')
plt.legend()
plt.show()
