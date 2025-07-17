import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Define the start and end dates
start = '2014-01-01'
end = '2024-01-01'

# Fetch data using yfinance
df = yf.download('AAPL', start=start, end=end)

# Display the first few rows
print(df.head())

# Reset the index
df = df.reset_index()

# Plot the Closing Price
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='AAPL Closing Price')
plt.title('AAPL Stock Price (2012-2022)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Calculate moving averages
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

# Plot the closing price along with moving averages
plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='AAPL Closing Price')
plt.plot(ma100, label='100-Day Moving Average', color='r')
plt.plot(ma200, label='200-Day Moving Average', color='g')
plt.title('AAPL Stock Price with Moving Averages (2012-2022)')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# Splitting the data into Training (70%) and Testing (30%) sets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])  # 70% for training
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])  # 30% for testing

print(f"Training data shape: {data_training.shape}")
print(f"Testing data shape: {data_testing.shape}")

# Scaling the data (normalizing between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Preparing the training data (creating sequences)
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

# Convert lists to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Optional: Print the shape of training data for verification
print(f"Training data (X) shape: {x_train.shape}")
print(f"Training labels (Y) shape: {y_train.shape}")

# Reshape the data for LSTM input (samples, timesteps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Initialize the LSTM model
model = Sequential()

# Add 4 LSTM layers with specified units and dropout rates
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))  # First LSTM layer (50 units)
model.add(Dropout(0.2))  # Dropout rate = 0.2

model.add(LSTM(units=60, return_sequences=True))  # Second LSTM layer (60 units)
model.add(Dropout(0.3))  # Dropout rate = 0.3

model.add(LSTM(units=80, return_sequences=True))  # Third LSTM layer (80 units)
model.add(Dropout(0.4))  # Dropout rate = 0.4

model.add(LSTM(units=120, return_sequences=False))  # Fourth LSTM layer (120 units)
model.add(Dropout(0.5))  # Dropout rate = 0.5

# Add the output Dense layer
model.add(Dense(units=1))  # Single neuron for predicting the next price

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Add early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with validation split
history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop])

# Plot training and validation loss
plt.figure(figsize=(12,6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Testing part
# Scale the testing data
data_testing_array = scaler.transform(data_testing)

x_test = []
y_test = []

for i in range(100, data_testing_array.shape[0]):
    x_test.append(data_testing_array[i-100:i])
    y_test.append(data_testing_array[i, 0])

# Convert lists to numpy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the test data for LSTM input
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)

# Inverse scale the predictions and actual values
predictions = scaler.inverse_transform(predictions)  # Convert predictions back to original scale
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))  # Convert test labels back to original scale

# Calculate MAE, RMSE, and MAPE
mae = mean_absolute_error(y_test, predictions)
rmse = math.sqrt(mean_squared_error(y_test, predictions))
mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
accuracy = 100 - mape

print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print(f"📊 Model Accuracy: {accuracy:.2f}%")

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price')  # Actual stock prices
plt.plot(predictions, label='Predicted Price')  # Predicted stock prices
plt.title('AAPL Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price ($)')
plt.legend()
plt.grid(True)
plt.show()

# Residual analysis
residuals = y_test - predictions
plt.figure(figsize=(12,6))
plt.hist(residuals, bins=50)
plt.title('Residuals Distribution')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

model.save('final_best_stock_model.h5')
print("✅ Model saved successfully!")

