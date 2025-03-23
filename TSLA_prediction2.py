import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load historical data
data = pd.read_csv('TSLA_Combined_processed.csv')  # Replace with your file path
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Feature engineering
data['PrevClose'] = data['Close'].shift(1)  # Previous day's closing price
data['PrevOpen'] = data['Open'].shift(1)    # Previous day's opening price
data['PrevVolume'] = data['Volume'].shift(1)  # Previous day's trading volume

# Drop rows with missing values EXCEPT for March 21, 2025
march_21_mask = data['Date'] == '2025-03-21'
data_to_drop = data[~march_21_mask].dropna()  # Drop rows with missing values except March 21, 2025
data = pd.concat([data_to_drop, data[march_21_mask]])  # Re-add March 21, 2025

# Define features (X) and targets (y_open, y_close, y_volume)
X = data[['Open', 'Close', 'PrevClose', 'PrevOpen', 'Volume', 'PrevVolume']]
y_open = data['NextOpen']  # Target for NextOpen prediction
y_close = data['NextClose']    # Target for NextClose prediction
y_volume = data['NextVolume']  # Target for NextVolume prediction

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rolling-window validation with a fixed time horizon (e.g., 2 years)
time_horizon = 2 * 252  # 2 years of trading days (assuming 252 trading days per year)

# Train the model on the most recent 2 years of data up to March 20, 2025
train_data = data[data['Date'] < '2025-03-21']
train_start_index = max(0, len(train_data) - time_horizon)  # Ensure we have at least 2 years of data
X_train = X_scaled[train_start_index:len(train_data)]
y_train_open = y_open[train_start_index:len(train_data)]
y_train_close = y_close[train_start_index:len(train_data)]
y_train_volume = y_volume[train_start_index:len(train_data)]

# Train models for NextOpen, Close, and Volume
model_open = RandomForestRegressor(n_estimators=100, random_state=42)
model_open.fit(X_train, y_train_open)

model_close = RandomForestRegressor(n_estimators=100, random_state=42)
model_close.fit(X_train, y_train_close)

model_volume = RandomForestRegressor(n_estimators=100, random_state=42)
model_volume.fit(X_train, y_train_volume)

# Predict NextOpen for March 21, 2025
march_21_data = data[data['Date'] == '2025-03-21']
if len(march_21_data) == 0:
    raise ValueError("March 21, 2025, not found in the dataset.")

current_open = march_21_data['Open'].values[0]
current_close = march_21_data['Close'].values[0]
current_volume = march_21_data['Volume'].values[0]
prev_close = march_21_data['PrevClose'].values[0]
prev_open = march_21_data['PrevOpen'].values[0]
prev_volume = march_21_data['PrevVolume'].values[0]

# Prepare features for prediction
features = np.array([[current_open, current_close, prev_close, prev_open, current_volume, prev_volume]])
features_df = pd.DataFrame(features, columns=X.columns)  # Convert to DataFrame with feature names
features_scaled = scaler.transform(features_df)  # Transform using the scaler

# Predict NextOpen, NextClose, and NextVolume for March 21, 2025
predicted_next_open_march_21 = model_open.predict(features_scaled)[0]
predicted_next_close_march_21 = model_close.predict(features_scaled)[0]
predicted_next_volume_march_21 = model_volume.predict(features_scaled)[0]
print(f"Predicted NextOpen for March 21, 2025: {predicted_next_open_march_21:.2f}")
print(f"Predicted NextClose for March 21, 2025: {predicted_next_close_march_21:.2f}")
print(f"Predicted NextVolume for March 21, 2025: {predicted_next_volume_march_21:.2f}")

# Use the predicted NextOpen for March 21, 2025, as the opening price for March 24, 2025
current_date = pd.to_datetime('2025-03-24')
current_open = predicted_next_open_march_21
current_close = predicted_next_close_march_21  # Use predicted NextClose for March 21, 2025
current_volume = predicted_next_volume_march_21  # Use predicted NextVolume for March 21, 2025

# Initialize variables
decisions = []
predicted_prices = []

# Iterate through the week (March 24–28, 2025)
for _ in range(5):  # 5 trading days
    # Prepare features for prediction
    features = np.array([[current_open, current_close, prev_close, prev_open, current_volume, prev_volume]])
    features_df = pd.DataFrame(features, columns=X.columns)  # Convert to DataFrame with feature names
    features_scaled = scaler.transform(features_df)  # Transform using the scaler

    # Predict NextOpen, Close, and Volume
    predicted_next_open = model_open.predict(features_scaled)[0]
    predicted_next_close = model_close.predict(features_scaled)[0]
    predicted_next_volume = model_volume.predict(features_scaled)[0]

    # Make decision based on predicted NextOpen price
    price_change = (predicted_next_open - current_open) / current_open * 100

    transaction_fee = 2.0  # 1% for buying + 1% for selling
    if price_change > transaction_fee:
        decision = 'Buy'
    elif price_change < -transaction_fee:
        decision = 'Sell'
    else:
        decision = 'Hold'
    decisions.append(decision)

    # Print results for the day
    print(f"Date: {current_date.date()}")
    print(f"Current Open: {current_open:.2f}")
    print(f"Predicted NextOpen: {predicted_next_open:.2f}")
    print(f"Predicted NextClose: {predicted_next_close:.2f}")
    print(f"Predicted NextVolume: {predicted_next_volume:.2f}")
    print(f"Decision: {decision}")
    print("------")

    # Ask for actual Close and Volume for the current day
    actual_open = float(input("Enter the actual Open price for the day: "))
    actual_close = float(input("Enter the actual Close price for the day: "))
    actual_volume = float(input("Enter the actual Volume for the day: "))

    # Update current_date, current_open, current_close, and current_volume for the next iteration
    prev_close = current_close  # Update prev_close to current_close
    prev_open = current_open    # Update prev_open to current_open
    prev_volume = current_volume  # Update prev_volume to current_volume

    current_date += pd.Timedelta(days=1)
    current_open = actual_open  # Use actual Open for the next day
    current_close = actual_close  # Use actual Close for the next day
    current_volume = actual_volume  # Use actual Volume for the next day

# Print predicted actions for the week
print("\nPredicted Actions for March 24–28, 2025:")
for i, decision in enumerate(decisions):
    print(f"March {24 + i}, 2025: {decision}")