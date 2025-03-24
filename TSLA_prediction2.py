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
data['NextOpen'] = data['Open'].shift(-1)   # Next day's opening price
data['NextClose'] = data['Close'].shift(-1) # Next day's closing price

# Drop rows with missing values EXCEPT for March 21, 2025
march_21_mask = data['Date'] == '2025-03-21'
data_to_drop = data[~march_21_mask].dropna()  # Drop rows with missing values except March 21, 2025
data = pd.concat([data_to_drop, data[march_21_mask]])  # Re-add March 21, 2025

# Define features (X) and targets (y)
X = data[['Open', 'Close', 'PrevClose', 'PrevOpen']]
y_open = data['NextOpen']
y_close = data['NextClose']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model on the most recent 3 years of data up to March 21, 2025
train_data = data[data['Date'] < '2025-03-21']
time_horizon = 3 * 252  # 3 years of trading days (assuming 252 trading days per year)
train_start_index = max(0, len(train_data) - time_horizon)  # Ensure we have at least 3 years of data
X_train = X_scaled[train_start_index:len(train_data)]
y_train_open = y_open[train_start_index:len(train_data)]
y_train_close = y_close[train_start_index:len(train_data)]

# Train models for NextOpen and NextClose
model_open = RandomForestRegressor(n_estimators=100, random_state=42)
model_open.fit(X_train, y_train_open)

model_close = RandomForestRegressor(n_estimators=100, random_state=42)
model_close.fit(X_train, y_train_close)

# Predict NextOpen and NextClose for March 21, 2025
march_21_data = data[data['Date'] == '2025-03-21']
if len(march_21_data) == 0:
    raise ValueError("March 21, 2025, not found in the dataset.")

# Extract values for March 21, 2025
current_open = march_21_data['Open'].values[0]
current_close = march_21_data['Close'].values[0]
prev_close = march_21_data['PrevClose'].values[0]
prev_open = march_21_data['PrevOpen'].values[0]

# Initialize account balance and position
initial_capital = 10000  # Initial capital in dollars
account_balance = initial_capital
shares_held = 0  # Number of shares held
transaction_fee_rate = 0.01  # 1% transaction fee for buy or sell

# Use the predicted Open for March 24, 2025, as the opening price for March 24, 2025
current_date = pd.to_datetime('2025-03-24')

# Initialize variables
decisions = []

# Iterate through the week (March 24–28, 2025)
for _ in range(5):  # 5 trading days
    # Prepare features for prediction
    features = np.array([[current_open, current_close, prev_close, prev_open]])
    features_df = pd.DataFrame(features, columns=X.columns)  # Convert to DataFrame with feature names
    features_scaled = scaler.transform(features_df)  # Transform using the scaler

    # Predict Current Open and Close
    predicted_open = model_open.predict(features_scaled)[0]  # Predicted opening price for the current day
    predicted_close = model_close.predict(features_scaled)[0]  # Predicted closing price for the current day

    temp_open = predicted_open
    temp_close = predicted_close

    # Prepare features for next open prediction
    features_next = np.array([[temp_open, temp_close, current_open, current_close]])
    features_df_next = pd.DataFrame(features_next, columns=X.columns)  # Convert to DataFrame with feature names
    features_scaled_next = scaler.transform(features_df_next)  # Transform using the scaler

    # Predict Next Open
    predicted_next_open = model_open.predict(features_scaled_next)[0]  # Predicted the opening price for the next day

    # Make decision based on predicted NextOpen price
    price_change = (predicted_next_open - predicted_open) / predicted_open * 100  # Corrected calculation

    if price_change > 2.0:  # Buy if price increase > transaction fee (1% buy + 1% sell)
        decision = 'Buy'
    elif price_change < -2.0:  # Sell if price decrease > transaction fee
        decision = 'Sell'
    else:
        decision = 'Hold'
    decisions.append(decision)

    # Execute trading decision
    if decision == 'Buy' and account_balance >= predicted_open:
        # Calculate number of shares to buy
        shares_to_buy = (account_balance * (1 - transaction_fee_rate)) // predicted_open
        shares_held += shares_to_buy
        account_balance -= shares_to_buy * predicted_open * (1 + transaction_fee_rate)
    elif decision == 'Sell' and shares_held > 0:
        # Sell all shares held
        account_balance += shares_held * predicted_open * (1 - transaction_fee_rate)
        shares_held = 0

    # Print results for the day
    print(f"Date: {current_date.date()}")
    print(f"Predicted Open (Current Day): {predicted_open:.2f}")
    print(f"Predicted Close (Current Day): {predicted_close:.2f}")
    print(f"Predicted NextOpen (Next Day): {predicted_next_open:.2f}")
    print(f"Decision: {decision}")
    print(f"Shares Held: {shares_held}")
    print(f"Account Balance: ${account_balance:.2f}")
    print("------")

    # Ask user for actual opening and closing prices
    actual_open = float(input(f"Enter the actual Open price for {current_date.date()}: "))
    actual_close = float(input(f"Enter the actual Close price for {current_date.date()}: "))

    # Update current_date, current_open, current_close, and previous prices for the next iteration
    prev_close = current_close  # Update prev_close to current_close
    prev_open = current_open    # Update prev_open to current_open

    current_date += pd.Timedelta(days=1)
    current_open = actual_open  # Use actual Open for the next day
    current_close = actual_close  # Use actual Close for the next day

# Final account value
final_value = account_balance + (shares_held * current_close)
print(f"\nFinal Account Value: ${final_value:.2f}")
print(f"Profit/Loss: ${final_value - initial_capital:.2f}")

# Print predicted actions for the week
print("\nPredicted Actions for March 24–28, 2025:")
for i, decision in enumerate(decisions):
    print(f"March {24 + i}, 2025: {decision}")