import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load historical data
data = pd.read_csv('TSLA_Combined_processed.csv')  # Replace with your file path
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Filter data for the last x years
last_date = data['Date'].iloc[-1]
x_years_ago = last_date - pd.DateOffset(years=5)
data = data[data['Date'] >= x_years_ago]

# Feature engineering
data['PrevClose'] = data['Close'].shift(1)  # Previous day's closing price
data['PrevOpen'] = data['Open'].shift(1)    # Previous day's opening price
data['NextOpen'] = data['Open'].shift(-1)    # Next day's opening price
data['NextClose'] = data['Close'].shift(-1)  # Next day's closing price
data.dropna(inplace=True)  # Drop rows with missing values

# Define features (X) and targets (y)
X = data[['Open', 'Close', 'PrevClose', 'PrevOpen']]
y_open = data['NextOpen']
y_close = data['NextClose']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rolling-window validation with a fixed time horizon (e.g., 2 years)
time_horizon = 3 * 252  # 3 years of trading days (assuming 252 trading days per year)
mse_open_list = []  # Store MSE for NextOpen predictions
mse_close_list = []  # Store MSE for NextClose predictions
mse_current_open_list = []  # Store MSE for current day's open predictions
decisions = []  # Store decisions (Buy/Sell/Hold)
actual_actions = []  # Store actual actions based on NextOpen price

# Iterate through the dataset
for i in range(len(data) - 2):
    # Ensure the training set has at least `time_horizon` days of data
    if i + 1 < time_horizon:
        continue  # Skip predictions for the first `time_horizon` days

    # Training period: Most recent `time_horizon` days up to the current day
    train_start_index = i + 1 - time_horizon  # Start index for training set
    train_end_index = i + 1  # End index for training set
    train_start_date = data['Date'].iloc[train_start_index]
    train_end_date = data['Date'].iloc[train_end_index]

    X_train = X_scaled[train_start_index:train_end_index]
    y_train_open = y_open[train_start_index:train_end_index]
    y_train_close = y_close[train_start_index:train_end_index]

    # Testing period: Next day
    test_date = data['Date'].iloc[i+1]
    X_test = X_scaled[i+1:i+2]
    y_test_open = y_open[i+1:i+2]
    y_test_close = y_close[i+1:i+2]

    # Train Random Forest Regressor for NextOpen
    model_open = RandomForestRegressor(n_estimators=100, random_state=42)
    model_open.fit(X_train, y_train_open)

    # Train Random Forest Regressor for NextClose
    model_close = RandomForestRegressor(n_estimators=100, random_state=42)
    model_close.fit(X_train, y_train_close)

    # Make predictions
    predicted_open = model_open.predict(X_test)[0]  # Predicted opening price for the current day
    predicted_next_open = model_open.predict(X_scaled[i+2:i+3])[0]  # Predicted opening price for the next day
    predicted_next_close = model_close.predict(X_scaled[i+2:i+3])[0]  # Predicted closing price for the next day

    # Evaluate predictions
    actual_open = data['Open'].iloc[i+1]  # Actual opening price for the current day
    mse_open = mean_squared_error(y_test_open, [predicted_next_open])
    mse_close = mean_squared_error(y_test_close, [predicted_next_close])
    mse_current_open = mean_squared_error([actual_open], [predicted_open])  # MSE for current day's open prediction
    mse_open_list.append(mse_open)
    mse_close_list.append(mse_close)
    mse_current_open_list.append(mse_current_open)

    # Make decision based on predicted NextOpen price
    prev_open = data['Open'].iloc[i]  # Opening price of the previous trading day
    price_change = (predicted_next_open - predicted_open) / predicted_open * 100

    transaction_fee = 2.0  # 1% for buying + 1% for selling
    if price_change > transaction_fee:
        decision = 'Buy'
    elif price_change < -transaction_fee:
        decision = 'Sell'
    else:
        decision = 'Hold'
    decisions.append(decision)

    # Determine actual action based on actual NextOpen price
    actual_next_open = y_test_open.iloc[0]
    actual_price_change = (actual_next_open - actual_open) / actual_open * 100
    if actual_price_change > transaction_fee:
        actual_action = 'Buy'
    elif actual_price_change < -transaction_fee:
        actual_action = 'Sell'
    else:
        actual_action = 'Hold'
    actual_actions.append(actual_action)

    # Print results for each day
    print(f"Date: {test_date.date()}")
    print(f"Previous Open: {prev_open:.2f}")
    print(f"Predicted Open (Current Day): {predicted_open:.2f}")
    print(f"Actual Open (Current Day): {actual_open:.2f}")
    print(f"Predicted NextOpen (Next Day): {predicted_next_open:.2f}")
    print(f"Actual NextOpen (Next Day): {actual_next_open:.2f}")
    print(f"Predicted NextClose (Next Day): {predicted_next_close:.2f}")
    print(f"Actual NextClose (Next Day): {y_test_close.iloc[0]:.2f}")
    print(f"Decision: {decision}")
    print(f"Actual Action: {actual_action}")
    print(f"MSE (Current Open): {mse_current_open:.2f}")
    print(f"MSE (NextOpen): {mse_open:.2f}")
    print(f"MSE (NextClose): {mse_close:.2f}")
    print("------")

# Print average MSE across all days
print(f"Average Current Open Prediction MSE: {np.mean(mse_current_open_list):.2f}")
print(f"Average NextOpen Prediction MSE: {np.mean(mse_open_list):.2f}")
print(f"Average NextClose Prediction MSE: {np.mean(mse_close_list):.2f}")

# Evaluate decision-making performance
accuracy = accuracy_score(actual_actions, decisions)
print(f"\nDecision Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(actual_actions, decisions))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(actual_actions, decisions, labels=['Buy', 'Hold', 'Sell']))

