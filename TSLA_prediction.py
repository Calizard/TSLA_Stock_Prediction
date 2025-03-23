import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load historical data
data = pd.read_csv('TSLA_Combined_processed.csv')  # Replace with your file path
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Feature engineering
data['PrevClose'] = data['Close'].shift(1)  # Previous day's closing price
data['PrevOpen'] = data['Open'].shift(1)    # Previous day's opening price
data['PrevVolume'] = data['Volume'].shift(1)  # Previous day's trading volume
data.dropna(inplace=True)  # Drop rows with missing values

# Define features (X) and target (y)
X = data[['Open', 'Close', 'PrevClose', 'PrevOpen', 'Volume', 'PrevVolume']]
y = data['NextOpen']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Rolling-window validation with a fixed time horizon (e.g., 2 years)
time_horizon = 2 * 252  # 2 years of trading days (assuming 252 trading days per year)
mse_list = []  # Store MSE for NextOpen predictions
decisions = []  # Store decisions (Buy/Sell/Hold)
actual_actions = []  # Store actual actions based on NextOpen price

# Iterate through the dataset
for i in range(len(data) - 1):
    # Ensure the training set has at least `time_horizon` days of data
    if i + 1 < time_horizon:
        continue  # Skip predictions for the first `time_horizon` days

    # Training period: Most recent `time_horizon` days up to the current day
    train_start_index = i + 1 - time_horizon  # Start index for training set
    train_end_index = i + 1  # End index for training set
    train_start_date = data['Date'].iloc[train_start_index]
    train_end_date = data['Date'].iloc[train_end_index]

    X_train = X_scaled[train_start_index:train_end_index]
    y_train = y[train_start_index:train_end_index]

    # Testing period: Next day
    test_date = data['Date'].iloc[i+1]
    X_test = X_scaled[i+1:i+2]
    y_test = y[i+1:i+2]

    # Train Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make prediction
    y_pred = model.predict(X_test)[0]

    # Evaluate prediction
    mse = mean_squared_error(y_test, [y_pred])
    mse_list.append(mse)

    # Make decision based on predicted NextOpen price
    current_open = data['Open'].iloc[i]
    predicted_next_open = y_pred
    price_change = (predicted_next_open - current_open) / current_open * 100

    transaction_fee = 2.0  # 1% for buying + 1% for selling
    if price_change > transaction_fee:
        decision = 'Buy'
    elif price_change < -transaction_fee:
        decision = 'Sell'
    else:
        decision = 'Hold'
    decisions.append(decision)

    # Determine actual action based on actual NextOpen price
    actual_next_open = y_test.iloc[0]
    actual_price_change = (actual_next_open - current_open) / current_open * 100
    if actual_price_change > transaction_fee:
        actual_action = 'Buy'
    elif actual_price_change < -transaction_fee:
        actual_action = 'Sell'
    else:
        actual_action = 'Hold'
    actual_actions.append(actual_action)

    # Print results for each day
    print(f"Date: {test_date.date()}")
    print(f"Current Open: {current_open:.2f}")
    print(f"Predicted NextOpen: {predicted_next_open:.2f}")
    print(f"Actual NextOpen: {actual_next_open:.2f}")
    print(f"Decision: {decision}")
    print(f"Actual Action: {actual_action}")
    print(f"MSE: {mse:.2f}")
    print("------")

# Print average MSE across all days
print(f"Average NextOpen Prediction MSE: {np.mean(mse_list):.2f}")

# Evaluate decision-making performance
accuracy = accuracy_score(actual_actions, decisions)
print(f"\nDecision Accuracy: {accuracy:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(actual_actions, decisions))

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(actual_actions, decisions, labels=['Buy', 'Sell', 'Hold']))