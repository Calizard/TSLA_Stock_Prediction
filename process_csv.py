import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('TSLA_Combined.csv')

# Step 2: Remove the unwanted columns
df = df.drop(columns=['High', 'Low', 'Adj Close'])

# Step 3: Add the "NextOpen" column
df['NextOpen'] = df['Open'].shift(-1)  # Shift the 'Open' column up by one row

# Step 4: Save the modified DataFrame to a new CSV file (optional)
df.to_csv('TSLA_Combined_processed.csv', index=False)