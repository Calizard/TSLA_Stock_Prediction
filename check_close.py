import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('TSLA_Combined.csv')

# Step 2: Check if 'Close' and 'Adj Close' columns have different values
mismatch_rows = df[df['Close'] != df['Adj Close']]

# Step 3: Print the result
if not mismatch_rows.empty:
    print("The 'Close' and 'Adj Close' columns have different values in the following rows:")
    print(mismatch_rows)
else:
    print("The 'Close' and 'Adj Close' columns are identical in all rows.")