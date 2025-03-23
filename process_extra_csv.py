import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('TSLA_Post2022.csv')

# Step 2: Multiply the specified columns by 3
columns_to_multiply = ['Open', 'High', 'Low', 'Close', 'Adj Close']
df[columns_to_multiply] = (df[columns_to_multiply] * 3).round(2)

# Step 3: Process the 'Volume' column
# Remove commas divide by 3, and round to the nearest 100
df['Volume'] = df['Volume'].str.replace(',', '').astype(int)  # Remove commas and convert to integer
df['Volume'] = (df['Volume'] / 3).round(-2).astype(int)  # Divide by 3, round to nearest 100, and convert to integer

# Step 4: Save the modified DataFrame to a new CSV file
df.to_csv('modified_file.csv', index=False)

# Display the modified DataFrame
print(df)