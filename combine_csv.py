import pandas as pd

# Step 1: Read both CSV files into DataFrames
df_part1 = pd.read_csv('TSLA_Part1.csv')
df_part2 = pd.read_csv('TSLA_Part2.csv')

# Step 2: Append the records from df_part2 to df_part1
combined_df = pd.concat([df_part1, df_part2], ignore_index=True)

# Step 3: Revert the changes
# Divide 'Open', 'High', 'Low', 'Close', and 'Adj Close' by 3
columns_to_divide = ['Open', 'High', 'Low', 'Close', 'Adj Close']
combined_df[columns_to_divide] = combined_df[columns_to_divide] / 3

# Multiply 'Volume' by 3
combined_df['Volume'] = combined_df['Volume'] * 3

# Step 4: Save the combined DataFrame to a new CSV file
combined_df.to_csv('TSLA_Combined.csv', index=False)

# Optional: Print the combined DataFrame to verify
print(combined_df)