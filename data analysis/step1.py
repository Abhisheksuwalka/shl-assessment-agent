import pandas as pd
import json

# Define file path
file_path = '../data/shl_product_catalog.json'

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Step 1: Basic Inspection
print("--- Data Shape ---")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

print("--- Columns and Data Types ---")
print(df.dtypes, "\n")

print("--- Missing Values ---")
# Check for nulls and empty strings or empty lists
missing_stats = pd.DataFrame({
    'Nulls': df.isnull().sum(),
    'Empty Strings': (df == '').sum(),
    'Empty Lists': df.apply(lambda col: col.map(lambda x: x == [] if isinstance(x, list) else False).sum())
})
print(missing_stats)
