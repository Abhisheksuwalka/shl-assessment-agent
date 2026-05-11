import pandas as pd
import json

# Define file path
file_path = '../data/shl_product_catalog.json'

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

print("=== Step 2: Univariate Analysis ===\n")

print("--- 1. Remote and Adaptive Status ---")
print("Remote:")
print(df['remote'].value_counts())
print("\nAdaptive:")
print(df['adaptive'].value_counts(), "\n")

print("--- 2. Top 'Keys' (Product Categories) ---")
# Explode the lists into separate rows to count properly
keys_exploded = df.explode('keys')
print(keys_exploded['keys'].value_counts().head(10), "\n")

print("--- 3. Top Job Levels ---")
job_levels_exploded = df.explode('job_levels')
print(job_levels_exploded['job_levels'].value_counts().head(10), "\n")

print("--- 4. Languages ---")
languages_exploded = df.explode('languages')
print(languages_exploded['languages'].value_counts().head(10), "\n")

print("--- 5. Duration (Extracting numeric values) ---")
# The duration raw often looks like "Approximate Completion Time in minutes = 30" or "30 minutes"
# We'll use regex to extract the first number found in the duration column
df['duration_numeric'] = df['duration_raw'].str.extract('(\d+)').astype(float)

print(df['duration_numeric'].describe())
