import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Define file path
file_path = '../data/shl_product_catalog.json'

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Preprocessing from Step 2
df['duration_numeric'] = df['duration_raw'].str.extract(r'(\d+)').astype(float)

print("=== Step 3: Bivariate/Multivariate Analysis ===\n")

print("--- 1. Duration by Adaptive vs. Non-Adaptive ---")
duration_by_adaptive = df.groupby('adaptive')['duration_numeric'].describe()
print(duration_by_adaptive[['count', 'mean', '50%', 'max']], "\n")

print("--- 2. Duration by Product Category (Keys) ---")
# Explode keys to get accurate category durations
df_keys = df.explode('keys')
duration_by_category = df_keys.groupby('keys')['duration_numeric'].describe()
# Filter for categories with at least a few samples and sort by mean duration
duration_by_category = duration_by_category[duration_by_category['count'] >= 5]
print(duration_by_category[['count', 'mean', '50%', 'max']].sort_values(by='mean', ascending=False), "\n")

print("--- 3. Top Categories (Keys) by Job Level ---")
# Explode both keys and job_levels
df_job_keys = df.explode('keys').explode('job_levels')

# Cross tabulation of Job Levels and Keys
df_job_keys = df_job_keys.reset_index(drop=True)
ct = pd.crosstab(df_job_keys['job_levels'], df_job_keys['keys'])

# Let's focus on a few key job levels: 'Entry-Level', 'Mid-Professional', 'Manager', 'Executive'
target_levels = ['Entry-Level', 'Mid-Professional', 'Manager', 'Executive']
ct_filtered = ct.loc[target_levels] if all(lvl in ct.index for lvl in target_levels) else ct

# Calculate percentage within each job level row to see the distribution
ct_pct = ct_filtered.div(ct_filtered.sum(axis=1), axis=0) * 100
# Round to 1 decimal place
ct_pct = ct_pct.round(1)

# Keep the most interesting categories: 'Knowledge & Skills', 'Personality & Behavior', 'Simulations'
key_cols = [col for col in ['Knowledge & Skills', 'Personality & Behavior', 'Simulations', 'Ability & Aptitude'] if col in ct_pct.columns]
import pandas as pd
pd.set_option('display.max_columns', None)
print(ct_pct[key_cols])


