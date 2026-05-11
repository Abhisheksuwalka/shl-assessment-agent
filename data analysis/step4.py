import pandas as pd
import json
import re
from collections import Counter

# Define file path
file_path = '../data/shl_product_catalog.json'

# Load the JSON data
with open(file_path, 'r') as f:
    data = json.load(f)

# Convert to pandas DataFrame
df = pd.DataFrame(data)

print("=== Step 4: Text Analysis ===\n")

# A basic set of stop words including generic test-related words we want to ignore
stop_words = set([
    "the", "and", "of", "to", "a", "in", "for", "is", "that", "this", "on", "with", "as", "are", 
    "or", "an", "be", "test", "measures", "knowledge", "skills", "ability", "designed", 
    "questions", "topics", "candidates", "understanding", "will", "which", "can", "their", 
    "such", "from", "by", "how", "it", "at", "who", "have", "you", "they", "not", "has", 
    "these", "following", "includes", "candidate", "about", "measure", "covers", "also",
    "use", "used", "using", "work"
])

def get_top_words(series, n=20):
    # Combine all text, convert to lowercase
    text = " ".join(series.dropna().tolist()).lower()
    # Extract words with 3 or more letters
    words = re.findall(r'\b[a-z]{3,}\b', text)
    # Filter out stop words
    filtered_words = [w for w in words if w not in stop_words]
    return Counter(filtered_words).most_common(n)

print("--- Top words in Product 'name' ---")
for word, count in get_top_words(df['name'], 15):
    print(f"{word}: {count}")

print("\n--- Top words in Product 'description' ---")
for word, count in get_top_words(df['description'], 15):
    print(f"{word}: {count}")

print("\n--- Top descriptive words: Executive vs. Mid-Professional ---")
# Explode job_levels to filter
df_exploded = df.explode('job_levels')
exec_desc = df_exploded[df_exploded['job_levels'] == 'Executive']['description']
mid_desc = df_exploded[df_exploded['job_levels'] == 'Mid-Professional']['description']

print("\nExecutive Level descriptions:")
for word, count in get_top_words(exec_desc, 10):
    print(f"{word}: {count}")

print("\nMid-Professional Level descriptions:")
for word, count in get_top_words(mid_desc, 10):
    print(f"{word}: {count}")
