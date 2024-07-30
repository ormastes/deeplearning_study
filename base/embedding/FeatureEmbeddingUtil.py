import os
import pandas as pd
from itertools import combinations
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

DATA_PATH= '/workspace/data'

# Ensure protocol buffers use Python implementation
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Download and unzip dataset if it doesn't exist
if not os.path.exists(f"{DATA_PATH}/wordnet-synonyms"):
    os.system(f"kaggle datasets download -d duketemon/wordnet-synonyms -p {DATA_PATH}")
    os.system(f"unzip {DATA_PATH}/wordnet-synonyms.zip -d {DATA_PATH}/wordnet-synonyms")

def _save(sets, name):
    output_df = pd.DataFrame({'part_of_speech': [str(p) for p, s in sets],
                  'merged_synonyms': [';'.join(str(s)) for p, s in sets]})

    output_dir = f"{DATA_PATH}/feature_embedding"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = f"{output_dir}/{name}.csv"
    output_df.to_csv(output_path, index=False)
    return output_df

def save(sets, name):
    import pickle
    with open(f"{DATA_PATH}/feature_embedding/{name}.pkl", "wb") as f:
        pickle.dump(sets, f)
    return sets

# Load the dataset
df = pd.read_csv(f"{DATA_PATH}/wordnet-synonyms/synonyms.csv")

# Prepare synonym sets and part_of_speech mappings
synonym_sets = []
for lemma, part_of_speech, syns in df.itertuples(index=False):
    syns = set(str(syns).split(';'))
    syns.add(lemma)
    synonym_sets.append((part_of_speech, frozenset(syns)))

# Function to determine if two sets of words are similar within a threshold
def are_similar(args):
    set1, set2, threshold = args
    common_words = set1.intersection(set2)
    diff1 = len(set1 - common_words)
    diff2 = len(set2 - common_words)
    max_diff = max(diff1, diff2)
    larger_set_size = max(len(set1), len(set2))
    return max_diff / larger_set_size <= threshold

def merge_sets(args):
    p1, s1, synonym_sets = args
    new_set = s1
    for p2, s2 in synonym_sets:
        if p1 == p2 and are_similar((new_set, s2, 0.1)):
            new_set = new_set.union(s2)
    return (p1, new_set)

print(f"Rows before merge: {len(synonym_sets)}")

save(synonym_sets, "synonyms")

# Merge sets using multiprocessing with progress bar
with Pool(cpu_count()) as pool:
    merged_sets = list(tqdm(pool.imap(merge_sets, [(p1, s1, synonym_sets) for p1, s1 in synonym_sets]), total=len(synonym_sets)))

# Remove duplicates
seen_sets = set()
unique_merged_sets = []
for p, s in merged_sets:
    if s not in seen_sets:
        unique_merged_sets.append((p, s))
        seen_sets.add(s)

print(f"Rows after merge: {len(unique_merged_sets)}")

# Save result to a file
output_df = save(unique_merged_sets, "unique_merged_sets")

print(f"Final row count: {len(output_df)}")

