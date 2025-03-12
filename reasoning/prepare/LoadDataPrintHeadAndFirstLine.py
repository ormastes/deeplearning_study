import os
import pandas as pd
from datasets import load_from_disk

# -----------------------------
# 1. Nutanix/CPP-UNITTEST-BENCH dataset
# -----------------------------
cpp_dataset = load_from_disk("/workspace/data/model/reasoning/raw/cpp_ut_bench")
print("Nutanix/CPP-UNITTEST-BENCH dataset:")
# In case the dataset is a DatasetDict with one or more splits:
for split in cpp_dataset.keys():
    print("  Split:", split)
    print("  Column Names:", cpp_dataset[split].column_names)
    print("  First record:", cpp_dataset[split][0])
    print("  -----")

# -----------------------------
# 2. Elfsong/Mercury dataset (saved as CompCodeVet)
# -----------------------------
mercury_dataset = load_from_disk("/workspace/data/model/reasoning/raw/CompCodeVet")
print("Elfsong/Mercury dataset:")
for split in mercury_dataset.keys():
    print("  Split:", split)
    print("  Column Names:", mercury_dataset[split].column_names)
    print("  First record:", mercury_dataset[split][0])
    print("  -----")

# -----------------------------
# 3. CITYWALK dataset (from CITYWALK.zip)
# -----------------------------
# Assuming the zip was extracted to the directory below:
citywalk_dir = "/workspace/data/model/reasoning/raw/CITYWALK"
files = os.listdir(citywalk_dir)
# Look for CSV file(s) in the directory
csv_files = [f for f in files if f.lower().endswith('.csv')]

if csv_files:
    csv_path = os.path.join(citywalk_dir, csv_files[0])
    citywalk_df = pd.read_csv(csv_path)
    print("CITYWALK dataset:")
    print("  Column Names:", citywalk_df.columns.tolist())
    if not citywalk_df.empty:
        print("  First row:", citywalk_df.iloc[0].to_dict())
    else:
        print("  (Dataset is empty)")
else:
    print("No CSV file found in CITYWALK folder.")
