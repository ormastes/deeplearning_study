import json
import os
from datasets import load_from_disk

# Load the dataset from disk
cpp_dataset = load_from_disk("/workspace/data/model/reasoning/raw/cpp_ut_bench")

# Define the output directory and ensure it exists
output_dir = "/workspace/data/model/reasoning/raw/cpp_ut_bench_json"
os.makedirs(output_dir, exist_ok=True)

print("Nutanix/CPP-UNITTEST-BENCH dataset:")
# Loop through each split in the dataset

for split in cpp_dataset.keys():
    print("  Split:", split)
    print("  Column Names:", cpp_dataset[split].column_names)
    # Define the output file path for the JSON file
    output_file = os.path.join(output_dir, f"{split}.json")
    # Get the first record from the current split
    first_record = cpp_dataset[split]
    with open(output_file, "w") as f:
        new_record_file = {}
        for record in first_record:
            print("record:", record)
            # get ID, Code, 'Unit Test - (Ground Truth)' and save
            new_record = {
                "Code": record.get("Code"),
                "Unit Test - (Ground Truth)": record.get("Unit Test - (Ground Truth)")
            }
            new_record_file[record.get("ID")] = new_record

        json.dump(new_record_file, f, indent=4)

    print(f"Saved first record of split '{split}' to {output_file}")
