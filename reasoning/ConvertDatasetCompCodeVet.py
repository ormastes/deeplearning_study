import os
import json
from datasets import load_from_disk, Dataset, DatasetDict

# Load the dataset from disk
mercury_dataset = load_from_disk("/workspace/data/model/reasoning/raw/CompCodeVet")

# Print basic info from the dataset
print("Elfsong/Mercury dataset:")
for split in mercury_dataset.keys():
    print("  Split:", split)
    print("  Column Names:", mercury_dataset[split].column_names)
    print("  First record:", mercury_dataset[split][0])
    print("  -----")

# Prepare containers for the converted data for JSON output and the new dataset
converted_json = {}
converted_ds_dict = {}

# Process each split of the dataset
for split in mercury_dataset.keys():
    json_records = []
    for record in mercury_dataset[split]:
        # "Test Object" is taken from the pretty_content field (which may be a list)
        test_object = record.get("pretty_content", [])

        # "Test Target" is extracted from each solution's "solution" value in the solutions list
        solutions = record.get("solutions", [])
        test_targets = [sol.get("solution", "") for sol in solutions]

        for target in test_targets:
            for object in test_object:
                if not target:
                    print("Empty target found in record:", record)
                if not test_object:
                    print("Empty test object found in record:", record)

                new_record = {
                    "Test Target": target,
                    "Test Object": object
                }
                json_records.append(new_record)

    converted_json[split] = json_records
    # Create a Hugging Face Dataset from the list of new records for this split
    converted_ds_dict[split] = Dataset.from_list(json_records)

# Print the converted JSON structure
print("\nConverted JSON:")
print(json.dumps(converted_json, indent=2))

# Save the new dataset to disk at the specified path
converted_dataset = DatasetDict(converted_ds_dict)
dataset_save_path = "/workspace/data/model/reasoning/refined/test_object"
converted_dataset.save_to_disk(dataset_save_path)
print(f"\nConverted dataset saved to '{dataset_save_path}'")

# Save the converted JSON to a file in the specified directory
json_save_dir = "/workspace/data/model/reasoning/refined/test_object_json"
os.makedirs(json_save_dir, exist_ok=True)
json_file_path = os.path.join(json_save_dir, "converted_data.json")
with open(json_file_path, "w", encoding="utf-8") as f:
    json.dump(converted_json, f, indent=2)
print(f"\nConverted JSON saved to '{json_file_path}'")
