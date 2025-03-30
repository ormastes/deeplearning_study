import json
from datasets import load_from_disk

# Load the dataset from disk
mercury_dataset = load_from_disk("/workspace/data/model/reasoning/raw/CompCodeVet")

# Print basic info from the dataset
print("Elfsong/Mercury dataset:")
for split in mercury_dataset.keys():
    print("  Split:", split)
    print("  Column Names:", mercury_dataset[split].column_names)
    print("  First record:", mercury_dataset[split][0])
    print("  -----")

# Convert each record into new JSON structure with "Test Object" and "Test Target"
converted_data = {}
for split in mercury_dataset.keys():
    converted_records = []
    for record in mercury_dataset[split]:
        # "Test Object" is the pretty_content field
        test_object = record.get("pretty_content", [])

        # "Test Target" is extracted from each solution's "solution" value in the solutions list
        solutions = record.get("solutions", [])
        test_targets = [sol.get("solution", "") for sol in solutions]

        new_record = {
            "Test Object": test_object,
            "Test Target": test_targets
        }
        converted_records.append(new_record)
    converted_data[split] = converted_records

# Print the converted JSON in a pretty format
print("\nConverted JSON:")
print(json.dumps(converted_data, indent=2))
