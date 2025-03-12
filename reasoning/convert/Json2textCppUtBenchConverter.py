import json
import sys


def convert_json_to_dataset(input_file, output_file):
    """
    Reads the JSON file from `input_file` and writes a dataset file to `output_file`
    where each sample is formatted with XML-like tags for selected keys.

    Expected keys:
      - "Test Target"
      - "Test Object"
      - "Input Data"
      - "Expected Output"
      - "Clang-repl Test"

    Each tag will have a newline after the opening tag and before the closing tag.
    A unique marker '####SAMPLE_END####' is added after each sample.
    """
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Wrap a single dict into a list if needed.
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Unexpected JSON format: expected a dict or a list of dicts.")

    marker = "####SAMPLE_END####"

    with open(output_file, "w", encoding="utf-8") as out:
        for sample in data:
            for key in ["Test Target", "Test Object", "Input Data", "Expected Output", "Clang-repl Test"]:
                if key in sample:
                    content = sample[key]
                    if isinstance(content, list):
                        content = "\n".join(content)
                    out.write(f"<{key}>\n{content}\n</{key}>\n")
            out.write(marker + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python converter.py input.json output.txt")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_json_to_dataset(input_file, output_file)
    print(f"Conversion complete. Dataset file saved as: {output_file}")
