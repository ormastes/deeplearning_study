
# --------------------------
# XML Dataset Parsing Functions
# --------------------------
def parse_xml_file(file_path):
    """
    Parse an XML file and extract test fields.
    Expected tags:
      <Test Target>, <Test Object>, <Input Data>,
      <Expected Output>, <Clang-repl Test>
    """
    tree = ET.parse(file_path)
    root = tree.getroot()
    examples = []
    for test in root.findall('TestCase'):
        test_target = test.find('Test Target').text.strip()
        # For training, we assume the ground-truth Test Object, Input Data, Expected Output are provided.
        test_object = test.find('Test Object').text.strip()
        input_data = test.find('Input Data').text.strip()
        expected_output = test.find('Expected Output').text.strip()
        clang_test = test.find('Clang-repl Test').text.strip()
        examples.append({
            'test_target': test_target,
            'test_object': test_object,
            'input_data': input_data,
            'expected_output': expected_output,
            'clang_test': clang_test
        })
    return examples


# --------------------------
# Custom Dataset for XML Data
# --------------------------
from torch.utils.data import Dataset


class XMLReasoningDataset(Dataset):
    def __init__(self, xml_files):
        self.examples = []
        for file in xml_files:
            self.examples.extend(parse_xml_file(file))
        # Each example: use Test Target as input and the combination of Test Object,
        # Input Data, and Expected Output as the target text.
        self.data = [{
            'input': ex['test_target'],
            'target': f"<Test Object>{ex['test_object']}</Test Object>\n"
                      f"<Input Data>{ex['input_data']}</Input Data>\n"
                      f"<Expected Output>{ex['expected_output']}</Expected Output>",
            'clang_test': ex['clang_test']
        } for ex in self.examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --------------------------
# Tokenization Function
# --------------------------
def tokenize_example(example):
    input_text = example['input']
    target_text = example['target']
    # Concatenate input and target with a separator if needed
    combined = input_text + "\n" + target_text
    tokens = tokenizer.encode(combined, max_length=config.context_len, truncation=True, padding='max_length')
    if tokenizer.pad_token_id in tokens:
        length = tokens.index(tokenizer.pad_token_id)
    else:
        length = len(tokens)
    return {'tokens': tokens, 'length': length, 'clang_test': example['clang_test'],
            'input_text': input_text, 'target_text': target_text}


def tokenize_and_cache_xml(xml_files, cache_file):
    if os.path.exists(cache_file):
        print(f"Loading cached tokenized data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            tokenized = pickle.load(f)
    else:
        print(f"Parsing and tokenizing XML files, using {cpu_count()} cores...")
        dataset = XMLReasoningDataset(xml_files)
        with Pool(cpu_count()) as p:
            tokenized_list = p.map(tokenize_example, dataset.data)
        tokenized = tokenized_list
        with open(cache_file, 'wb') as f:
            print(f"Storing tokenized data in {cache_file}...")
            pickle.dump(tokenized, f)
    return tokenized


# --------------------------
# Prepare Dataset and DataLoader
# --------------------------
xml_files = ['./data/test_data.xml']  # List your XML files here
cache_file = os.path.join(config.cache_dir, "xml_tokenized.pkl")
tokenized = tokenize_and_cache_xml(xml_files, cache_file)