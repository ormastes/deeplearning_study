import subprocess
import time
import os
from urllib.parse import urlparse
from pylspclient import LspClient, JsonRpcEndpoint, JsonRpcStreamReader, JsonRpcStreamWriter


def get_file_path_from_uri(uri):
    # Convert file:// URI to local file path.
    parsed = urlparse(uri)
    return os.path.abspath(os.path.join(parsed.netloc, parsed.path))


def extract_text_from_range(file_path, range_obj):
    """
    Extract text from file_path using the provided LSP range.
    The range_obj should contain 'start' and 'end', each with 'line' and 'character'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start = range_obj['start']
    end = range_obj['end']

    if start['line'] == end['line']:
        return lines[start['line']][start['character']:end['character']]
    else:
        extracted = []
        # First line from start character to end of line.
        extracted.append(lines[start['line']][start['character']:])
        # Full lines in between.
        for i in range(start['line'] + 1, end['line']):
            extracted.append(lines[i])
        # Last line from beginning to end character.
        extracted.append(lines[end['line']][:end['character']])
        return "".join(extracted)


def start_clangd(compile_commands_dir=None):
    """
    Start clangd with optional compile_commands.json support.
    If compile_commands_dir is provided, clangd will look for compile_commands.json there.
    """
    cmd = ['clangd', '--log=verbose']
    if compile_commands_dir:
        cc_path = os.path.join(compile_commands_dir, "compile_commands.json")
        if os.path.isfile(cc_path):
            cmd.append(f'--compile-commands-dir={compile_commands_dir}')
        else:
            print(f"Warning: compile_commands.json not found in {compile_commands_dir}")
    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    return process


def initialize_client(process, root_uri):
    endpoint = JsonRpcEndpoint(
        JsonRpcStreamReader(process.stdout),
        JsonRpcStreamWriter(process.stdin)
    )
    client = LspClient(endpoint)
    # Initialize LSP with minimal capabilities.
    client.initialize_process(root_uri=root_uri, process_id=1, capabilities={})
    client.initialized()
    return client


def open_document(client, file_path):
    """
    Open a file and notify clangd.
    """
    file_uri = "file://" + os.path.abspath(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    # For C/C++ source files, use languageId "cpp".
    client.did_open(file_uri, languageId="cpp", version=1, text=text)
    return file_uri


def get_definition(client, file_uri, line, character):
    """
    Request the definition for a symbol at the given position.
    This function returns a dictionary containing the location info and the full text of the definition.
    """
    definition = client.definition(file_uri, line, character)
    if not definition:
        return None

    # clangd might return a single dict or a list of locations.
    location = definition[0] if isinstance(definition, list) else definition
    def_uri = location.get('uri')
    def_range = location.get('range')
    def_file = get_file_path_from_uri(def_uri)
    def_text = extract_text_from_range(def_file, def_range) if def_range else "<range missing>"
    return {"location": location, "text": def_text}


def get_document_symbols(client, file_uri):
    """
    Retrieve document symbols for the file.
    """
    return client.document_symbol(file_uri)


def main():
    # Set these paths appropriately.
    file_path = "example.cpp"  # path to your C/C++ source file
    project_root = "/path/to/your/project"  # project root directory (with compile_commands.json)
    root_uri = "file://" + os.path.abspath(project_root)

    # Start clangd with compile_commands.json support.
    process = start_clangd(compile_commands_dir=project_root)
    client = initialize_client(process, root_uri)

    # Open the document.
    file_uri = open_document(client, file_path)

    # Allow some time for clangd to index the file.
    time.sleep(2)

    # === Retrieve and Print Definition for a Symbol ===
    # Example: get definition of a symbol at line 10, character 5.
    definition_result = get_definition(client, file_uri, line=10, character=5)
    if definition_result:
        print("Definition for symbol at (10,5):")
        print("Location:", definition_result["location"])
        print("Definition Text:\n", definition_result["text"])
    else:
        print("No definition found at the given position.")

    # === Retrieve Document Symbols and Extract Full Definitions ===
    symbols = get_document_symbols(client, file_uri)
    if symbols:
        print("\nDocument Symbols:")
        for symbol in symbols:
            kind = symbol.get('kind')
            name = symbol.get('name')
            # For classes (kind 5), list public functions and variables with their full definitions.
            if kind == 5:
                children = symbol.get('children', [])
                public_members = []
                for child in children:
                    child_kind = child.get('kind')
                    # 12: Function, 13: Variable.
                    if child_kind in [12, 13]:
                        child_range = child.get('range')
                        # Assume the member is defined in the same file.
                        member_text = extract_text_from_range(os.path.abspath(file_path),
                                                              child_range) if child_range else "<no range>"
                        public_members.append(f"{child.get('name')}: {member_text.strip()}")
                # Join with ";\n" as requested.
                signature = ";\n".join(public_members) if public_members else "No public members found"
                print(f"Class: {name}\nPublic members:\n{signature}\n")
            elif kind in [12, 13]:
                # For function (12) and variable (13), extract the full definition text.
                symbol_range = symbol.get('range')
                def_text = extract_text_from_range(os.path.abspath(file_path),
                                                   symbol_range) if symbol_range else "<no range>"
                symbol_type = "Function" if kind == 12 else "Variable"
                print(f"{symbol_type}: {name}\nDefinition:\n{def_text.strip()}\n")
            else:
                print(f"Other symbol (kind {kind}): {name}")
    else:
        print("No document symbols found.")

    # Clean up: terminate clangd process.
    process.terminate()


if __name__ == '__main__':
    main()
