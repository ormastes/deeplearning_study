# Import the ClangReplInterface (assumed available in the same directory)
from ClangReplInterface import ClangReplInterface
import xml.etree.ElementTree as ET

# --------------------------
# Define Loss Penalty Settings
# --------------------------
ERROR_PENALTY = 5.0
FAIL_PENALTY = 2.0
TOKEN_PENALTY = 1.0
TOKEN_TAG_PENALTY = 0.5
LENGTH_PENALTY = 1.0
MIN_TAG_LENGTH = 5  # Minimum character length required for a valid tag
MIN_OUTPUT_LENGTH = 20  # Minimum character length required for a valid output

clang_repl_interface = ClangReplInterface()

def compute_reasoning_penalty(question, generated_text):
    """
    Run the Clang test and compute a penalty based on the result.
    Also, apply additional penalties if required tokens are missing or length is too short.
    """
    penalty = 0.0
    xml_generated_text = "<Test Case>\n" + generated_text + "\n</Test Case>"
    # parse xml string
    try:
        root = ET.fromstring(xml_generated_text)
    except ET.ParseError as e:
        return ERROR_PENALTY, "ERROR"
    clang_test = root.find('Clang-repl Test').text.strip()
    if clang_test is None:
        return ERROR_PENALTY, "ERROR"
    # Run the Clang REPL test using the provided test command
    resultDict = clang_repl_interface.do_execute_sync(clang_test)
    result = resultDict['status']
    if result == "ERROR":
        penalty += ERROR_PENALTY
    elif result == "FAIL":
        penalty += FAIL_PENALTY
    # Check for required tokens in the output. For example, ensure output has tags:
    required_tags = ["<Test Object>", "<Input Data>", "<Expected Output>"]
    for tag in required_tags:
        if tag not in generated_text:
            penalty += TOKEN_PENALTY
        else:
            if len(root.find(tag).text.strip()) < MIN_TAG_LENGTH:
                penalty += TOKEN_PENALTY
    # Check minimum length
    if len(generated_text) < MIN_OUTPUT_LENGTH:
        penalty += LENGTH_PENALTY
    return penalty, result