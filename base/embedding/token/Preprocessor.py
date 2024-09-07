import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def preprocess_text(text):
    """Replace Windows-style newlines with Linux-style newlines."""
    return text.replace('\r\n', '\n').replace('\r', '\n')


def get_word_category(word):
    """Return the category (part of speech) of a word."""
    return pos_tag([word])[0][1]


def is_independent_clause(words):
    """Check if the list of words form an independent clause."""
    # For simplicity, consider a clause with subject (noun) and predicate (verb) as independent
    has_noun = any(get_word_category(word).startswith('NN') for word in words)
    has_verb = any(get_word_category(word).startswith('VB') for word in words)
    return has_noun and has_verb


def find_division_points(sentence):
    """Identify potential division points in a sentence."""
    division_points = []
    words = word_tokenize(sentence)
    tagged_words = pos_tag(words)

    # Define division point keywords and symbols
    division_keywords = {'and', 'but', 'or', 'because', 'so', 'which', 'who', 'where', 'when', 'that'}
    conjunctions = {'and', 'but', 'or', 'so'}
    punctuation_marks = {'.', ',', ';', ':'}

    # Define open, close, and open_close symbols
    open_symbols = {'[', '(', '{', '/*'}
    close_symbols = {']', ')', '}', '*/'}
    open_close_symbols = {"'", '"', "'''", '"""', '`'}

    for i, (word, tag) in enumerate(tagged_words):
        if word.lower() in division_keywords:
            # Check if this is a conjunction connecting independent clauses
            if word.lower() in conjunctions:
                before = words[:i]
                after = words[i + 1:]
                if is_independent_clause(before) and is_independent_clause(after):
                    division_points.append(i)
            else:
                division_points.append(i)
        elif word in punctuation_marks:
            division_points.append(i)
        elif word in open_symbols:
            words[i] += "<|v_key|><|v_value|>"
        elif word in close_symbols:
            words[i] += "<|v_key|><|v_value|>"
        elif word in open_close_symbols:
            words[i] += "<|v_key|><|v_value|>"

    return division_points


def divide_sentence(sentence):
    """Divide a sentence based on conjunctions, clauses, punctuation, etc."""
    division_points = find_division_points(sentence)
    words = word_tokenize(sentence)

    # If no division points found, return the sentence as is
    if not division_points:
        return [sentence]

    # Split the sentence at the division points
    divided_sentences = []
    start = 0
    for point in division_points:
        divided_sentences.append(' '.join(words[start:point]).strip())
        start = point + 1

    # Add the remaining part of the sentence
    if start < len(words):
        divided_sentences.append(' '.join(words[start:]).strip())

    # Further divide sentences using punctuation marks
    final_sentences = []
    for s in divided_sentences:
        final_sentences.extend(sent_tokenize(s))

    # Simplify complex structures and remove unnecessary details
    simplified_sentences = []
    for s in final_sentences:
        if ',' in s:
            parts = s.split(',')
            simplified_sentences.extend(part.strip() for part in parts)
        else:
            simplified_sentences.append(s.strip())

    return "<|v_key|><|v_value|>".join(simplified_sentences)