import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

from base.config.Config import FeatureEmbeddingLLM
from base.embedding.feature.Types import Synonym, SymbolSynonym
import pickle

_symbol_synonyms = None

def get_symbol_synonyms():
    global _symbol_synonyms
    if _symbol_synonyms is None:
        config = FeatureEmbeddingLLM()
        symbols_synonyms_file = f"{config.feature_embedding_dir}/symbols_synonyms.pkl"

        # load the total_synonyms and total_symbols
        with open(symbols_synonyms_file, "rb") as f:
            _symbol_synonyms = pickle.load(f)

    return _symbol_synonyms

special_tokens = set()
special_tokens.update(['<|pad|>',
                       '<|cls|>',  # token for classification
                       '<|mask|>',
                       '<|comma|>',  # token for comma
                       '<|sep|>',  # token for separator
                       '<|period|>',  # token for period
                       '<|question|>',  # token for question
                       '<|capital|>',  # token for capital letters
                       '<|unknown|>',  # token for unknown words
                       '<|definition|>',  # token before definition
                       '<|eot|>',  # end of text
                       '<|eos|>',  # end of sequence
                       '<|bps|>',  # beginning of sequence
                       '<|url|>',  # token for url
                       '<|code|>',  # token for code
                       '<|nlu|>',  # token for natural language understanding
                       '<|hdr|>',  # header token
                       '<|foot|>',  # footer token
                       '<|email|>',  # token to replace email addresses
                       '<|num|>',  # token to replace numeric values
                       '<|time|>',  # token for time expressions
                       '<|date|>',  # token for date expressions
                       '<|phone|>',  # token for phone numbers
                       '<|address|>',  # token for physical addresses
                       '<|user|>',  # token to replace user handles
                       '<|hashtag|>',  # token to replace hashtags
                       '<|spl|>',  # special delimiter token
                       '<|spe|>',  # another special delimiter
                       '<|v_key|>',
                       '<|v_value|>',
                       '<|reserved3|>', '<|reserved4|>', '<|reserved5|>',
                       '<|reserved6|>', '<|reserved7|>', '<|reserved8|>', '<|reserved9|>', '<|reserved10|>'])


if __name__ == "__main__":


    config = FeatureEmbeddingLLM()
    df = config.synonyms

    synonyms = []
    for row in df.itertuples():
        part_of_speech = row.part_of_speech
        merged_synonyms = row.merged_synonyms
        merged_synonyms = merged_synonyms.split(';')
        # arrays to string
        merged_synonyms = ''.join(merged_synonyms)
        first_parenthesis = merged_synonyms.find('{')
        last_parenthesis = merged_synonyms.rfind('}')
        assert first_parenthesis is not None and last_parenthesis is not None
        merged_synonyms = merged_synonyms[first_parenthesis + 1:last_parenthesis].strip()

        assert "<|comma|>" not in merged_synonyms
        assert "<|sep|>" not in merged_synonyms
        # split by , but not "\,"
        merged_synonyms = merged_synonyms.replace("\,", "<|comma|>")
        merged_synonyms = merged_synonyms.replace("\|", "<|sep|>")
        merged_synonyms = merged_synonyms.split(",")
        new_merged_synonyms = []
        for x in merged_synonyms:
            x = x.strip()
            if x.startswith("'") and x.endswith("'"):
                x = x[1:-1]
            x.replace("<|comma|>", ",")
            if x.startswith('"') and x.endswith('"'):
                x = x[1:-1]
            if '|' in x:
                for y in x.split("|"):
                    y = y.replace("<|sep|>", "|")
                    new_merged_synonyms.append(y)
            else:
                x = x.replace("<|sep|>", "|")
                new_merged_synonyms.append(x)
        merged_synonyms = new_merged_synonyms
        synonyms.append(Synonym(part_of_speech, merged_synonyms))

    part_of_speech_set = set()
    # delete from synonyms
    new_synonyms = []
    for synonym in synonyms:
        new_merged_synonyms = [w for w in synonym.merged_synonyms if ' ' not in w]
        # change first letter to lower case
        new_merged_synonyms = set(w[0].lower() + w[1:] for w in new_merged_synonyms)
        if len(new_merged_synonyms) > 1:
            part_of_speech_set.add(synonym.part_of_speech)
            new_synonyms.append(Synonym(synonym.part_of_speech, new_merged_synonyms))
    synonyms = new_synonyms

    wordset = set()
    for synonym in synonyms:
        wordset.update(synonym.merged_synonyms)

    wordlist = list(wordset)
    # order wordlist from longest to shortest
    wordlist.sort(key=lambda x: len(x), reverse=True)

    keywords = set()


    def read_file(file_name, symbols, multiple_words=True):
        with open(file_name, "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                words = line.split('\t')
                if multiple_words and len(words) <= 1:
                    print(f"Error: {line}")
                firstWord = words[0].split(' ')[0]
                # if startwith capital letter
                if firstWord[0].isupper() and len(firstWord) > 1:
                    # count capital letters
                    count = 0
                    for c in firstWord:
                        if c.isupper():
                            count += 1
                    if count > 1:
                        continue
                    firstWord = firstWord.lower()
                symbols.add(firstWord)


    read_file("basic_word.txt", keywords, multiple_words=False)
    read_file("basic_word_meaning.txt", keywords)
    read_file("basic_verb.txt", keywords, multiple_words=False)
    read_file("basic_noun.txt", keywords)
    read_file("program_common.txt", keywords)
    read_file("c_cpp.txt", keywords)
    read_file("txt.txt", keywords)
    read_file("python.txt", keywords)
    read_file("chemical.txt", keywords)
    read_file("economy.txt", keywords)
    read_file("genetic.txt", keywords)
    read_file("javascript.txt", keywords, multiple_words=False)
    read_file("latex.txt", keywords, multiple_words=False)
    read_file("law.txt", keywords)
    read_file("medical.txt", keywords)
    read_file("music.txt", keywords)
    read_file("physics.txt", keywords)

    read_file("korean.txt", keywords, multiple_words=False)

    symbol_synonyms = []

    connector = Synonym("program_connector", [' ', '->', '.', '->*', '.*', '::', '<|capital|>'])
    symbol_synonyms.append(connector)

    statement_ends = Synonym("program_statement_end", ['.', ';', '\n'])
    symbol_synonyms.append(statement_ends)

    block_starts = Synonym("program_block_start", ['{', ':', '.'])
    symbol_synonyms.append(block_starts)

    block_ends = Synonym("program_block_end", ['}', '\n', '.'])
    symbol_synonyms.append(block_ends)

    comment_starts = Synonym("program_comment_start", [',', '(', '//', '#', '/*'])
    symbol_synonyms.append(comment_starts)

    comment_ends = Synonym("program_comment_end", [',', ')', '\n', '*/'])
    symbol_synonyms.append(comment_ends)

    arithmetics = Synonym("program_arithmetics", ['+', '-', '*', '/', '//', '%', '++', '--',
                                                  '==', '!=', '>', '<', '>=', '<=', '<=>', '!', '&&', '||', '~', '&',
                                                  '|',
                                                  '^', '<<', '>>', 'is', 'and', 'or', 'not', 'in', ':=', '+=', '-=',
                                                  '*=',
                                                  '/=', '//=', '%=', '&=', '|=', '^=', '<<=', '>>='])
    symbol_synonyms.append(arithmetics)

    assign = Synonym("program_assign", ['=', ':=', '+=', '-=', '*=',
                                        '/=', '//=', '%=', '&=', '|=', '^=', '<<=', '>>='])
    symbol_synonyms.append(assign)

    open_syn = Synonym("program_open", ['[', '(', '{', '<', ',', '//', '#', '/*', '`', "'", '"', "'''", '"""', '```'])
    symbol_synonyms.append(open_syn)

    close_syn = Synonym("program_close", [']', ')', ')', '}', '>', ',', '\n', '*/', '`', "'", '"', "'''", '"""', '```'])
    symbol_synonyms.append(close_syn)

    symbol_set = set()
    symbol_set.update(['_', '(', ')', ',', ':', '{', '}', '<', '>', ';'])
    symbol_set.update(connector.merged_synonyms)
    symbol_set.update(statement_ends.merged_synonyms)
    symbol_set.update(block_starts.merged_synonyms)
    symbol_set.update(block_ends.merged_synonyms)
    symbol_set.update(comment_starts.merged_synonyms)
    symbol_set.update(comment_ends.merged_synonyms)
    symbol_set.update(arithmetics.merged_synonyms)
    symbol_set.update(assign.merged_synonyms)
    symbol_set.update(open_syn.merged_synonyms)
    symbol_set.update(close_syn.merged_synonyms)

    symbols = Synonym("program_symbols", symbol_set)
    symbol_synonyms.append(symbols)

    alphabet = set()
    alphabet.add('is')
    alphabet.add('and')
    alphabet.add('or')
    alphabet.add('not')
    alphabet.add('in')

    symbol_only = symbol_set - alphabet - special_tokens

    ordered_symbols = list(symbol_only)
    ordered_symbols.sort(key=lambda x: len(x), reverse=True)
    ordered_symbols.remove(' ')
    ordered_symbols = [' '] + ordered_symbols

    print(len(part_of_speech_set))  # adjective, adverb, noun, verb, satellite, (UNKNOWN ?)
    print(len(synonyms))  # 67440
    print(len(wordlist))  # 59764
    print('\n'.join(wordlist[:10]))

    print(len(ordered_symbols))  # 68
    print(len(symbol_synonyms))  # 11

    total_synonyms = synonyms + symbol_synonyms
    print(len(total_synonyms))  # 67440

    # assert wordlist and ordered_symbols are disjoint
    assert len(set(wordlist).intersection(set(ordered_symbols))) == 0
    assert len(set(wordlist).intersection(set(special_tokens))) == 0
    # remove wordlist from basic_words
    keywords = list(keywords)
    keywords = [w for w in keywords if w not in wordlist]
    special_tokens_list = list(special_tokens)
    special_tokens_list.sort(key=lambda x: len(x), reverse=True)
    special_tokens_list.remove('<|pad|>')
    special_tokens_list = ['<|pad|>'] + special_tokens_list
    rest_ordered_symbols = list(special_tokens) + wordlist + keywords
    rest_ordered_symbols.sort(key=lambda x: len(x), reverse=True)
    total_symbols = special_tokens_list + ordered_symbols + rest_ordered_symbols
    synonymed_symbols = special_tokens_list + ordered_symbols + wordlist
    print(len(total_symbols))  # 53069
    total_part_of_speech = set()  # 16 {'program_assign', 'adjective', 'program_comment_end', 'program_statement_end', 'program_block_start', 'program_comment_start', 'satellite', 'program_open', 'program_connector', 'noun', 'verb', 'program_block_end', 'program_close', 'program_symbols', 'program_arithmetics', 'adverb'}
    for synonym in total_synonyms:
        total_part_of_speech.add(synonym.part_of_speech)

    # Save the total_synonyms and total_symbols
    symbol_synonyms = SymbolSynonym(total_symbols, total_synonyms, synonymed_symbols, total_part_of_speech)

    symbols_synonyms_file = f"{config.feature_embedding_dir}/symbols_synonyms.pkl"

    with open(symbols_synonyms_file, "wb") as f:
        pickle.dump(symbol_synonyms, f)

    # load the total_synonyms and total_symbols
    with open(symbols_synonyms_file, "rb") as f:
        symbol_synonyms = pickle.load(f)

    effective_embed_dim = config.embed_dim - 10
    config.voca_embed_dim = effective_embed_dim

