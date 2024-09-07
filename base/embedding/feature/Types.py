class Synonym:
    def __init__(self, part_of_speech, merged_synonyms):
        self.part_of_speech = part_of_speech
        self.merged_synonyms = merged_synonyms


class SymbolSynonym:
    def __init__(self, symbols, synonyms, synonymed_symbols, part_of_speech):
        self.symbols = symbols
        self.synonyms = synonyms
        self.synonymed_symbols = synonymed_symbols
        self.part_of_speech = part_of_speech
