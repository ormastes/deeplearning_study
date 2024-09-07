from base.config.CommonLanguage import CommonLanguageType, PartOfSpeech
from base.config.DataTextLanguage import DataTextType
from base.config.HumanLanguage import LanguageOrderCategory, LanguageCategory, LanguageFamily
from base.config.ProgrammingLanguage import *


class Language:
    def __init__(self):
        self.TokenRepeat_SIZE = 8
        self.PairDepth_SIZE = 8  # 1 sign bit, 7 bits for depth, default value is 127
        self.VirtualEmbedding_SIZE = 8
        # Virtual, Beacon, BeaconKey, BeaconValue
        # Open, Close, OpenClose, reserved
        self.Reserved_SIZE = 8 + 96 + 128

        self.PartOfSpeech_SIZE = PartOfSpeech.SIZE.value
        assert self.PartOfSpeech_SIZE == 32
        self.CommonLanguageType_Size = CommonLanguageType.SIZE.value
        assert self.CommonLanguageType_Size == 16
        self.DataTextType_Size = DataTextType.SIZE.value
        assert self.DataTextType_Size == 8
        self.LanguageOrderCategory_Size = LanguageOrderCategory.SIZE.value
        assert self.LanguageOrderCategory_Size == 8
        self.LanguageCategory_Size = LanguageCategory.SIZE.value
        assert self.LanguageCategory_Size == 4
        self.LanguageFamily_Size = LanguageFamily.SIZE.value
        assert self.LanguageFamily_Size == 16
        self.ProgrammingLanguageType_Size = ProgrammingLanguageType.SIZE.value
        assert self.ProgrammingLanguageType_Size == 8
        self.ProgrammingLanguageSyntax_Size = ProgrammingLanguageSyntax.SIZE.value
        assert self.ProgrammingLanguageSyntax_Size == 8
        self.ProgrammingLanguageLevel_Size = ProgrammingLanguageLevel.SIZE.value
        assert self.ProgrammingLanguageLevel_Size == 4
        self.ProgrammingLanguageExecution_Size = ProgrammingLanguageExecution.SIZE.value
        assert self.ProgrammingLanguageExecution_Size == 4
        self.ProgrammingLanguageTyping_Size = ProgrammingLanguageTyping.SIZE.value
        assert self.ProgrammingLanguageTyping_Size == 4
        self.ProgrammingLanguageMemoryManagement_Size = ProgrammingLanguageMemoryManagement.SIZE.value
        assert self.ProgrammingLanguageMemoryManagement_Size == 4
        self.ProgrammingLanguageParadigm_Size = ProgrammingLanguageParadigm.SIZE.value
        assert self.ProgrammingLanguageParadigm_Size == 8
        self.ProgrammingLanguageConcurrency_Size = ProgrammingLanguageConcurrency.SIZE.value
        assert self.ProgrammingLanguageConcurrency_Size == 4

        self._TOTAL_SIZE = (self.PartOfSpeech_SIZE +
                            self.CommonLanguageType_Size + self.DataTextType_Size + self.LanguageOrderCategory_Size +
                            self.LanguageCategory_Size + self.LanguageFamily_Size + self.ProgrammingLanguageType_Size +
                            self.ProgrammingLanguageSyntax_Size + self.ProgrammingLanguageLevel_Size +
                            self.ProgrammingLanguageExecution_Size + self.ProgrammingLanguageTyping_Size +
                            self.ProgrammingLanguageMemoryManagement_Size + self.ProgrammingLanguageParadigm_Size +
                            self.ProgrammingLanguageConcurrency_Size)

        assert self._TOTAL_SIZE == 128

        self.TOTAL_SIZE = 128 + 128 + 128
        assert self.TOTAL_SIZE == (self._TOTAL_SIZE + self.TokenRepeat_SIZE + self.PairDepth_SIZE
                                   + self.VirtualEmbedding_SIZE + self.Reserved_SIZE)

