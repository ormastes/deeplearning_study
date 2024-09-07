

from enum import Enum

# 1. SVO (Subject-Verb-Object) Languages:
# Characteristics: The typical sentence structure is Subject-Verb-Object.
# Examples: English, Mandarin Chinese, Spanish, French.
# 2. SOV (Subject-Object-Verb) Languages:
# Characteristics: The sentence structure is Subject-Object-Verb.
# Examples: Japanese, Korean, Turkish, Hindi.
# 3. VSO (Verb-Subject-Object) Languages:
# Characteristics: The sentence structure is Verb-Subject-Object.
# Examples: Arabic, Classical Hebrew, Irish.
# 4. VOS (Verb-Object-Subject) Languages:
# Characteristics: The sentence structure is Verb-Object-Subject.
# Examples: Malagasy, Fijian.
# 5. OSV (Object-Subject-Verb) and OVS (Object-Verb-Subject) Languages:
# Characteristics: These are relatively rare and have specific structures where the object comes first.
# Examples: Some Amazonian languages, like Hixkaryana.
# 6. Free Word Order Languages:
# Characteristics: Word order is more flexible, and meaning is often conveyed through inflection or context.
# Examples: Russian, Latin.
# 7. Agglutinative Languages:
# Characteristics: Words are formed by stringing together morphemes, each representing a single grammatical category.
# Examples: Finnish, Hungarian, Turkish.
# 8. Fusional (Inflectional) Languages:
# Characteristics: Words can change forms to express different grammatical categories through inflection.
# Examples: Russian, Latin, Spanish.
# 9. Isolating (Analytic) Languages:
# Characteristics: Words typically do not change form, and grammatical relations are indicated by word order and auxiliary words.
# Examples: Chinese, Vietnamese.
# 10. Polysynthetic Languages:
# Characteristics: Words are often very complex, composed of many morphemes that together express what would be a whole sentence in other languages.
# Examples: Inuktitut, Mohawk.


class LanguageOrderCategory(Enum):
    SVO = 0
    SOV = 1
    VSO = 2
    VOS = 3
    OSV = 4
    OVS = 5
    FreeWordOrder = 6
    reserved = 7
    SIZE = 8


class LanguageCategory(Enum):
    Agglutinative = 0
    Fusional = 1
    Isolating = 2
    Polysynthetic = 3

    SIZE = 4

# language families
# Indo-European Family:
#
# Germanic (e.g., English, German, Dutch)
# Romance (e.g., Spanish, French, Italian)
# Slavic (e.g., Russian, Polish, Czech)
# Indo-Iranian (e.g., Hindi, Persian)
# Sino-Tibetan Family:
#
# Sinitic (e.g., Mandarin, Cantonese)
# Tibeto-Burman (e.g., Burmese, Tibetan)
# Afro-Asiatic Family:
#
# Semitic (e.g., Arabic, Hebrew, Amharic)
# Berber (e.g., Tuareg, Kabyle)
# Cushitic (e.g., Somali, Oromo)
# Niger-Congo Family:
#
# Bantu (e.g., Swahili, Zulu)
# Kwa (e.g., Akan, Yoruba)
# Dravidian Family:
#
# Languages primarily in southern India (e.g., Tamil, Telugu)
# Altaic Hypothesis (disputed):
#
# Turkic (e.g., Turkish, Uzbek)
# Mongolic (e.g., Mongolian)
# Tungusic (e.g., Manchu)
# Uralic Family:
#
# Finno-Ugric (e.g., Finnish, Hungarian)
# Austronesian Family:
#
# Malayo-Polynesian (e.g., Tagalog, Malagasy)
# Austroasiatic Family:
#
# Mon-Khmer (e.g., Khmer, Vietnamese)
# Munda (e.g., Santali)
# Japanese-Ryukyuan:
#
# Includes Japanese and the Ryukyuan languages.
# Koreanic:
#
# Includes Korean and its dialects.
# Isolate Languages:
#
# Languages that do not belong to any known family (e.g., Basque, Ainu)
# Native American Language Families:
#
# Numerous distinct families and isolates, like Na-Dené (e.g., Navajo), Algonquian (e.g., Cree), and Mayan (e.g., Yucatec Maya).
# Papuan Languages:
#
# Various language families in New Guinea with significant grammatical diversity.
# Australian Aboriginal Languages:
#
# Numerous language families, each with its unique grammatical structures.

class  LanguageFamily(Enum):
    IndoEuropean = 1
    SinoTibetan = 2
    AfroAsiatic = 3
    NigerCongo = 4
    Dravidian = 5
    Altaic = 6
    Uralic = 7
    Austronesian = 8
    Austroasiatic = 9
    JapaneseRyukyuan = 10
    Koreanic = 11
    Isolate = 12
    NativeAmerican = 13
    Papuan = 14
    AustralianAboriginal = 15
    SIZE = 16

# Country and Language Categories
class Country:
    name : str
    language : str
    order_category : LanguageOrderCategory # size 8
    category : set[LanguageCategory] # size 4
    family : LanguageFamily # size 16
    def __init__(self, name, language, order_category, category=None, family=None):
        self.name = name
        self.language = language
        self.order_category = order_category
        self.category = set(category) if category else set()
        self.family = family

# United States: English (SVO, Fusional/Isolating)
# China: Mandarin Chinese (SVO, Isolating)
# Japan: Japanese (SOV, Agglutinative)
# Germany: German (SVO, Fusional)
# India: Hindi (SOV, Fusional)
# United Kingdom: English (SVO, Fusional/Isolating)
# France: French (SVO, Fusional)
# Italy: Italian (SVO, Fusional)
# Canada: English (SVO, Fusional/Isolating), French (SVO, Fusional)
# South Korea: Korean (SOV, Agglutinative)
# Russia: Russian (SVO, Fusional)
# Brazil: Portuguese (SVO, Fusional)
# Australia: English (SVO, Fusional/Isolating)
# Spain: Spanish (SVO, Fusional)
# Mexico: Spanish (SVO, Fusional)
# Indonesia: Indonesian (SVO, Isolating)
# Netherlands: Dutch (SVO, Fusional)
# Saudi Arabia: Arabic (VSO, Fusional)
# Turkey: Turkish (SOV, Agglutinative)
# Switzerland: German (SVO, Fusional)
# Sweden: Swedish (SVO, Fusional)
# Poland: Polish (SVO, Fusional)
# Belgium: Dutch (SVO, Fusional)
# Thailand: Thai (SVO, Isolating)
# Austria: German (SVO, Fusional)
# Greece: Greek (SVO, Fusional)
# Norway: Norwegian (SVO, Fusional)
# Denmark: Danish (SVO, Fusional)
# Finland: Finnish (SVO, Agglutinative)
# Egypt: Arabic (VSO, Fusional)
# Czech Republic: Czech (SVO, Fusional)
# Portugal: Portuguese (SVO, Fusional)

UnitStates = Country("United States", "English", LanguageOrderCategory.SVO,
                     [LanguageCategory.Fusional, LanguageCategory.Isolating], LanguageFamily.IndoEuropean)
China = Country("China", "Mandarin Chinese", LanguageOrderCategory.SVO, [LanguageCategory.Isolating], LanguageFamily.SinoTibetan)
Japan = Country("Japan", "Japanese", LanguageOrderCategory.SOV, [LanguageCategory.Agglutinative], LanguageFamily.JapaneseRyukyuan)
Germany = Country("Germany", "German", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
India = Country("India", "Hindi", LanguageOrderCategory.SOV, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
#UnitedKingdom = Country("United Kingdom", "English", LanguageOrderCategory.SVO, [LanguageCategory.Fusional, LanguageCategory.Isolating], LanguageFamily.IndoEuropean)
France = Country("France", "French", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Italy = Country("Italy", "Italian", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
#Canada = Country("Canada", "English", LanguageOrderCategory.SVO, [LanguageCategory.Fusional, LanguageCategory.Isolating], LanguageFamily.IndoEuropean)
Korea = Country("South Korea", "Korean", LanguageOrderCategory.SOV, [LanguageCategory.Agglutinative], LanguageFamily.Koreanic)
Russia = Country("Russia", "Russian", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Brazil = Country("Brazil", "Portuguese", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
#Australia = Country("Australia", "English", LanguageOrderCategory.SVO, [LanguageCategory.Fusional, LanguageCategory.Isolating], LanguageFamily.IndoEuropean)
Spain = Country("Spain", "Spanish", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
#Mexico = Country("Mexico", "Spanish", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Indonesia = Country("Indonesia", "Indonesian", LanguageOrderCategory.SVO, [LanguageCategory.Isolating], LanguageFamily.Austronesian)
Netherlands = Country("Netherlands", "Dutch", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
SaudiArabia = Country("Saudi Arabia", "Arabic", LanguageOrderCategory.VSO, [LanguageCategory.Fusional], LanguageFamily.AfroAsiatic)
Turkey = Country("Turkey", "Turkish", LanguageOrderCategory.SOV, [LanguageCategory.Agglutinative], LanguageFamily.Altaic)
Switzerland = Country("Switzerland", "German", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Sweden = Country("Sweden", "Swedish", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Poland = Country("Poland", "Polish", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Belgium = Country("Belgium", "Dutch", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Thailand = Country("Thailand", "Thai", LanguageOrderCategory.SVO, [LanguageCategory.Isolating], LanguageFamily.SinoTibetan)
#Austria = Country("Austria", "German", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Greece = Country("Greece", "Greek", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Norway = Country("Norway", "Norwegian", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Denmark = Country("Denmark", "Danish", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Finland = Country("Finland", "Finnish", LanguageOrderCategory.SVO, [LanguageCategory.Agglutinative], LanguageFamily.Uralic)
Egypt = Country("Egypt", "Arabic", LanguageOrderCategory.VSO, [LanguageCategory.Fusional], LanguageFamily.AfroAsiatic)
CzechRepublic = Country("Czech Republic", "Czech", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)
Portugal = Country("Portugal", "Portuguese", LanguageOrderCategory.SVO, [LanguageCategory.Fusional], LanguageFamily.IndoEuropean)

LanguageSet = {
    UnitStates.language: UnitStates,
    China.language: China,
    Japan.language: Japan,
    Germany.language: Germany,
    India.language: India,
    #UnitedKingdom.language: UnitedKingdom,
    France.language: France,
    Italy.language: Italy,
    #Canada.language: Canada,
    Korea.language: Korea,
    Russia.language: Russia,
    Brazil.language: Brazil,
    #Australia.language: Australia,
    Spain.language: Spain,
    #Mexico.language: Mexico,
    Indonesia.language: Indonesia,
    Netherlands.language: Netherlands,
    SaudiArabia.language: SaudiArabia,
    Turkey.language: Turkey,
    Switzerland.language: Switzerland,
    Sweden.language: Sweden,
    Poland.language: Poland,
    Belgium.language: Belgium,
    Thailand.language: Thailand,
    #Austria.language: Austria,
    Greece.language: Greece,
    Norway.language: Norway,
    Denmark.language: Denmark,
    Finland.language: Finland,
    Egypt.language: Egypt,
    CzechRepublic.language: CzechRepublic,
    Portugal.language: Portugal
}
