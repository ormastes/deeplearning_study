
# Tree-like data representation formats are structured in a way that allows data to be nested and organized hierarchically, making them both machine-readable and, to varying extents, human-readable. JSON, XML, and TOML are widely known examples, but there are several other formats that can also be considered. Here’s a grouping of some notable human-readable, tree-like data representation formats:
#
# 1. Key-Value Pair Formats
# Characteristics: These formats represent data as key-value pairs, often allowing for hierarchical structures through nested objects or arrays.
# Examples:
# JSON (JavaScript Object Notation): Simple and widely used, especially in web APIs. Uses curly braces {} for objects and square brackets [] for arrays.
# TOML (Tom's Obvious, Minimal Language): Designed to be easy to read due to its clear syntax, it’s often used in configuration files.
# YAML (YAML Ain't Markup Language): Known for its readability, it uses indentation to define structure and supports complex data types.
# HCL (HashiCorp Configuration Language): Used in tools like Terraform, it’s a configuration language that is easy to read and write.
# 2. Markup Languages
# Characteristics: These formats use tags or other delimiters to define the structure and presentation of the data. They often include metadata about the content.
# Examples:
# XML (eXtensible Markup Language): Highly flexible and self-descriptive, often used for data interchange and document storage.
# HTML (Hypertext Markup Language): Used primarily for web pages, HTML defines the structure of content on the web.
# SGML (Standard Generalized Markup Language): The parent language of XML and HTML, it’s a standard for defining markup languages.
# Markdown: While primarily used for formatting text, Markdown can represent tree-like structures using headers, lists, and links.
# 3. Declarative Configuration Formats
# Characteristics: Designed specifically for configuration management, these formats are often simple and easy to read, with clear delineation of sections and parameters.
# Examples:
# INI: A simple, informal standard for configuration files, with sections and key-value pairs.
# Properties Files: Used in Java and other environments for configuration, similar to INI but with a slightly different syntax.
# HOCON (Human-Optimized Config Object Notation): A superset of JSON designed for human-friendly configuration files.
# TOML: Also fits here due to its simplicity and use in configuration.
# 4. S-expression Formats
# Characteristics: These formats represent data as nested lists of symbols or values, enclosed in parentheses. They are highly regular and can be both code and data.
# Examples:
# LISP (LISt Processing): While primarily a programming language, LISP's S-expressions can represent tree-like data structures.
# EDN (Extensible Data Notation): A data format similar to JSON but more flexible, often used with Clojure.
# 5. Data Serialization Formats
# Characteristics: These formats are designed to serialize complex data structures into a string format that can be easily transmitted or stored, while still being somewhat human-readable.
# Examples:
# Protocol Buffers (proto3): Primarily designed for performance and efficiency, it can still be somewhat readable, especially when viewed in its text format.
# MessagePack: A binary serialization format that is efficient, though less human-readable. However, it has a JSON-like structure.
# CBOR (Concise Binary Object Representation): Similar to MessagePack but with more data types and extensibility.
# TOML: Also used in data serialization due to its clear, human-readable syntax.
# 6. Hierarchical Data Formats
# Characteristics: These formats explicitly represent data in a hierarchical structure, making them ideal for representing complex relationships between elements.
# Examples:
# YAML: Fits here as well because it naturally represents hierarchies through indentation.
# XML: Also belongs here due to its nested tag structure.
# JSON: Frequently used to represent hierarchical data in web APIs.
# Summary of Notable Human-Readable Tree-Like Formats:
# Key-Value Pair Formats: JSON, TOML, YAML, HCL
# Markup Languages: XML, HTML, SGML, Markdown
# Declarative Configuration Formats: INI, Properties Files, HOCON, TOML
# S-expression Formats: LISP, EDN
# Data Serialization Formats: Protocol Buffers (text format), MessagePack (less readable), CBOR
# Hierarchical Data Formats: YAML, XML, JSON
# These formats all provide different approaches to representing data in a way that is structured, organized, and relatively easy for humans to read and understand. The choice of format often depends on the specific use case, such as data interchange, configuration, or serialization.

from enum import Enum
class  DataTextType(Enum):
    KEY_VALUE_PAIR_FORMATS = 0
    MARKUP_LANGUAGES = 1
    DECLARATIVE_CONFIGURATION_FORMATS = 2
    S_EXPRESSION_FORMATS = 3
    DATA_SERIALIZATION_FORMATS = 4
    HIERARCHICAL_DATA_FORMATS = 5
    PLAIN_TEXT = 6
    SIZE=8

class DataTextLanguage():
    name: str
    data_type: set[DataTextType]
    def __init__(self, name, data_type):
        self.name = name
        self.data_type = data_type

JSON = DataTextLanguage("JSON", [DataTextType.KEY_VALUE_PAIR_FORMATS, DataTextType.HIERARCHICAL_DATA_FORMATS])
TOML = DataTextLanguage("TOML", [DataTextType.KEY_VALUE_PAIR_FORMATS, DataTextType.DECLARATIVE_CONFIGURATION_FORMATS])
YAML = DataTextLanguage("YAML", [DataTextType.KEY_VALUE_PAIR_FORMATS, DataTextType.HIERARCHICAL_DATA_FORMATS])
HCL = DataTextLanguage("HCL", [DataTextType.KEY_VALUE_PAIR_FORMATS, DataTextType.DECLARATIVE_CONFIGURATION_FORMATS])
XML = DataTextLanguage("XML", [DataTextType.MARKUP_LANGUAGES, DataTextType.HIERARCHICAL_DATA_FORMATS])
HTML = DataTextLanguage("HTML", [DataTextType.MARKUP_LANGUAGES, DataTextType.HIERARCHICAL_DATA_FORMATS])
SGML = DataTextLanguage("SGML", [DataTextType.MARKUP_LANGUAGES, DataTextType.HIERARCHICAL_DATA_FORMATS])
Markdown = DataTextLanguage("Markdown", [DataTextType.MARKUP_LANGUAGES, DataTextType.PLAIN_TEXT])
INI = DataTextLanguage("INI", [DataTextType.DECLARATIVE_CONFIGURATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
Properties_Files = DataTextLanguage("Properties Files", [DataTextType.DECLARATIVE_CONFIGURATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
HOCON = DataTextLanguage("HOCON", [DataTextType.DECLARATIVE_CONFIGURATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
LISP = DataTextLanguage("LISP", [DataTextType.S_EXPRESSION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
EDN = DataTextLanguage("EDN", [DataTextType.S_EXPRESSION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
Protocol_Buffers = DataTextLanguage("Protocol Buffers", [DataTextType.DATA_SERIALIZATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
MessagePack = DataTextLanguage("MessagePack", [DataTextType.DATA_SERIALIZATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
CBOR = DataTextLanguage("CBOR", [DataTextType.DATA_SERIALIZATION_FORMATS, DataTextType.KEY_VALUE_PAIR_FORMATS])
PLAIN_TEXT = DataTextLanguage("PLAIN_TEXT", [DataTextType.PLAIN_TEXT])

data_text_languages = [JSON, TOML, YAML, HCL, XML, HTML, SGML, Markdown, INI, Properties_Files, HOCON, LISP, EDN, Protocol_Buffers, MessagePack, CBOR, PLAIN_TEXT]
