
#  These languages are often referred to as formal languages or logical languages. Here are a few examples:
#
# 1. Mathematical Notation
# Description: While not a programming language per se, mathematical notation is a highly formalized system of symbols and rules used to represent mathematical concepts and operations. It is an artificial language created to concisely and precisely convey mathematical ideas.
# Usage: Used universally in mathematics, physics, engineering, and other sciences.
# 2. Propositional and Predicate Logic
# Description: These are formal languages used in logic, mathematics, and computer science to express statements that can be true or false. They use symbols to represent logical operations like AND, OR, NOT, and quantifiers like "for all" (∀) and "there exists" (∃).
# Usage: Foundational in fields such as formal verification, artificial intelligence, and theoretical computer science.
# 3. Zermelo–Fraenkel Set Theory (ZFC)
# Description: A formal language used in mathematics to describe set theory, which forms the foundation of much of modern mathematics. It consists of axioms that define the properties and operations of sets.
# Usage: Used by mathematicians to work with and define concepts related to sets, functions, and relations.
# 4. Lambda Calculus
# Description: A formal system in mathematical logic and computer science for expressing computation based on function abstraction and application. Lambda calculus forms the foundation of functional programming languages.
# Usage: Theoretical computer science, functional programming languages (e.g., Haskell, Lisp).
# 5. Formal Grammars (e.g., Backus-Naur Form)
# Description: These are formal systems used to describe the syntax of programming languages and other formal languages. They specify the rules for generating valid strings in a language.
# Usage: Used in compiler design, parsing, and defining programming languages.
# 6. Category Theory
# Description: A branch of mathematics that deals with abstract structures and relationships between them. It uses a formal language that is very general and can describe many different kinds of mathematical structures in a unified way.
# Usage: Foundational in some areas of theoretical computer science, particularly in functional programming and type theory.
# 7. Constructed Languages (Conlangs) for Specific Domains
# Description: These are artificial languages created for specific purposes, such as Lojban, a constructed language designed to reflect the principles of logic, or Loglan, which was designed to test the Sapir-Whorf hypothesis about language and thought.
# Usage: Used in experiments, language studies, and sometimes in specific subcultures or communities interested in linguistic precision.
# 8. Formal Specification Languages
# Description: These languages are used to formally specify the behavior of systems, particularly in software and hardware design. Examples include Z notation and VDM (Vienna Development Method).
# Usage: Used in software engineering to rigorously define system specifications and verify correctness.
# Summary
# These artificial languages share similarities with mathematics in their precision, formality, and ability to represent complex ideas unambiguously. They are often used in areas where exactness and clarity are paramount, such as in logic, mathematics, theoretical computer science, and formal methods in software engineering. Unlike natural languages, which evolve organically, these artificial languages are deliberately designed to fulfill specific purposes, often within scientific, mathematical, or logical contexts.

from enum import Enum


class CommonLanguageType(Enum):
    MATHEMATICAL_NOTATION = 0
    PROPOSITIONAL_AND_PREDICATE_LOGIC = 1
    ZERMELO_FRAENKEL_SET_THEORY = 2
    LAMBDA_CALCULUS = 3
    FORMAL_GRAMMARS = 4
    CATEGORY_THEORY = 5
    CONSTRUCTED_LANGUAGES = 6
    FORMAL_SPECIFICATION_LANGUAGES = 7
    HUMAN_LANGUAGE = 8
    PROGRAMMING_LANGUAGE = 9
    SIZE=16

# { 'adjective', 'satellite', 'noun', 'verb', 'adverb'
# 'program_assign', 'program_comment_end', 'program_statement_end', 'program_block_start', 'program_comment_start',  'program_open', 'program_connector',  'program_block_end', 'program_close', 'program_symbols', 'program_arithmetics'}
class PartOfSpeech(Enum):
    ADJECTIVE = 0
    SATELLITE = 1
    NOUN = 2
    VERB = 3
    ADVERB = 4
    PROGRAM_ASSIGN = 5
    PROGRAM_COMMENT_END = 6
    PROGRAM_STATEMENT_END = 7
    PROGRAM_BLOCK_START = 8
    PROGRAM_COMMENT_START = 9
    PROGRAM_OPEN = 10
    PROGRAM_CONNECTOR = 11
    PROGRAM_BLOCK_END = 12
    PROGRAM_CLOSE = 13
    PROGRAM_SYMBOLS = 14
    PROGRAM_ARITHMETICS = 15
    SIZE=32