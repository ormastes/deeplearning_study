# 1. Imperative Languages
# Characteristics: These languages are based on the concept of giving a sequence of commands for the computer to perform.
# Examples:
# C: A foundational language used for system programming and developing operating systems.
# Java: Widely used in enterprise environments, Android development, and web applications.
# C++: An extension of C with object-oriented features, used in system/software development, game development, etc.
# 2. Object-Oriented Languages
# Characteristics: Focuses on objects that contain both data and methods, promoting reuse through inheritance and polymorphism.
# Examples:
# Python: A versatile language used in web development, data science, automation, and more.
# Java: Also fits here due to its strong support for object-oriented principles.
# C++: Although it’s multi-paradigm, it is heavily used for object-oriented programming.
# C#: Used extensively in developing Windows applications, games (with Unity), and enterprise software.
# 3. Functional Languages
# Characteristics: Emphasizes functions as the primary building blocks, avoiding changing states and mutable data.
# Examples:
# JavaScript: While primarily used for web development, it supports functional programming.
# Python: Supports functional programming alongside object-oriented programming.
# Haskell: A purely functional language, often used in academic settings.
# 4. Scripting Languages
# Characteristics: Typically used for writing scripts, automating tasks, or gluing together systems. Often interpreted rather than compiled.
# Examples:
# Python: Commonly used for scripting and automation.
# JavaScript: The primary language for client-side web scripting.
# Ruby: Often used for web development, especially with the Ruby on Rails framework.
# PHP: Widely used for server-side scripting in web development.
# 5. Declarative Languages
# Characteristics: Focuses on describing what the program should accomplish rather than how to do it.
# Examples:
# SQL: Used for querying and managing data in relational databases.
# HTML/CSS: Used to structure and style web pages.
# 6. Low-Level Languages
# Characteristics: Closer to machine code, providing greater control over hardware but requiring more detailed management of resources.
# Examples:
# Assembly Language: Used for programming directly at the hardware level.
# C: Though not as low-level as assembly, C is still closer to the hardware compared to most other languages.
# Top 10 Most Used Programming Languages Grouped by Paradigm/Category:
# JavaScript (Scripting, Functional, Imperative)
# Python (Object-Oriented, Scripting, Functional)
# Java (Object-Oriented, Imperative)
# C# (Object-Oriented, Imperative)
# C++ (Object-Oriented, Imperative)
# PHP (Scripting, Imperative)
# C (Imperative, Low-Level)
# SQL (Declarative)
# TypeScript (Scripting, Object-Oriented, Functional)
# Ruby (Object-Oriented, Scripting)

from enum import Enum
class ProgrammingLanguageType(Enum):
    IMPERATIVE = 0
    OBJECT_ORIENTED = 1
    FUNCTIONAL = 2
    SCRIPTING = 3
    DECLARATIVE = 4
    LOGIC = 5
    preserved0 = 6
    preserved1 = 7
    SIZE = 8

 # 1. C-Like Syntax
# Characteristics: These languages have a syntax style derived from C, which includes the use of curly braces {}, semicolons ;, and similar control structures (if, for, while).
# Examples:
# C
# C++
# Java
# JavaScript
# C#
# Objective-C
# PHP
# Rust
# Go
# Explanation: These languages share a common ancestry in C, and their syntax is often recognizable due to similar structure, operators, and control flow constructs. Programmers familiar with one of these languages can usually read and understand code in another with minimal effort.
#
# 2. Python-Like Syntax
# Characteristics: Known for its simplicity, Python's syntax is characterized by indentation to define blocks, the absence of mandatory semicolons, and a focus on readability.
# Examples:
# Python
# Ruby (though Ruby has a more flexible syntax)
# Julia
# Bash (to some extent, due to its reliance on indentation and scripting style)
# Explanation: These languages prioritize readability and have less syntactic noise compared to C-like languages. They tend to use indentation rather than braces for code blocks, which makes the code visually clean and easier to follow.
#
# 3. Lisp-Like Syntax
# Characteristics: Based on s-expressions (symbolic expressions), these languages use parentheses extensively to denote code structure, with a prefix notation where the operator comes before the operands.
# Examples:
# Lisp
# Scheme
# Clojure
# Racket
# Explanation: Lisp-like languages are known for their minimalist and uniform syntax, where the code is represented as lists (sequences of symbols and other lists). This uniformity makes them very powerful for metaprogramming and symbolic computation.
#
# 4. ML-Like Syntax
# Characteristics: These languages are known for their strong type systems and functional programming roots, often using pattern matching, type inference, and algebraic data types.
# Examples:
# ML (MetaLanguage)
# OCaml
# Haskell (although it introduces significant syntactic sugar compared to ML)
# F# (inspired by ML but with C-like features)
# Explanation: ML and its descendants have a concise and expressive syntax focused on functional programming, with an emphasis on immutability and type safety. Their syntax is quite different from C-like languages, emphasizing pattern matching and expression-oriented code.
#
# 5. Perl-Like Syntax
# Characteristics: Perl’s syntax is known for its flexibility and "there's more than one way to do it" philosophy, featuring extensive use of sigils ($, @, %).
# Examples:
# Perl
# Ruby (to some extent, due to its flexible syntax and influence from Perl)
# PHP (shares some syntax elements with Perl)
# Explanation: Perl and similar languages offer a very flexible syntax that can appear chaotic but allows a lot of power and expressiveness, especially in text processing and scripting tasks.
#
# 6. SQL-Like Syntax
# Characteristics: Declarative and focused on database queries, SQL-like languages emphasize querying and manipulating sets of data using SELECT, INSERT, UPDATE, DELETE statements.
# Examples:
# SQL
# PL/SQL (Procedural Language/SQL)
# T-SQL (Transact-SQL)
# Explanation: These languages are highly specialized for interacting with databases and have a declarative style where you describe what data to retrieve rather than how to retrieve it.
#
# 7. Prolog-Like Syntax
# Characteristics: Logic programming syntax focused on defining relations and querying them, using a series of facts and rules.
# Examples:
# Prolog
# Mercury
# Explanation: Prolog-like languages use a very distinct syntax focused on logical assertions and queries, making them very different from most other language families. They are mainly used in areas like artificial intelligence and computational linguistics.
#
# Summary of the Top 10 Most Used Languages by Grammatical Similarity:
# C-Like Syntax:
#
# C, C++, Java, JavaScript, C#, PHP, Go
# Python-Like Syntax:
#
# Python, Ruby
# Lisp-Like Syntax:
#
# Clojure (not among the top 10 but notable for Lisp-like syntax)
# ML-Like Syntax:
#
# Haskell (significant in functional programming but not as widely used as others in the top 10)

class ProgrammingLanguageSyntax(Enum):
    C_LIKE = 0
    PYTHON_LIKE = 1
    LISP_LIKE = 2
    ML_LIKE = 3
    PERL_LIKE = 4
    SQL_LIKE = 5
    PROLOG_LIKE = 6
    preserved = 7
    SIZE = 8


#  These groupings often reflect the purpose, design philosophy, or target application domain of the languages. Here are some additional ways to categorize programming languages:
#

# 1. By Level of Abstraction
# Low-Level Languages:
#
# Description: Closer to machine code, providing fine-grained control over hardware but requiring detailed management of resources.
# Examples: Assembly, C
# High-Level Languages:
#
# Description: More abstract, allowing programmers to write code that is more distant from the machine’s architecture, focusing on complex operations and ease of use.
# Examples: Python, Java, Ruby
# Mid-Level Languages:
#
# Description: Balances low-level hardware control with high-level features, often used for system programming.
# Examples: C++, Rust

class ProgrammingLanguageLevel(Enum):
    LOW_LEVEL = 0
    HIGH_LEVEL = 1
    MID_LEVEL = 2
    preserved = 3
    SIZE = 4

# 2. By Execution Model
# Compiled Languages:
#
# Description: Translated into machine code by a compiler before execution, often resulting in faster performance.
# Examples: C, C++, Rust, Go
# Interpreted Languages:
#
# Description: Executed line-by-line by an interpreter, offering more flexibility and ease of debugging but potentially slower performance.
# Examples: Python, Ruby, JavaScript
# Just-in-Time (JIT) Compiled Languages:
#
# Description: Compiled into intermediate bytecode, which is then executed or further compiled at runtime for a balance of speed and flexibility.
# Examples: Java, C#, JavaScript (in modern engines like V8)

class ProgrammingLanguageExecution(Enum):
    COMPILED = 0
    INTERPRETED = 1
    JIT_COMPILED = 2
    preserved = 3
    SIZE = 4

# 4. By Typing Discipline
# Static Typing:
#
# Description: Variable types are explicitly declared and checked at compile-time, leading to more predictable and error-resistant code.
# Examples: Java, C, Rust, Haskell
# Dynamic Typing:
#
# Description: Variable types are determined at runtime, offering more flexibility and quicker development but potentially leading to runtime errors.
# Examples: Python, JavaScript, Ruby
# Strong Typing:
#
# Description: Strict enforcement of type rules, where implicit conversions are minimal or not allowed.
# Examples: Haskell, Rust, Java
# Weak Typing:
#
# Description: More lenient type rules, allowing for implicit conversions between types, which can lead to more flexibility but also subtle bugs.
# Examples: JavaScript, PHP

class ProgrammingLanguageTyping(Enum):
    STATIC = 0
    DYNAMIC = 1
    STRONG = 2
    WEAK = 3
    SIZE = 4

# 5. By Memory Management
# Manual Memory Management:
#
# Description: The programmer explicitly allocates and deallocates memory, providing control but requiring careful management to avoid leaks and errors.
# Examples: C, C++
# Automatic Memory Management (Garbage Collection):
#
# Description: The language runtime automatically handles memory allocation and deallocation by garbage collection, reducing the risk of memory leaks.
# Examples: Java, Python, C#
# Reference Counting:
#
# Description: Memory is managed through counting references to objects; when no references remain, the memory is deallocated.
# Examples: Objective-C, Swift
#
# Description: Like Rust, which uses a combination of static analysis and ownership rules to ensure memory safety.
# Examples: Rust

class ProgrammingLanguageMemoryManagement(Enum):
    MANUAL = 0
    GC = 1
    REFERENCE_COUNTING = 2
    STATIC_ANALYSIS = 3
    SIZE = 4

# 6. By Paradigm
# Declarative Languages:
#
# Description: Focus on what should be done, rather than how to do it. Includes languages for querying databases or configuring systems.
# Examples: SQL, HTML, CSS
# Procedural Languages:
#
# Description: Based on the concept of procedures or routines, where code is organized into blocks that execute sequentially.
# Examples: C, Pascal, Fortran
# Event-Driven Languages:
#
# Description: Programming is centered around responding to events, often used in GUI applications.
# Examples: JavaScript, Visual Basic, Swift (for iOS development)
#
# Functional Languages:
#

class ProgrammingLanguageParadigm(Enum):
    DECLARATIVE = 0
    PROCEDURAL = 1
    FUNCTIONAL = 2
    OBJECT_ORIENTED = 3
    GENERIC = 4
    LOGIC = 5
    MULTI_PARADIGM = 6
    CONCURRENT = 7
    SIZE = 8
# 9. By Concurrency Model
# Single-Threaded Languages:
#
# Description: Languages designed to operate in a single thread, where all operations are sequential.
# Examples: JavaScript (though it can handle asynchronous operations with event loops)
# Multi-Threaded Languages:
#
# Description: Languages that support the creation and management of multiple threads, enabling concurrent execution.
# Examples: Java, C#, C++
# Actor Model Languages:
#
# Description: Languages that use the actor model for concurrency, where "actors" are the fundamental units of computation.
# Examples: Erlang, Scala (with Akka)

class ProgrammingLanguageConcurrency(Enum):
    SINGLE_THREADED = 0
    MULTI_THREADED = 1
    ACTOR_MODEL = 2
    EVENT_DRIVEN = 3
    SIZE = 4

class ProgrammingLanguage:
    name: str
    type: set[ProgrammingLanguageType]  # size 8
    syntax: set[ProgrammingLanguageSyntax]  # size 8
    level: ProgrammingLanguageLevel  # size 4
    execution: set[ProgrammingLanguageExecution]  # size 4
    typing: set[ProgrammingLanguageTyping]  # size 4
    memory_management: ProgrammingLanguageMemoryManagement  # size 4
    paradigm: set[ProgrammingLanguageParadigm]  # size 4
    concurrency: set[ProgrammingLanguageConcurrency]  # size 4

    def __init__(self, name: str, type: set[ProgrammingLanguageType], syntax: set[ProgrammingLanguageSyntax], level: ProgrammingLanguageLevel, execution: set[ProgrammingLanguageExecution], typing: set[ProgrammingLanguageTyping], memory_management: ProgrammingLanguageMemoryManagement, paradigm: set[ProgrammingLanguageParadigm], concurrency: set[ProgrammingLanguageConcurrency]):
        self.name = name
        self.type = type
        self.syntax = syntax
        self.level = level
        self.execution = execution
        self.typing = typing
        self.memory_management = memory_management
        self.paradigm = paradigm
        self.concurrency = concurrency


C = ProgrammingLanguage(
    "C",
    {ProgrammingLanguageType.IMPERATIVE},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.LOW_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.PROCEDURAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

CXX = ProgrammingLanguage(
    "C++",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.LOW_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.OBJECT_ORIENTED,

     ProgrammingLanguageParadigm.GENERIC},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Java = ProgrammingLanguage(
    "Java",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

JavaScript = ProgrammingLanguage(
    "JavaScript",
    {ProgrammingLanguageType.IMPERATIVE,  ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.FUNCTIONAL, ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED, ProgrammingLanguageConcurrency.EVENT_DRIVEN}
)

TypeScript = ProgrammingLanguage(
    "TypeScript",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.FUNCTIONAL, ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED, ProgrammingLanguageConcurrency.EVENT_DRIVEN}
)

C_SHARP = ProgrammingLanguage(
    "C#",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Python = ProgrammingLanguage(
    "Python",
    {ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.SCRIPTING, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.PYTHON_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL,
     ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.GENERIC},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Ruby = ProgrammingLanguage(
    "Ruby",
    {ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.PYTHON_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Haskell = ProgrammingLanguage(
    "Haskell",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.ML_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED, ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

SQL = ProgrammingLanguage(
    "SQL",
    {ProgrammingLanguageType.DECLARATIVE},
    {ProgrammingLanguageSyntax.SQL_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.DECLARATIVE},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Rust = ProgrammingLanguage(
    "Rust",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.LOW_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Go = ProgrammingLanguage(
    "Go",
    {ProgrammingLanguageType.IMPERATIVE},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.LOW_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.CONCURRENT},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Objective_C = ProgrammingLanguage(
    "Objective-C",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

PL_SQL = ProgrammingLanguage(
    "PL/SQL",
    {ProgrammingLanguageType.DECLARATIVE},
    {ProgrammingLanguageSyntax.SQL_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.DECLARATIVE},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

T_SQL = ProgrammingLanguage(
    "T-SQL",
    {ProgrammingLanguageType.DECLARATIVE},
    {ProgrammingLanguageSyntax.SQL_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.DECLARATIVE},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Prolog = ProgrammingLanguage(
    "Prolog",
    {ProgrammingLanguageType.DECLARATIVE},
    {ProgrammingLanguageSyntax.PROLOG_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.LOGIC},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Mercury = ProgrammingLanguage(
    "Mercury",
    {ProgrammingLanguageType.DECLARATIVE, ProgrammingLanguageType.LOGIC},
    {ProgrammingLanguageSyntax.PROLOG_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.MANUAL,
    {ProgrammingLanguageParadigm.LOGIC},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

ML = ProgrammingLanguage(
    "ML",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.ML_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

LISP = ProgrammingLanguage(
    "LISP",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Scheme = ProgrammingLanguage(
    "Scheme",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Clojure = ProgrammingLanguage(
    "Clojure",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Racket = ProgrammingLanguage(
    "Racket",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

OCaml = ProgrammingLanguage(
    "OCaml",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.ML_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED, ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Perl = ProgrammingLanguage(
    "Perl",
    {ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.PERL_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.MULTI_PARADIGM},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

PHP = ProgrammingLanguage(
    "PHP",
    {ProgrammingLanguageType.SCRIPTING, ProgrammingLanguageType.IMPERATIVE},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Swift = ProgrammingLanguage(
    "Swift",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.PROCEDURAL, ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Kotlin = ProgrammingLanguage(
    "Kotlin",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Scala = ProgrammingLanguage(
    "Scala",
    {ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Dart = ProgrammingLanguage(
    "Dart",
    {ProgrammingLanguageType.OBJECT_ORIENTED, ProgrammingLanguageType.SCRIPTING, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.C_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.STATIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED, ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Julia = ProgrammingLanguage(
    "Julia",
    {ProgrammingLanguageType.IMPERATIVE, ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.PYTHON_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.COMPILED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.MULTI_THREADED}
)

Bash = ProgrammingLanguage(
    "Bash",
    {ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.PYTHON_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.PROCEDURAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

R = ProgrammingLanguage(
    "R",
    {ProgrammingLanguageType.SCRIPTING},
    {ProgrammingLanguageSyntax.PYTHON_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

Elixir = ProgrammingLanguage(
    "Elixir",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

ERLANG = ProgrammingLanguage(
    "ERLANG",
    {ProgrammingLanguageType.FUNCTIONAL},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.FUNCTIONAL},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)

SMALLTALK = ProgrammingLanguage(
    "SMALLTALK",
    {ProgrammingLanguageType.OBJECT_ORIENTED},
    {ProgrammingLanguageSyntax.LISP_LIKE},
    ProgrammingLanguageLevel.HIGH_LEVEL,
    {ProgrammingLanguageExecution.INTERPRETED},
    {ProgrammingLanguageTyping.DYNAMIC},
    ProgrammingLanguageMemoryManagement.GC,
    {ProgrammingLanguageParadigm.OBJECT_ORIENTED},
    {ProgrammingLanguageConcurrency.SINGLE_THREADED}
)