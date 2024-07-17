
# enum for log levels
from enum import Enum
from functools import total_ordering
@total_ordering
class LogLevel(Enum):
    VERBOSE = 1
    DEBUG = 2
    INFO = 3
    NOTICE = 4
    WARNING = 5
    ERROR = 6
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented
class Logger:
    instance = None

    def __new__(cls):
        """ creates a singleton object, if it is not created,
        or else returns the previous singleton object"""
        if cls.instance is None:
            cls.instance = super(Logger, cls).__new__(cls)
            cls.name = "Logger"
            cls.instance.level = LogLevel.WARNING
        return cls.instance

    @classmethod
    def get_instance(cls, level=LogLevel.WARNING):
        if cls.instance is None:
            cls.instance = Logger()
            cls.instance.level = level
        return cls.instance

    # takes multiple parameters
    def verbose(self, *args):
        if self.level <= LogLevel.VERBOSE:
            print(f"{self.name}: ", *args)

    def info(self,*args):
        if self.level <= LogLevel.INFO:
            print(f"{self.name}: ", *args)

    def notice(self, *args):
        if self.level <= LogLevel.NOTICE:
            print(f"{self.name}: ", *args)

    def warning(self, *args):
        if self.level <= LogLevel.WARNING:
            print(f"{self.name}: ", *args)

    def debug(self, *args):
        if self.level <= LogLevel.DEBUG:
            print(f"{self.name}: ", *args)

    def shape(self, name, tensor, expected_shape):
        if self.level <= LogLevel.DEBUG:
            assert tensor.shape == expected_shape, f"Shape mismatch for {name}. Expected {expected_shape}, got {tensor.shape}"
            print(f"Shape {self.name}: ", tensor.shape)

    def error(self, *args):
        if self.level <= LogLevel.ERROR:
            print(f"{self.name}: ", *args)
