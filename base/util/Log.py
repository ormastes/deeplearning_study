
# enum for log levels
from enum import Enum
from functools import total_ordering
@total_ordering
class LogLevel(Enum):
    VERBOSE = 1
    INFO = 2
    NOTICE = 3
    WARNING = 4
    DEBUG = 5
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
            cls.instance.level = LogLevel.NOTICE
        return cls.instance

    @classmethod
    def get_instance(cls):
        if cls.instance is None:
            cls.instance = Logger()
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

    def error(self, *args):
        if self.level <= LogLevel.ERROR:
            print(f"{self.name}: ", *args)
