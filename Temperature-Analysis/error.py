class Error(Exception):
    pass

class RedFlagDictKeyNotFound(Error):
    def __init__(self, message="Key for the Red Flag Dictionary is not found."):
        self.message = message
        super().__init__(self.message)


class PathError(Error):
    def __init__(self, message="Check file path"):
        self.message = message
        super().__init__(self.message)

class DataFrameEmpty(Error):
     def __init__(self, message="Data Frame is empty\nMake sure you entered correct start and finish dates"):
        self.message = message
        super().__init__(self.message)

class RunTypeError(Error):
    def __init__(self, message="Please make sure you entered correct run type"):
        self.message = message
        super().__init__(self.message)

class CodeError(Error):
    def __init__(self, message="Ask admin for help."):
        self.message = message
        super().__init__(self.message)