"""
Tiki的常用函数库
"""


class LazyImport:
    def __init__(self, module_name: str):
        self.__module_name: str = module_name
        self.__module = None

    def __getattr__(self, name: str):
        if self.__module is None:
            self.__module = __import__(self.__module_name)
        return getattr(self.__module, name)
