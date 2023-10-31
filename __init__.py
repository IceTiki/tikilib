"""
Tiki的常用函数库
"""


class LazyImport:
    def __init__(self, module_name: str, fromlist: list[str] = tuple()):
        self.__module_name: str = module_name
        self.__module = None
        self.__fromlist = fromlist

    def __getattr__(self, name: str):
        if self.__module is None:
            self.__module = __import__(self.__module_name, fromlist=self.__fromlist)
        return getattr(self.__module, name)


def _tmp_replace_import(a: str):
    import re

    def repl(a: re.Match):
        n, m = a.group("n"), a.group("m")
        return f'__globals["{n}"] = LazyImport("{m}")'

    print('if __name__ == "__main__":')
    new = re.sub(r"import (?P<m>\w*) as (?P<n>[\S]*)", repl, a)
    for i in a.splitlines():
        print("    " + i)
    print(
        """else:
    from . import LazyImport

    __globals = globals()"""
    )
    for i in new.splitlines():
        print("    " + i)
