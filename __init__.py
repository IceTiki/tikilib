"""
Tiki的常用函数库
"""
import types as _types
import typing as _typing


class LazyImport:
    def __init__(
        self,
        module_name: str,
        fromlist: list[str] = tuple(),
        assigned_model: None | _types.ModuleType = None,
    ):
        self.__module_name: str = module_name
        self.__module: None | _types.ModuleType = assigned_model
        self.__fromlist = fromlist

    def __getattr__(self, name: str):
        if self.__module is None:
            self.__module = __import__(self.__module_name, fromlist=self.__fromlist)
        if name in self.__fromlist:
            cls = self.__class__
            ins: _typing.Self = cls(name, assigned_model=getattr(self.__module, name))
            return ins
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
