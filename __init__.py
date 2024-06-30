"""
Tiki的常用函数库
"""
import typing as _typing
import types as _types
import re as _re
import dataclasses as _dataclasses
import importlib as _importlib


class BatchLazyImport:
    # TODO: dose not support "import xxx.yyy as z"
    VAR_PATT = r"[a-zA-Z_]\w*"
    ATTR_PATT = f"(?:{VAR_PATT})(?:\.(?:{VAR_PATT}))*"
    BASE_PATT = r" *(?:(?:from) +(?P<from>[\w\.]*) +)?import +(?P<imp>[\w, ]*\w) *(?:#(?P<comment>.*))?"

    class LazyImportModel:
        def __init__(self, model_name, import_now: bool = False) -> None:
            """
            Using `importlib.import_module` to import `model_name`, which should be directing to a model.
            """
            self.__model_name = model_name
            self.__model_obj = None
            if import_now:
                self.__import_model()

        def __import_model(self):
            self.__model_obj = _importlib.import_module(self.__model_name)
            if not isinstance(self.__model_obj, _types.ModuleType):
                raise ImportError(
                    f"{self.__model_name} is not a model, but {type(self.__model_name)}"
                )

        def __getattr__(self, __name: str) -> _typing.Any:
            if self.__model_obj is None:
                self.__import_model()

            return getattr(self.__model_obj, __name)

    @_dataclasses.dataclass
    class BaseMatch:
        from_info: _typing.List = _dataclasses.field(default_factory=list)
        import_info: _typing.List[_typing.Tuple[str, str]] = _dataclasses.field(
            default_factory=list
        )
        comment: str = ""

    @classmethod
    def base_match(cls, line: str) -> _typing.Union[BaseMatch, None]:
        """
        Examples
        ---
        >>> base_match("import os as _os")
        <<< LazyImport.BaseMatch(from_info=[], import_info=[('os', '_os')], comment='')
        >>> base_match("import os as _osfrom Crypto.Cipher import AES as _AES  # pycryptodome")
        <<< LazyImport.BaseMatch(from_info=['Crypto', 'Cipher'], import_info=[('AES', '_AES')], comment=' pycryptodome')

        Notes
        ---
        If a line dosen't import sentence, then return `None`.
        """
        result = cls.BaseMatch()
        base_match: _re.Match = _re.match(cls.BASE_PATT, line)
        if base_match is None:
            return None

        from_str: _typing.Union[str, None] = base_match.group("from")
        imp_str: _typing.Union[str, None] = base_match.group("imp")
        result.comment = (
            base_match.group("comment")
            if base_match.group("comment") is not None
            else ""
        )

        # from
        if from_str:
            if not _re.match(cls.ATTR_PATT, from_str):
                return None
            result.from_info = from_str.split(".")
        else:
            result.from_info = []
        # imp_str
        if imp_str is None:
            return None
        imp_str: str  # should be like "xx as xx, yy, zz as zz"
        imp_split_patt: str = r"([a-zA-Z_]\w*(?: +as +[a-zA-Z_]\w*)?)(?:, *([a-zA-Z_]\w*(?: +as +[a-zA-Z_]\w*)?))*"
        imp_split_match: _re.Match = _re.match(imp_split_patt, imp_str)
        if imp_split_match is None:
            return None

        result.import_info = []
        imp_alias_patt = r"(?P<model>[a-zA-Z_]\w*)(?: +as +(?P<alias>[a-zA-Z_]\w*))?"
        for i in imp_split_match.groups():
            if i is None:
                continue
            i: str
            alias_match = _re.match(imp_alias_patt, i)
            if alias_match.group("alias") is None:
                imp_info_one = alias_match.group("model"), alias_match.group("model")
            else:
                imp_info_one = (alias_match.group("model"), alias_match.group("alias"))
            result.import_info.append(imp_info_one)

        return result

    def __init__(
        self,
        globals: _typing.Dict[str, _typing.Any],
        locals: _typing.Dict[str, _typing.Any],
        import_string: str,
    ) -> None:
        """
        Parameters
        ---
        globals : dict[str, Any]
            globals()
        locals : dict[str, Any]
            locals()
        import_string : str
            Likes
            '''
            from xxx.yyy import zzz as z, aaa as a, b
            import zzz as z
            '''
        """
        not_lazy_flag: bool = _typing.TYPE_CHECKING
        for line in import_string.splitlines():
            base_match = self.base_match(line)
            if base_match is None:
                continue

            for model, alias in base_match.import_info:
                model_path = base_match.from_info.copy()
                model_path.append(model)
                lazy_model = self.LazyImportModel(
                    ".".join(model_path), import_now=not_lazy_flag
                )
                globals[alias] = lazy_model
                locals[alias] = lazy_model
