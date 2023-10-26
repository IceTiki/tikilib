if __name__ == "__main__":
    # 标准库
    import json as _json
    import random as _random
    import itertools as _itertools
    import typing as _typing

    # 第三方库
    import yaml as _yaml  # pyyaml
else:
    from . import LazyImport

    # 标准库
    _json = LazyImport("_json")
    _random = LazyImport("random")
    _itertools = LazyImport("itertools")
    _typing = LazyImport("typing")

    # 第三方库
    _yaml = LazyImport("yaml")  # pyyaml


class YamlFile:
    @staticmethod
    def load(yml_path="data.yml", encoding="utf-8"):
        """读取Yaml文件"""
        with open(yml_path, "r", encoding=encoding) as f:
            return _yaml.load(f, Loader=_yaml.FullLoader)

    @staticmethod
    def write(item, yml_path="data.yml", encoding="utf-8"):
        """写入Yaml文件"""
        with open(yml_path, "w", encoding=encoding) as f:
            _yaml.dump(item, f, allow_unicode=True)


class JsonFile:
    @staticmethod
    def load(json_path="data.json", encoding="utf-8"):
        """读取Json文件"""
        with open(json_path, "r", encoding=encoding) as f:
            return _json.load(f)

    @staticmethod
    def write(item, json_path="data.json", encoding="utf-8", ensure_ascii=False):
        """写入Json文件"""
        with open(json_path, "w", encoding=encoding) as f:
            _json.dump(item, f, ensure_ascii=ensure_ascii)


class Random:
    hex_chr = "0123456789ABCDEF"
    word_chr = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @staticmethod
    def random_hex(chara: str = hex_chr, len_: int = 8):
        return "".join(_random.choices(chara, k=len_))

    @staticmethod
    def random_word(chara: str = word_chr, len_: int = 8):
        return "".join(_random.choices(chara, k=len_))


def _test_gene_markdown_table(
    dataframe: _typing.Dict[_typing.Any, _typing.Sequence],
    key_arr: _typing.Sequence = None,
    head_arr: _typing.Sequence = None,
    fillvalue: str = "",
):
    """

    Parameters
    ---
    dataframe : typing.Dict
        数据表, 其值均为序列
    key_arr : typing.Sequence, default = dataframe.keys()
        在数据表中取值的键序列
    head_arr : typing.Sequence, default = dataframe.keys()
        表头序列
    fillvalue : str
        较短序列的填充值
    """
    if key_arr is None:
        key_arr, value_arr = zip(*dataframe.items())
    else:
        value_arr = map(lambda x: dataframe[x], key_arr)
    if head_arr is None:
        head_arr = key_arr

    text = "|" + "|".join(map(str, head_arr)) + "|" + "\n"
    text += "|" + "|".join(map(lambda x: ":-:", range(len(head_arr)))) + "|" + "\n"
    for values in _itertools.zip_longest(*value_arr, fillvalue=fillvalue):
        text += "|" + "|".join(map(str, values)) + "|" + "\n"
    return text
