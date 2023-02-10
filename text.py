# 标准库
import json as _json

# 第三方库
import yaml as _yaml  # pyyaml


class YamlFile:
    @staticmethod
    def load(ymlFile="data.yml", encoding="utf-8"):
        """读取Yaml文件"""
        with open(ymlFile, "r", encoding=encoding) as f:
            return _yaml.load(f, Loader=_yaml.FullLoader)

    @staticmethod
    def write(item, ymlFile="data.yml", encoding="utf-8"):
        """写入Yaml文件"""
        with open(ymlFile, "w", encoding=encoding) as f:
            _yaml.dump(item, f, allow_unicode=True)


class JsonFile:
    @staticmethod
    def load(jsonFile="data.json", encoding="utf-8"):
        """读取Json文件"""
        with open(jsonFile, "r", encoding=encoding) as f:
            return _json.load(f)

    @staticmethod
    def write(item, jsonFile="data.json", encoding="utf-8", ensure_ascii=False):
        """写入Json文件"""
        with open(jsonFile, "w", encoding=encoding) as f:
            _json.dump(item, f, ensure_ascii=ensure_ascii)
