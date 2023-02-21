import json
import pathlib


def load_json(jsonFile="data.json", encoding="utf-8"):
    """读取Json文件"""
    with open(jsonFile, "r", encoding=encoding) as f:
        return json.load(f)


tikilib_path = pathlib.Path(__file__).parent
json_data = load_json(tikilib_path / "library.json")
