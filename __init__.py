"""
Tiki的常用函数库
"""


class Counter:
    """计数器"""

    data = {}

    def __init__(self, name, start=0):
        """
        :params name: 计数器名称
        :params start: 初始数值
        """
        self.name = name
        if name in self.data:
            self.data[name] += 1
        else:
            self.data[name] = start

    def __str__(self) -> str:
        return str(self.data[self.name])
