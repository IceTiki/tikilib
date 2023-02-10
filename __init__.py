# 标准库
import random

# from .file import *
# from .sendmsg import *
# from .crypto import *
# from .math import *
# from .enhance import *
# from .image import *
# from .plot import *


class Misc:
    """杂项"""

    hex_chr = "0123456789ABCDEF"
    word_chr = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

    @staticmethod
    def random_hex(chara: str = hex_chr, len_: int = 8):
        return "".join(random.choices(chara, k=len_))

    @staticmethod
    def random_word(chara: str = word_chr, len_: int = 8):
        return "".join(random.choices(chara, k=len_))


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
