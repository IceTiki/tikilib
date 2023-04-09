# 标准库
import io
from functools import partial
from pathlib import Path
from typing import Iterable

# 第三方库
import numpy as np
from PIL import Image  # pillow
import cv2 as _cv2  # opencv-python




class Dhash:
    def __init__(self, image, resize=(33, 32)):
        """
        :param image:(str|bytes) 图片
        :param resize: (tuple)宽, 高
        """
        self.resize = resize
        if type(image) == str:
            self.image: Image.Image = Image.open(image)
        elif type(image) == bytes:
            self.image: Image.Image = Image.open(io.BytesIO(image))
        self._grayscale_Image()
        self.dhash = self._hash_String

    def _grayscale_Image(self):
        """
        缩放并灰度图片
        """
        smaller_image = self.image.resize(self.resize)  # 将图片进行缩放
        grayscale_image = smaller_image.convert("L")  # 将图片灰度化
        self.image = grayscale_image
        return self.image

    @property
    def _hash_String(self):
        """
        计算Dhash
        """
        pixels = list(self.image.getdata())
        hash_string = "".join(
            "1" if pixels[row - 1] > pixels[row] else "0"
            for row in range(1, len(pixels) + 1)
            if row % self.resize[0]
        )
        return int(hash_string, 2)  # 把64位数当作2进制的数值并转换成十进制数值

    def __sub__(self, dhash):
        if type(dhash) == Dhash:
            dhash = dhash.dhash
        return self.hamming_distance(self.dhash, dhash)

    def __str__(self) -> str:
        return str(self.dhash)

    @property
    def threshold(self, factor=0.05):
        """判定图片是否相同阈值(严格0.05, 正常0.1)"""
        return self.resize[0] * self.resize[1] * factor

    @staticmethod
    def hamming_distance(dhash1, dhash2):
        """
        汉明距离计算
        :param dhash1, dhash2: int类型的DHash
        :returns :int类型, 汉明距离
        """
        difference = dhash1 ^ dhash2  # 将两个数值进行异或运算
        # 异或运算后计算两数不同的个数, 一般 『<resize[0]*resize[1]*0.1』, 可视为同一或相似图片
        return bin(difference).count("1")


def img2gif(imgs: Iterable, output: Path, **kwargs):
    """
    图像转gif
    :param imgs: 图片
    :param output: gif输出(str | Path | file object)
    """
    kwargs.setdefault("save_all", True)
    kwargs.setdefault("loop", True)
    imgs = [Image.open(img) for img in imgs]

    imgs[0].save(output, append_images=imgs[1:], **kwargs)


def wechat_image_decode(dat_dir: Path, img_dir: Path = None):
    """
    解码dat或rst文件为图片
    :param dat_dir: dat文件路径
    :param img_dir: 输出图片文件的路径(自动将拓展名更改为对应图片类型)
    """
    dat_dir = Path(dat_dir)
    img_dir = Path(img_dir) if img_dir is not None else dat_dir
    img_xor = {
        ".jpeg": (0xFF, 0xD8, 0xFF),
        ".png": (0x89, 0x50, 0x4E),
        ".gif": (0x47, 0x49, 0x46),
    }

    with open(dat_dir, "rb") as dat_file_read:
        # 判断图片格式, 判断异或值和扩展名
        head = np.fromfile(dat_file_read, dtype="uint8", count=3)
        for may_suffix, may_xor in img_xor.items():
            may_head = head ^ may_xor
            if may_head[0] == may_head[1] == may_head[2]:  # 三异或值相等, 确定格式
                img_suffix, img_xor = may_suffix, may_head[0]
                break
        else:
            raise ValueError("未知数据格式")
        # 准备转码
        dat_file_read.seek(0)
        img_dir = img_dir.with_suffix(img_suffix)
        # 开始转码
        with open(img_dir, "wb") as img_file_write:
            for dat_chunk in iter(partial(dat_file_read.read, 1024 * 1024), b""):
                n_b1 = np.frombuffer(dat_chunk, dtype="uint8")
                img_block: np.ndarray = n_b1 ^ img_xor
                img_block = img_block.tobytes()
                img_file_write.write(img_block)
