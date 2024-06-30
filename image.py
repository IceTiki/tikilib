if __name__ == "__main__":
    # 标准库
    import io as _io
    import pathlib as _pathlib
    import typing as _typing
    import itertools as _itertools
    import functools as _functools

    # 第三方库
    import numpy as _np
    from PIL import Image as _Image  # pillow
    import cv2 as _cv2  # opencv-python
    import scipy as _scipy
else:
    from . import BatchLazyImport

    BatchLazyImport(
        globals(),
        locals(),
        """
    # 标准库
    import io as _io
    import pathlib as _pathlib
    import typing as _typing
    import itertools as _itertools
    import functools as _functools

    # 第三方库
    import numpy as _np
    from PIL import Image as _Image  # pillow
    import cv2 as _cv2  # opencv-python
    import scipy as _scipy
    """,
    )

class CvIo:
    @staticmethod
    def load(
        file: _typing.Union[str, bytes, _pathlib.Path, _typing.BinaryIO],
        flags: int = _cv2.IMREAD_COLOR,
    ) -> _np.ndarray:
        """
        cv2.imread的支持中文路径版本

        Parameters
        ---
        file : str | bytes | os.PathLike[Any] | _IOProtocol
            文件, 用np.fromfile读取
        flags : int, default = 1
            cv2.IMREAD_UNCHANGED = -1
            cv2.IMREAD_GRAYSCALE = 0
            cv2.IMREAD_COLOR = 1

        """
        cv_img: _np.ndarray = _cv2.imdecode(_np.fromfile(file, dtype=_np.uint8), flags)
        return cv_img

    @staticmethod
    def write(
        img: _np.ndarray,
        file: _typing.Union[str, bytes, _pathlib.Path, _typing.BinaryIO],
        params=None,
        ext: str = None,
    ) -> None:
        """
        cv2.imwrite的支持中文路径版本

        Parameters
        ---
        img : np.ndarray
            图像
        file : str | bytes | os.PathLike[Any] | _IOProtocol
            输出路径, 用np.ndarray.tofile输出
        params : None
            imencode的params
        ext : None
            扩展名, 如果文件名为str|Path, 则读取其扩展名。否则默认取".png"。
        """
        if ext is None:
            if isinstance(file, (str, _pathlib.Path)):
                ext = _pathlib.Path(file).suffix
            else:
                ext = ".png"

        encode_data: _np.ndarray = _cv2.imencode(
            ext, img, *([params] if params else [])
        )[1]
        encode_data.tofile(file)

    @staticmethod
    def show(img: _np.ndarray, winname=None):
        winname = winname or "image"
        _cv2.imshow(winname, img)


class CvBlending:
    """
    基于opencv模块产生的numpy数组, 对图片叠加提供混合模式。
        混合模式的效果会尽量贴近Photoshop, 但会略有不同(因为数据类型等原因, 会有微量差异)

    NOTICE 目前只能处理unsign int8类型的BGR数组

    参考
    ---
    公式参考: https://blog.csdn.net/onafioo/article/details/54232689
        ps图层混合计算公式
            注意, 图片中的公式有些mask是<=128, 实际上是<128
            另外, 点光中B>=128时, 使用max而不是min
    原理参考: https://www.zhihu.com/question/22883942/answer/35657823
        如何通俗易懂的理解 Photoshop 中，关于图层混合模式那11大种类的意思？ - 宋顺宁.Seany的回答 - 知乎
    公式参考: https://zhuanlan.zhihu.com/p/23905865
        一篇文章彻底搞清PS混合模式的原理 - 以梦为马的文章 - 知乎

    数据处理
    ---
    经过调试, 为了接近PhotoShop的效果, 可以对数据做以下处理
        使用np.maximum(x, 1)避免被除数为0
        最终结果用np.clip(x, 0, 255)避免负数和超过255

    Numpy特性
    ---
    numpy中unsign类型如果出现负数, 则会从最大值开始减。
        比如a = np.array([4,4,4], dtype=np.uint8)
        那么-a = [252, 252, 252]
        但-1*a = [-4, -4, -4], 因为-1自动被广播为数组, 而且类型变为int16(因为int16才能装完uint8的正数)
    使用「/」, numpy会自动将数组转为float64类型。使用「//」, numpy会保留int类型。
    用mask(掩码)会将数组一维化
    可以对bool数组使用乘法, True视为1, Flase视为0

    色彩特性
    ---
    浅色和深色的实现中, 尝试过HSV的V、HSL和L、平均灰度来表示明度。最终opencv的加权灰度最符合photoshop效果。

    杂项
    ---
    灰度计算公式 https://blog.csdn.net/kuweicai/article/details/73414138
        Gray = (4898*R + 9618*G + 1868*B) >> 14
    Adobe的色彩空间
        https://www.zhihu.com/question/62362890/answer/345690499
        PS混合模式中色相、饱和度、颜色、明度4个模式的计算公式是什么？ - 卡米雷特的回答 - 知乎
    类型检查代码（检查值的范围）
    """

    def __set_dtype_for_param_and_return(
        param_type=_np.int32, return_type=_np.uint8, return_clip=None
    ):
        """
        :param return_clip: (tuple)将返回值限定在一定范围(min, max)
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                change_param = lambda x: (
                    _np.array(x, dtype=param_type) if isinstance(x, _np.ndarray) else x
                )
                args = (change_param(i) for i in args)
                kwargs = {change_param(i): j for i, j in kwargs}
                result = func(*args, **kwargs)
                if return_clip:
                    return _np.array(_np.clip(result, *return_clip), return_type)

                return _np.array(result, return_type)

            return wrapper

        return decorator

    @staticmethod
    def _for_debug_check_range(
        xrange=(0, 255, 256),
        yrange=(0, 255, 256),
        func=lambda x, y: x + y - x * y // 128,
    ):
        from tikilib import plot

        X, Y, F = plot.gene_gird(xrange, yrange, func)
        title = f"min: {_np.min(F)}|max: {_np.max(F)}"
        fig, ax = plot.gene_fig_ax()
        ax: plot.plt.Axes
        ax.set_title(title)
        ax.contourf(X, Y, F)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def invert(img: _np.ndarray):
        """反相"""
        return 255 - img

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def transparent(img_1: _np.ndarray, img_2: _np.ndarray, alpha: _np.ndarray):
        """不透明度"""
        return (img_1 * (255 - alpha) + img_2 * alpha) // 255

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def dissolve(canvas: _np.ndarray, img: _np.ndarray, alpha: _np.ndarray):
        """
        溶解

        Paramters
        ---
        canvas, img : np.ndarray
            置于底层和置于顶层的图片。
            - 长宽 : canvas与img的长宽相同。
            - 通道数(img.shape[-1]) : canvas与img的通道数相同。
            - 类型(img.dtype) : np.uint8, 取值范围为0~255。

        alpha : np.ndarray
            img的溶解透明度权重, 0~255分别对应溶解透明度为0~100%。
            - 长宽 : 与canvas, img相同。
            - 通道数(img.shape[-1]) : 1。
            - 类型 : np.uint8, 取值范围为0~255。

        Returns
        ---
        img_result : np.ndarray
            叠加完成的图片
            - 通道数(img.shape[-1]) : 与canvas, img相同。
            - 类型 : np.uint8, 取值范围为0~255。
        """
        random_mask = _np.random.randint(
            0, 256, canvas.shape[:-1]
        )  #  np.random.randint生成的随机数不包含高位。
        mask = random_mask <= alpha
        channel_n = canvas.shape[-1]
        mask = _np.expand_dims(mask, channel_n - 1).repeat(
            channel_n, axis=channel_n - 1
        )
        return _np.where(mask, img, canvas)

    # 变暗模式

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def dark(img_1: _np.ndarray, img_2: _np.ndarray):
        """变暗"""
        return _np.minimum(img_1, img_2)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def multiply(img_1: _np.ndarray, img_2: _np.ndarray):
        """正片叠底"""
        return img_1 * img_2 // 255

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def color_burn(img_1: _np.ndarray, img_2: _np.ndarray):
        """颜色加深"""
        return img_1 - _np.minimum(
            img_1,  # np.minimum避免出现负数
            (255 - img_1)
            * (255 - img_2)
            // _np.maximum(img_2, 1),  # np.maximum避免出现除数为0
        )

    @staticmethod
    @__set_dtype_for_param_and_return(_np.int16)
    def linear_burn(img_1: _np.ndarray, img_2: _np.ndarray):
        """线性加深"""
        return _np.maximum(img_1 + img_2 - 255, 0)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def darker_color(img_1: _np.ndarray, img_2: _np.ndarray):
        """深色"""
        mask = _cv2.cvtColor(img_2, _cv2.COLOR_BGR2GRAY) <= _cv2.cvtColor(
            img_1, _cv2.COLOR_BGR2GRAY
        )
        mask = _np.expand_dims(mask, 2).repeat(3, axis=2)
        return _np.where(mask, img_2, img_1)

    # 变亮模式

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def lighten(img_1: _np.ndarray, img_2: _np.ndarray):
        """变亮"""
        return _np.maximum(img_1, img_2)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def screen(img_1: _np.ndarray, img_2: _np.ndarray):
        """滤色"""
        return 255 - ((255 - img_1) * (255 - img_2)) // 255

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def color_dodge(img_1: _np.ndarray, img_2: _np.ndarray):
        """颜色减淡"""
        return _np.minimum(255, img_1 + img_1 * img_2 // _np.maximum(255 - img_2, 1))

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def linear_dodge(img_1: _np.ndarray, img_2: _np.ndarray):
        """线性减淡(添加)"""
        return img_1 + _np.minimum(255 - img_1, img_2)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def lighter_color(img_1: _np.ndarray, img_2: _np.ndarray):
        """浅色"""
        mask = _cv2.cvtColor(img_2, _cv2.COLOR_BGR2GRAY) >= _cv2.cvtColor(
            img_1, _cv2.COLOR_BGR2GRAY
        )
        mask = _np.expand_dims(mask, 2).repeat(3, axis=2)
        return _np.where(mask, img_2, img_1)

    # 饱和度模式

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def overlay(img_1: _np.ndarray, img_2: _np.ndarray):
        """叠加"""
        result = _np.zeros(img_1.shape, dtype=img_1.dtype)
        mask = img_1 < 128
        invert_mask = ~mask
        result[mask] = img_1[mask] * img_2[mask] // 128
        result[invert_mask] = 255 - (
            (255 - img_1[invert_mask]) * (255 - img_2[invert_mask]) // 128
        )
        return result

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def soft_light(img_1: _np.ndarray, img_2: _np.ndarray):
        """柔光"""
        result = _np.zeros(img_1.shape, dtype=img_1.dtype)

        # mask
        mask = img_2 < 128
        img_m1, img_m2 = img_1[mask], img_2[mask]

        result[mask] = (
            img_m1 * img_m2 // 128
            + (img_m1 * img_m1) // 255 * (255 - 2 * img_m2) // 255
        )

        # invert_mask
        invert_mask = ~mask
        img_im1, img_im2 = img_1[invert_mask], img_2[invert_mask]

        res_2t = 2 * img_im2 - 255
        result[invert_mask] = img_im1 * (255 - img_im2) // 128 + _np.sqrt(
            img_im1 * res_2t // 255 * res_2t
        )
        return _np.minimum(255, result)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def hard_light(img_1: _np.ndarray, img_2: _np.ndarray):
        """强光"""
        result = _np.zeros(img_1.shape, dtype=img_1.dtype)

        # mask
        mask = img_2 < 128
        result[mask] = img_1[mask] * img_2[mask] // 128

        # invert_mask
        invert_mask = ~mask
        result[invert_mask] = (
            255 - (255 - img_1[invert_mask]) * (255 - img_2[invert_mask]) // 128
        )
        return result

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def vivid_light(img_1: _np.ndarray, img_2: _np.ndarray):
        """亮光"""
        result = _np.zeros(img_1.shape, dtype=img_1.dtype)

        # mask
        mask = img_2 < 128
        img_m1, img_m2 = img_1[mask], img_2[mask]

        img_m2_double = 2 * img_m2
        result[mask] = img_m1 - _np.minimum(  #  np.minimum避免出现负数
            img_m1,
            (255 - img_m1)
            * (255 - img_m2_double)
            // _np.maximum(1, img_m2_double),  #  np.maximum避免被除数为0
        )

        # invert_mask
        invert_mask = ~mask
        img_im1, img_im2 = img_1[invert_mask], img_2[invert_mask]

        result[invert_mask] = _np.minimum(
            255, img_im1 + img_im1 * (2 * img_im2 - 255) // (2 * (255 - img_im2))
        )
        return result

    @staticmethod
    @__set_dtype_for_param_and_return(_np.int16)
    def linear_light(img_1: _np.ndarray, img_2: _np.ndarray):
        """线性光"""
        return _np.clip(img_1 + 2 * img_2 - 255, 0, 255)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def pin_light(img_1: _np.ndarray, img_2: _np.ndarray):
        """点光"""
        result = _np.zeros(img_1.shape, dtype=img_1.dtype)

        # mask
        mask = img_2 < 128
        img_m1, img_m2 = img_1[mask], img_2[mask]

        result[mask] = _np.minimum(img_m1, 2 * img_m2)

        # invert_mask
        invert_mask = ~mask
        img_im1, img_im2 = img_1[invert_mask], img_2[invert_mask]

        result[invert_mask] = _np.maximum(img_im1, 2 * (img_im2 - 128) + 1)
        return result

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def hard_mix(img_1: _np.ndarray, img_2: _np.ndarray):
        """实色混合"""
        return _np.where(
            255 - img_1 < img_2,
            _np.array(255, dtype=img_1.dtype),
            _np.array(0, dtype=img_1.dtype),
        )

    # 差集模式

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def difference(img_1: _np.ndarray, img_2: _np.ndarray):
        """差值"""
        return _np.where(
            img_1 >= img_2, img_1 - img_2, img_2 - img_1
        )  # 等价于np.abs(img_1 - img_2)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def exclusion(img_1: _np.ndarray, img_2: _np.ndarray):
        """排除"""
        return img_1 + img_2 - img_1 * img_2 // 128

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint8)
    def subtract(img_1: _np.ndarray, img_2: _np.ndarray):
        """减去"""
        return _np.where(img_1 >= img_2, img_1 - img_2, 0)

    @staticmethod
    @__set_dtype_for_param_and_return(_np.uint16)
    def divide(img_1: _np.ndarray, img_2: _np.ndarray):
        """划分"""
        return _np.minimum(255, img_1 * 255 // _np.maximum(1, img_2))

    # 颜色模式(HSL系)
    # todo色相
    # todo饱和度
    # todo颜色
    # todo明度


class CvOperation:
    """
    基于opencv模块产生的numpy数组, 进行各种操作
    """

    LiteralPosition = _typing.Literal["center", "top", "bottom", "left", "right"]

    @classmethod
    def put_on_canvas_slice(
        cla,
        canvas_shape: tuple[int, int],
        img_shape: tuple[int, int],
        align: LiteralPosition = "center",
    ):
        """
        :params canvas_shape, img_shape: 幕布形状, 图片形状(先高后宽)
        将img叠加在canvas上(返回叠加切片)
        使用方法:
            canvas[*put_on_canvas_slice(canvas.shape, img.shape, align), :]
        """
        # todo检查img长宽都比canvas小
        canvas_high, canvas_width = canvas_shape[:2]
        img_high, img_width = img_shape[:2]

        high_center = slice(
            int((canvas_high - img_high) / 2),
            int((canvas_high + img_high) / 2),
        )
        width_center = slice(
            int((canvas_width - img_width) / 2),
            int((canvas_width + img_width) / 2),
        )
        slice_top = slice(0, img_high)
        slice_bottom = slice(canvas_high - img_high, canvas_high)
        slice_left = slice(0, img_width)
        slice_right = slice(canvas_width - img_width, canvas_width)

        if align == "center":
            result_slice = high_center, width_center
        elif align == "top":
            result_slice = slice_top, width_center
        elif align == "bottom":
            result_slice = slice_bottom, width_center
        elif align == "left":
            result_slice = high_center, slice_left
        elif align == "right":
            result_slice = high_center, slice_right
        else:
            raise ValueError(f'不支持的对齐方式"{align}" (param align)')
        return tuple(result_slice)

    @classmethod
    def put_on_canvas(
        cla,
        canvas: _np.ndarray,
        img: _np.ndarray,
        align: LiteralPosition = "center",
        mode=None,
    ):
        """
        Introduction
        ---
        将img叠加在canvas上

        Parameters
        ---
        canvas : numpy.ndarray
            底部图片
        img : numpy.ndarray
            叠加在上的图片
        align : str
            有center/top/bottom/left/right
        mode : f(x,y)-->numpy.ndarray
            叠加模式。传入一个二元函数，默认为覆盖模式。
        """
        mode = mode or (lambda img_1, img_2: img_2)

        img = cla.resize_on_canvas(canvas, img)
        canvas = canvas.copy()
        canvas[*cla.put_on_canvas_slice(canvas.shape, img.shape, align), :] = mode(
            canvas[*cla.put_on_canvas_slice(canvas.shape, img.shape, align), :], img
        )
        return canvas

    @staticmethod
    def resize_on_canvas(canvas: _np.ndarray, img: _np.ndarray) -> _np.ndarray:
        """
        将img调整到刚好能放置在canvas上(长或宽一致)
        返回调整后的img
        """
        # cv2.resize先宽后高, np.ndarray.shape先高后宽
        # canvas_shape = canvas.shape[1::-1]
        # img_shape = img.shape[1::-1]
        canvas_high, canvas_width, _ = canvas.shape
        img_high, img_width, _ = img.shape

        width_ratio = img_width / canvas_width
        high_ratio = img_high / canvas_high
        new_scale = max(width_ratio, high_ratio)
        if width_ratio >= high_ratio:
            new_shape = (canvas_width, int(img_high / new_scale))
        else:
            new_shape = (int(img_width / new_scale), canvas_high)
        return _cv2.resize(img, new_shape)

    @classmethod
    def joint(cla, img_array: _typing.List[_typing.List[_np.ndarray]]) -> _np.ndarray:
        """
        todo 缺少注释
        先列索引后行索引
        """
        if isinstance(img_array[0], _np.ndarray):
            img_array = [img_array]

        img_array_shape = (len(img_array), len(img_array[0]))
        idx_iter = _itertools.product(range(len(img_array)), range(len(img_array[0])))
        idx_iter = list(
            filter(lambda x: isinstance(img_array[x[0]][x[1]], _np.ndarray), idx_iter)
        )

        max_shape = [0, 0, 3]
        for i, j in idx_iter:
            img = img_array[i][j]
            max_shape[0] = max(max_shape[0], img.shape[0])
            max_shape[1] = max(max_shape[1], img.shape[1])
            max_shape[2] = max(max_shape[2], img.shape[2])

        for i, j in idx_iter:
            img = img_array[i][j]
            canvas = _np.full(max_shape, 255)
            img_array[i][j] = cla.put_on_canvas(canvas, img)

        big_canvas = _np.full(
            tuple(i * j for i, j in zip(max_shape, img_array_shape + (1,))), 255
        )

        for i, j in idx_iter:
            big_canvas[
                i * max_shape[0] : (i + 1) * max_shape[0],
                j * max_shape[1] : (j + 1) * max_shape[1],
                :,
            ] = img_array[i][j]

        return big_canvas


class Dhash:
    @staticmethod
    def calculate(img_path: str | bytes | _pathlib.Path, shape=(64, 65)) -> str:
        """
        计算Dhash, 以hex字符串储存结果

        Parameters
        ---
        img_path : str | bytes | Path
            图片路径
        """
        img = CvIo.load(img_path)
        img = _cv2.resize(img, shape)
        img = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)

        img: _np.ndarray = img[:-1, :] > img[1:, :]

        hash_bool_array = img.flatten("C")
        hash_bool_array: _np.ndarray = hash_bool_array.astype("uint8")

        match hash_bool_array.shape[0] % 8:
            # 补0, 确保数组长度是8的倍数
            case 0:
                pass
            case padding:
                hash_bool_array = _np.pad(
                    hash_bool_array,
                    (0, 8 - padding),
                    "constant",
                    constant_values=(0, 0),
                )
        # 8 * bool_ -> uint8
        hex_result: _np.ndarray = sum(
            (2**i * hash_bool_array[i::8] for i in range(8))
        )
        # uint8 -> bytes -> hex
        hex_result = hex_result.tobytes().hex()
        return hex_result

    @staticmethod
    def diffence(hash1: str, hash2: str) -> float:
        """
        借助汉明距离计算差异率
        一般小于10%, 可视为同一或相似图片

        - hash字符串应取计算时, resize到相同shape的两者。
        """

        def hex_to_bool_array(hex_string: str) -> _np.ndarray:
            # hex -> bytes -> uint8 -> 8 * bool_
            item = _np.frombuffer(bytes.fromhex(hex_string), dtype="uint8")
            result = _np.zeros(item.shape[0] * 8, dtype="bool_")
            for i in range(8):
                # 通过与操作和移位操作将uint8拆开为8个bool_
                result[i::8] = _np.right_shift(
                    _np.bitwise_and(item, _np.array(2**i, dtype="uint8")), i
                )
            return result

        hash1, hash2 = map(hex_to_bool_array, (hash1, hash2))
        hash1: _np.ndarray
        hash2: _np.ndarray
        return _np.count_nonzero(_np.bitwise_xor(hash1, hash2)) / hash1.shape[0]

    @classmethod
    def _test_find_similar(cls, folder: str, factor=0.1):
        """查找文件夹中的相似图片"""
        file_hash: _typing.List[tuple[_pathlib.Path, str]] = []

        res_str = []

        for i, item in enumerate(
            filter(lambda x: x.is_file(), _pathlib.Path(folder).glob("**\\*"))
        ):
            if item.suffix not in (".jpg", ".jpeg", ".png"):
                continue

            try:
                dhash = cls.calculate(item)
            except Exception as e:
                print(e)
                continue

            for file_path_2, dhash_2 in file_hash:
                if cls.diffence(dhash, dhash_2) <= factor:
                    res_str.append(f"{file_path_2.name}\t{item.name}")
                    break

            file_hash.append((item, dhash))

            if i % 100 == 0:
                print(i)

        for i in res_str:
            print(i)
        print("ok")


class __old_Dhash:
    def __init__(self, image, resize=(33, 32)):
        """
        :param image:(str|bytes) 图片
        :param resize: (tuple)宽, 高
        """
        self.resize = resize
        if type(image) == str:
            self.image: _Image.Image = _Image.open(image)
        elif type(image) == bytes:
            self.image: _Image.Image = _Image.open(_io.BytesIO(image))
        self._grayscale_Image()
        self.dhash = self._hash_string

    def _grayscale_Image(self):
        """
        缩放并灰度图片
        """
        smaller_image = self.image.resize(self.resize)  # 将图片进行缩放
        grayscale_image = smaller_image.convert("L")  # 将图片灰度化
        self.image = grayscale_image
        return self.image

    @property
    def _hash_string(self):
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


def _test_find_color(bgrimg: _np.ndarray, color="#0000ff", sort=True):
    bgr_color_tuple = (color[1:3], color[3:5], color[5:])[::-1]
    bgr_color_tuple = tuple(map(lambda x: int(x, 16), bgr_color_tuple))
    position_x = []
    position_y = []
    for i, j in _itertools.product(range(bgrimg.shape[0]), range(bgrimg.shape[1])):
        pix = bgrimg[i][j]
        bgr = tuple(int(i) for i in pix)
        if bgr_color_tuple == bgr:
            position_x.append(i)
            position_y.append(j)
    if sort and position_x:
        position_x, position_y = zip(
            *sorted(list(zip(position_x, position_y)), key=lambda x: x[1])
        )
    return (position_x, position_y)


def _test_get_curve(
    imgpath,
    data_color: dict = {"curve1": "#ff0000"},
    edge_color="#00ff00",
    axis_shape=(0.02, 4000),
):
    """
    BGR color tuple
    """
    bgrimg = _cv2.imread(imgpath)
    data = {k: _test_find_color(bgrimg, v) for k, v in data_color.items()}
    edge_x, edge_y = _test_find_color(bgrimg, edge_color)

    edge_shape = (
        max(edge_x) - min(edge_x),
        max(edge_y) - min(edge_y),
    )

    for k in data.keys():
        v = data[k]
        posi_x, posi_y = v
        data[k] = {
            "x": [(i - min(edge_y)) / edge_shape[1] * axis_shape[0] for i in posi_y],
            "y": [
                (1 - (i - min(edge_x)) / edge_shape[0]) * axis_shape[1] for i in posi_x
            ],
        }
    return data


def img2gif(imgs: _typing.Iterable, output: _pathlib.Path, **kwargs):
    """
    图像转gif
    :param imgs: 图片
    :param output: gif输出(str | Path | file object)
    """
    kwargs.setdefault("save_all", True)
    kwargs.setdefault("loop", True)
    imgs: _typing.List[_Image.Image] = [_Image.open(img) for img in imgs]
    imgs[0].save(output, append_images=imgs[1:], **kwargs)


def wechat_image_decode(dat_dir: _pathlib.Path, img_dir: _pathlib.Path = None):
    """
    解码dat或rst文件为图片
    :param dat_dir: dat文件路径
    :param img_dir: 输出图片文件的路径(自动将拓展名更改为对应图片类型)
    """
    dat_dir = _pathlib.Path(dat_dir)
    img_dir = _pathlib.Path(img_dir) if img_dir is not None else dat_dir
    img_xor = {
        ".jpeg": (0xFF, 0xD8, 0xFF),
        ".png": (0x89, 0x50, 0x4E),
        ".gif": (0x47, 0x49, 0x46),
    }

    with open(dat_dir, "rb") as dat_file_read:
        # 判断图片格式, 判断异或值和扩展名
        head = _np.fromfile(dat_file_read, dtype="uint8", count=3)
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
            for dat_chunk in iter(
                _functools.partial(dat_file_read.read, 1024 * 1024), b""
            ):
                n_b1 = _np.frombuffer(dat_chunk, dtype="uint8")
                img_block: _np.ndarray = n_b1 ^ img_xor
                img_block = img_block.tobytes()
                img_file_write.write(img_block)
