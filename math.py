if __name__ == "__main__":
    import numpy as _np
    import typing as _typing
else:
    from . import LazyImport

    __globals = globals()
    __globals["_np"] = LazyImport("numpy")
    __globals["_typing"] = LazyImport("typing")


def dichotomy(a=-10, b=10, func=lambda x: x, times=100):
    """
    二分法求零点
    :params a, b: 区间起终。本函数在[a,b]区间内，寻找func的零点并返回零点位置。
    :params func: 函数
    :params times: 迭代次数
    """
    for _ in range(times + 1):
        if func(a) == 0:
            return a
        elif func(b) == 0:
            return b
        elif func(a) * func(b) > 0:
            return "error"
        x0 = (a + b) / 2
        if func(x0) == 0:
            return x0
        elif func(x0) * func(a) > 0:
            a = x0
        elif func(x0) * func(b) > 0:
            b = x0
        # print('%6.6f|%6.6f' % (x0, func(x0)))
    return x0


def mod(a, b):
    """取模(等价于python中的%)"""
    return a - a // b * b


def rem(a, b):
    """取余"""
    return a - int(a / b) * b


class MathFunction:
    @staticmethod
    def two_dimensional_gaussian_distribution(
        x,
        y,
        sigma_1: float = 1,
        sigma_2: float = 1,
        mu_1: float = 0,
        mu_2: float = 0,
        rho: float = 0,
    ):
        """二维正态分布"""
        c1 = 1 / (2 * _np.pi * sigma_1 * sigma_2 * _np.sqrt(1 - rho**2))
        c2 = -1 / (2 * (1 - rho**2))
        c3 = (
            (x - mu_1) ** 2 / sigma_1**2
            + (y - mu_2) ** 2 / sigma_2**2
            - 2 * rho * (x - mu_1) * (y - mu_2) / (sigma_1 * sigma_2)
        )
        return c1 * _np.exp(c2 * c3)

    @staticmethod
    def gaussian_distribution(x, sigma: float = 1, mu: float = 0):
        """正态分布"""
        return _np.exp(-((x - mu) ** 2) / 2 * sigma**2) / (
            _np.sqrt(2 * _np.pi) * sigma
        )


def normalization(arr: _np.ndarray) -> _np.ndarray:
    """依据数组的最大值和最小值归一化"""
    arr = _np.array(arr)
    max_, min_ = arr.max(), arr.min()
    arr = arr - min_
    arr = arr if (max_ - min_) == 0 else arr / (max_ - min_)
    return arr
def axis_angle2rotation_matrix(axis_vector: np.ndarray, left: bool = False):
    """
    将「轴角」转换为「旋转矩阵」

    Parameters
    ---
    axis_vector : np.ndarray
        代表转动量在x, y, z上的分量, 其模长即为转角(弧度制)
    left : bool, default = False
        是否使用左手系(伸出拇指, 握紧四指时, 拇指为向量方向, 四指为转动方向)

    Note
    ---
    公式来源:
        - [三维旋转：欧拉角、四元数、旋转矩阵、轴角之间的转换](https://zhuanlan.zhihu.com/p/45404840)
        - [机器人正运动学---姿态描述之轴角（旋转向量）](https://blog.csdn.net/hitgavin/article/details/106713290)
    """
    modulus = _np.linalg.norm(axis_vector, 2)  # 模长, 即为转动角度
    angle = modulus if left else -modulus
    cos_a = _np.cos(angle)
    sin_a = _np.sin(angle)

    x, y, z = axis_vector / modulus
    return _np.array(
        [
            [
                (1 - cos_a) * x**2 + cos_a,
                (1 - cos_a) * x * y - z * sin_a,
                (1 - cos_a) * x * z + y * sin_a,
            ],
            [
                (1 - cos_a) * x * y + z * sin_a,
                (1 - cos_a) * y**2 + cos_a,
                (1 - cos_a) * y * z - x * sin_a,
            ],
            [
                (1 - cos_a) * x * z - y * sin_a,
                (1 - cos_a) * y * z + x * sin_a,
                (1 - cos_a) * z**2 + cos_a,
            ],
        ]
    )
