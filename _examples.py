"""代码练习或参考示例"""
import typing
import pathlib
import time

import cv2
from itertools import product


def find_data_from_img(
    imgpath, data_color=(0, 0, 255), angle_color=(0, 255, 0), axis_shape=(0.02, 4000)
):
    """
    todo 注释
    BGR color tuple
    angle: 从左上到右下
    """
    item = cv2.imread(imgpath)
    data = []
    angle = []
    for i, j in product(range(item.shape[0]), range(item.shape[1])):
        pix = item[i][j]
        bgr = tuple(int(i) for i in pix)
        if data_color == bgr:
            data.append((i, j))
        if len(angle) < 2 and angle_color == bgr:
            angle.append((i, j))

    angle_shape = (angle[1][0] - angle[0][0], angle[1][1] - angle[0][1])
    data.sort(key=lambda x: x[1], reverse=False)

    X = [(i[1] - angle[0][1]) / angle_shape[1] * axis_shape[0] for i in data]
    Y = [(1 - (i[0] - angle[0][0]) / angle_shape[0]) * axis_shape[1] for i in data]
    return X, Y


def nbase_generator_loop(base: int, digit: int, endian: typing.Literal["S", "B"] = "B"):
    """
    n进制tuple生成器

    Parameters
    ---
    base : int
        基
    digit : int
        位
    endian : {'S', 'B'}, optional
        "S"代表小端序, "B"代表大端序
    Yields:
    ---
    tuple[int, ...]
    """
    stack: list[int] = [0] * digit
    if endian == "S":
        while True:
            yield tuple(stack)
            deep: int = 0
            while deep < digit:
                if stack[deep] + 1 >= base:
                    # 进位
                    stack[deep] = 0
                    deep += 1
                else:
                    # 本位+1
                    stack[deep] += 1
                    break
            else:
                break
    elif endian == "B":
        reset_deep = digit - 1
        while True:
            yield tuple(stack)
            deep: int = reset_deep
            while deep >= 0:
                if stack[deep] + 1 >= base:
                    # 进位
                    stack[deep] = 0
                    deep -= 1
                else:
                    # 本位+1
                    stack[deep] += 1
                    break
            else:
                break
    else:
        raise ValueError('parameter "endian" only support value "S" or "B".')


def nbase_generator_recursion(base: int, digit: int, endian: typing.Literal["S", "B"] = "B"):
    """
    n进制tuple生成器

    Parameters
    ---
    base : int
        基
    digit : int
        位
    endian : {'S', 'B'}, optional
        "S"代表小端序, "B"代表大端序

    Yields:
    ---
    tuple[int, ...]
    """
    if digit == 0:
        return
    if digit == 1:
        yield from (tuple([i]) for i in range(base))
    else:
        if endian == "S":
            yield from (
                tuple([j]) + i
                for i in nbase_generator_recursion(base, digit - 1, endian)
                for j in range(base)
            )
        elif endian == "B":
            yield from (
                tuple([i]) + j
                for i in range(base)
                for j in nbase_generator_recursion(base, digit - 1, endian)
            )
        else:
            raise ValueError('parameter "endian" only support value "S" or "B".')


def product_(*iterables: typing.Iterable):
    """
    作用等同于itertools.product

    Note
    ---
    是大端序(先迭代最后一位)
    """
    iter_count = len(iterables)
    if iter_count == 0:
        return
    if iter_count == 1:
        yield from (tuple([i]) for i in iterables[0])
    else:
        yield from (
            tuple([i]) + j for i in iterables[0] for j in product_(*iterables[1:])
        )


def tree(root_path: pathlib.Path, __level=0):
    """模拟tree命令, 列出文件夹中的所有文件"""
    root_path = pathlib.Path(root_path)
    try:
        for item in root_path.iterdir():
            if item.is_dir():
                yield "\t" * __level + f"- [x] `{item.name}`"
                yield from tree(item, __level + 1)
            else:
                yield "\t" * __level + f"- [ ] `{item.name}`"
    except WindowsError as e:
        print(e)
        return

def print_ddl(ddl: str):
    deadline: time.struct_time = time.strptime(ddl, "%Y-%m-%d--%H-%M-%S")  # 解析字符串为时间元组
    deadline: float = time.mktime(deadline)  # 解析为时间戳(也就是距离)
    to_ddl: float = deadline - time.time()

    d, h, m, s = map(
        int,
        (
            to_ddl / 86400,  # 86400是一天的秒数
            (to_ddl % 86400) / 3600,
            (to_ddl % 3600) / 60,
            (to_ddl % 60),
        ),
    )
    print(f"距离作业回收还剩下: {d}天{h}时{m}分{s}秒")