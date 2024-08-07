if __name__ == "__main__":
    from matplotlib import pyplot as _plt
    import numpy as _np
    import seaborn as _sns
else:
    from . import BatchLazyImport

    BatchLazyImport(
        globals(),
        locals(),
        """
    from matplotlib import pyplot as _plt
    import numpy as _np
    import seaborn as _sns
    """,
    )


def chinese_font_support() -> None:
    """matplotlib的中文显示支持的自动设置函数"""
    _plt.rcParams["font.sans-serif"] = ["MicroSoft YaHei"]  # 用来正常显示中文标签
    # plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号




def set_color_palette_from_seaborn(color_palette_name="Set1"):
    """常用: tab20c, tab20, Set1"""
    _sns.set_palette(_sns.color_palette(color_palette_name))


def get_ax_matrix(
    nrows=2, ncols=2, figsize=(12, 8)
) -> tuple[_plt.Figure, _np.ndarray | _plt.Axes, list[_plt.Axes]]:
    """

    Returns
    ---
    fig, axs, ax_linear_list : plt.Figure, np.ndarray | plt.Axes, list[plt.Axes]
        Figure对象, axes矩阵, axes列表(一维)
        - 行列均为1时, axs为matplotlib.axes._axes.Axes
        - 行列其一为1时, axs为一维的np.ndarray
        - 行列均大于1时, axs为二维的np.ndarray
        - ax_linear_list为一维的列表
    """
    fig, axs = _plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True
    )

    if nrows == 1 and ncols == 1:
        ax_linear_list = [axs]
    elif nrows == 1 or ncols == 1:
        ax_linear_list = axs.tolist()
    else:
        ax_linear_list = axs.flatten().tolist()

    # ax.set_aspect(1)

    return fig, axs, ax_linear_list
    x = np.linspace(*x_range)
    y = np.linspace(*y_range)
    X, Y = np.meshgrid(x, y)
    F = func(X, Y)
    return X, Y, F
