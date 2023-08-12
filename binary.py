# 标准库
import pathlib as _pathlib
import typing as _typing

# 第三方库
import pandas as _panda  # pandas
import fitz as _fitz  # fitz, PyMuPDF
import py7zr as _py7zr  # py7zr
import imghdr as _imghdr
import numpy as _numpy

# tikilib
from . import system as _ts


class PandasExcelSheet:
    def __init__(self, file_path: str | _pathlib.Path, sheet_name: str):
        """
        Parameters
        ---
        file_path : str | pathlib.Path
            xlsx文件路径
        sheet_name : str
            表格名称
        """
        self.file_path: _pathlib.Path = _pathlib.Path(file_path)
        self.sheet_name: str = sheet_name
        self.__read_cache: _panda.DataFrame = None

    def write(self, data: _panda.DataFrame, **kwargs):
        """
        Parameters
        ---
        data : ndarray (structured or homogeneous), Iterable, dict, or DataFrame
            用于DataFrame初始化
        kwargs
            panda.DataFrame.to_excel的参数, index的默认值为False, header的默认值为True
        """
        kwargs.setdefault("index", False)
        kwargs.setdefault("header", True)
        _panda.DataFrame(data).to_excel(self.file_path, self.sheet_name, **kwargs)

    def __read(self, force_update: bool = False):
        """更新表格信息"""
        if self.__read_cache == None or force_update:
            self.__read_cache = _panda.read_excel(self.file_path, self.sheet_name)
        return self.__read_cache

    @property
    def dataframe(self) -> _panda.DataFrame:
        return self.__read()

    @property
    def shape(self) -> tuple[int, int]:
        """
        表格形状:tuple(行数:int, 列数:int)
        """
        return self.__read().shape

    @property
    def index_of_column(self) -> _panda.Index:
        """
        列标题/首行, 空白的标题会填充为"Unnamed"
        """
        return self.__read().columns

    @property
    def values(self) -> _numpy.ndarray:
        """
        表格数据(不含标题行/首行))

        Returns
        ---
        numpy.ndarray
            行优先矩阵, 包含表格所有行的数据(不含标题行)
        """
        return self.__read().values


class MuPdf:
    @staticmethod
    def pdf2png(
        pdf_path: _typing.Union[str, _pathlib.Path] = "a.pdf",
        output_folder: _typing.Union[str, _pathlib.Path] = "image",
        zoom: _typing.Union[int, tuple[int, int]] = 2,
        division: tuple[int, int] = (1, 1),
    ):
        """
        将pdf转为png

        Parameters
        ---
        pdf_path : str | pathlib.Path
            pdf路径
        output_folder : str | pathlib.Path
            输出文件夹
        zoom : int | tuple[int, int]
            放大系数(与图片清晰度有关)(如果是元组, 则先宽后高)
        division : tuple[int, int]
            将每一页(水平方向, 垂直方向)分割为n份

        Note
        ---
        pixmap支持的保存图片格式见https://pymupdf.readthedocs.io/en/latest/pixmap.html#supported-output-image-formats
        """
        pdf_path, output_folder = map(_pathlib.Path, (pdf_path, output_folder))
        name = pdf_path.stem
        pdf = _fitz.Document(pdf_path)
        if isinstance(zoom, int):
            zoom = tuple((zoom, zoom))

        output_folder.mkdir(parents=True, exist_ok=True)

        for page_number, page in enumerate(pdf):
            """逐页遍历"""
            page: _fitz.Page
            mat: _fitz.Matrix = _fitz.Matrix(*zoom).prerotate(0)  # 页面大小属性
            rect: _fitz.Rect = page.rect  # 页面总范围
            width = rect.width
            height = rect.height
            delta_width = width / division[0]
            delta_height = height / division[1]
            for x_index in range(division[0]):
                for y_index in range(division[1]):
                    """逐个切图"""
                    print(
                        f"processing: {x_index}/{division[0]}-{y_index}/{division[1]}"
                    )
                    clip = _fitz.Rect(
                        delta_width * x_index,
                        delta_height * y_index,
                        delta_width * (x_index + 1),
                        delta_height * (y_index + 1),
                    )  # 裁剪范围(x0, y0, x1, y1)
                    pix: _fitz.Pixmap = page.get_pixmap(matrix=mat, clip=clip)
                    pix.save(
                        output_folder / f"{name}_{page_number}_y{y_index}x{x_index}.png"
                    )

    @staticmethod
    def img2pdf(
        pic_iter: _typing.Iterable[_typing.Union[str, _pathlib.Path]] = _pathlib.Path(
            "."
        ).iterdir(),
        pdf_name: _typing.Union[str, _pathlib.Path] = "images.pdf",
        filter_: _typing.Callable = _imghdr.what,
    ):
        """
        图片转pdf

        Parameters
        ---
        pic_iter : typing.Iterable[str | pathlib.Path]
            图片列表
        pdf_name : str | pathlib.Path
            输出的pdf名字(pdf保存在图片文件夹中)
        filter_ : typing.Callable
            过滤器
        """
        pic_iter = filter(filter_, pic_iter)
        with _fitz.Document() as doc:
            for img in pic_iter:
                imgdoc: _fitz.Document = _fitz.Document(img)  # 打开图片
                pdfbytes = imgdoc.convert_to_pdf()  # 使用图片创建单页的 PDF
                imgpdf = _fitz.open("pdf", pdfbytes)
                doc.insert_pdf(imgpdf)  # 将当前页插入文档

            pdf_name = _pathlib.Path(pdf_name)

            # 保存在图片文件夹下
            pdf_name.parent.mkdir(exist_ok=True, parents=True)
            doc.save(pdf_name)  # 保存pdf文件

    @staticmethod
    def extra_pdf(
        resource_pdf_path: _typing.Union[str, _pathlib.Path],
        output_pdf_path: _typing.Union[str, _pathlib.Path] = None,
        pattern: int | _typing.Iterable[int] = None,
    ):
        """
        从pdf中导出指定页

        Parameters
        ---
        resource_pdf_path : str | pathlib.Path
            原始pdf
        output_pdf_path: str | pathlib.Path
            输出路径(默认为原始pdf的名称加上"_extracted")
        pattern: int | Iterable[int]
            页数(以0开始)或页数序列
        Returns
        ---
        pathlib.Path
            输出路径
        """
        resource_pdf_path = _pathlib.Path(resource_pdf_path)
        output_pdf_path = (
            _pathlib.Path(output_pdf_path)
            if output_pdf_path is not None
            else _ts.Path.avoid_exist_path(
                resource_pdf_path.with_stem(resource_pdf_path.stem + "_extracted")
            )
        )
        resouce_pdf = _fitz.Document(resource_pdf_path)
        if pattern is None:
            pattern = range(len(resouce_pdf))
        elif isinstance(pattern, int):
            pattern = (pattern,)

        newpdf = _fitz.Document()
        for page_num in pattern:
            newpdf.insert_pdf(resouce_pdf, page_num, page_num)

        newpdf.save(output_pdf_path)
        return output_pdf_path

    @staticmethod
    def combine_pdf(
        origin_pdf_iter: _typing.Iterable[_typing.Union[str, _pathlib.Path]],
        output_pdf_path: _typing.Union[str, _pathlib.Path],
    ):
        """
        合并pdf

        Parameters
        ---
        origin_pdf_iter : typing.Iterable[str | pathlib.Path]]
            原始pdf文件列表
        output_pdf_path : str | _pathlib.Path
            输出pdf文件
        """
        origin_pdf_iter = [_pathlib.Path(i) for i in origin_pdf_iter]
        output_pdf_path = _pathlib.Path(output_pdf_path)
        output_pdf = _fitz.Document()
        for i in origin_pdf_iter:
            pdf_tobe_insert = _fitz.Document(i)
            output_pdf.insert_pdf(pdf_tobe_insert)
        output_pdf.save(output_pdf_path)

    @classmethod
    def split_pdf(
        cla,
        resource_pdf_path: _typing.Union[str, _pathlib.Path],
        output_folder: _typing.Union[str, _pathlib.Path] = None,
    ) -> list[_pathlib.Path]:
        """
        将pdf切割为单页

        Parameters
        ---
        resource_pdf_path : str | pathlib.Path
            原始pdf
        output_folder: str | pathlib.Path
            输出文件夹
        Returns
        ---
        list[pathlib.Path]
            输出路径
        """
        resource_pdf_path = _pathlib.Path(resource_pdf_path)
        output_folder = (
            _pathlib.Path(output_folder)
            if output_folder is not None
            else resource_pdf_path.parent
        )
        output_folder.mkdir(exist_ok=True, parents=True)

        pdf = _fitz.Document(resource_pdf_path)
        output_pdf_path_list = []
        for page_num in range(len(pdf)):
            newpdf = _fitz.Document()
            newpdf.insert_pdf(pdf, page_num, page_num)

            output_pdf_path = (
                output_folder / f"{resource_pdf_path.stem}_p{page_num+1}.pdf"
            )

            newpdf.save(output_pdf_path)
            output_pdf_path_list.append(output_pdf_path)
        return output_pdf_path_list

    @staticmethod
    def pdf_to_a4(origin_pdf: _pathlib.Path, output_pdf: _pathlib.Path = None):
        # TODO可用, 但测试中
        origin_pdf = _pathlib.Path(origin_pdf)
        output_pdf = output_pdf or origin_pdf.with_stem(f"{origin_pdf.stem}_A4")

        src = _fitz.Document(origin_pdf)
        doc = _fitz.Document()
        for ipage in src:
            rotation = ipage.rotation
            if rotation in {90, 270}:
                fmt = _fitz.paper_rect("a4-l")
                ipage.set_rotation(0)
            else:
                fmt = _fitz.paper_rect("a4")
            page: _fitz.Page = doc.new_page(width=fmt.width, height=fmt.height)
            page.show_pdf_page(page.rect, src, ipage.number)
            page.set_rotation(rotation)
        src.close()
        doc.save(output_pdf)


class Py7zr:
    """7z压缩包相关(无需初始化, 调用静态函数即可)"""

    @staticmethod
    def decompression(
        zip_path: str, output_folder: str, password: str = None, **kwargs
    ):
        """
        7z解压
        """
        password = password if password else None
        with _py7zr.SevenZipFile(zip_path, password=password, mode="r", **kwargs) as z:
            z.extractall(output_folder)

    @staticmethod
    def compression(
        zip_path: _pathlib.Path,
        input_folder: _pathlib.Path,
        password: str = None,
        **kwargs,
    ):
        """
        7z压缩——默认无压缩。若有密码则使用AES256且加密文件名。
        """
        password = password if password else None

        zip_path, input_folder = map(_pathlib.Path, (zip_path, input_folder))
        if password:
            crypyto_kwargs = {
                "header_encryption": True,
                "filters": [
                    {"id": _py7zr.FILTER_COPY},
                    {"id": _py7zr.FILTER_CRYPTO_AES256_SHA256},
                ],
            }
        else:
            crypyto_kwargs = {
                "header_encryption": False,
                "filters": [{"id": _py7zr.FILTER_COPY}],
            }
        with _py7zr.SevenZipFile(
            zip_path, password=password, mode="w", **crypyto_kwargs, **kwargs
        ) as z:
            z.writeall(input_folder, "")

    @staticmethod
    def check_password(zip_path: str, password: str = None) -> bool:
        """
        检查7z文件是否匹配密码
            判断方式是, 是否能顺利读取压缩包内容
            压缩包无密码时任何密码都正确, 出现其他错误时也会返回False
        """
        try:
            zipfile = _py7zr.SevenZipFile(zip_path, password=password, mode="r")
            zipfile.close()
            return True
        except Exception:
            return False

    @staticmethod
    def test(zip_path: str, password: str = None):
        """测试压缩包中各个文件的CRC值"""
        password = password if password else None
        with _py7zr.SevenZipFile(zip_path, password=password, mode="r") as z:
            return z.test()
