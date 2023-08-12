# 标准库
import pathlib as __pathlib
import typing as __typing
from dataclasses import dataclass as __dataclass

# 第三方库
import pandas as __pd  # pandas
import fitz as __fitz  # fitz, PyMuPDF
import py7zr as __py7zr  # py7zr
import imghdr as __imghdr


# class LI:
#     '''LazyImport'''
#     pd = None #  pandas
#     @classmethod
#     def __getattr__(self, __name: str):
#         if __name == "pd":
#             import pandas as pd
#             self.pf = pd
#             return pd
#         else:
#             raise AttributeError(__name)


class PandasExcel:
    @__dataclass
    class ExcelData:
        data: __typing.Any
        shape: tuple
        index: list[str]
        row_data: list[list[__typing.Any]]  # 逐行数据(但不包括首行(列标题)):

    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name

    def read(self):
        data = __pd.read_excel(self.file_path, self.sheet_name)
        shape = data.shape  # 形状:tuple(行数:int, 列数:int)
        index = list(data.columns)  # 列标题(首行):list[str]
        rowData = data.values.tolist()
        return self.ExcelData(data, shape, index, rowData)

    def write(self, data, **kwargs):
        kwargs.setdefault("index", False)
        kwargs.setdefault("header", True)
        __pd.DataFrame(data).to_excel(self.file_path, self.sheet_name, **kwargs)


class MuPdf:
    @staticmethod
    def pdf2png(
        pdf_path: __typing.Union[str, __pathlib.Path] = "a.pdf",
        output_folder: __typing.Union[str, __pathlib.Path] = "image",
        zoom: __typing.Union[int, tuple[int, int]] = 2,
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
        pdf_path, output_folder = map(__pathlib.Path, (pdf_path, output_folder))
        name = pdf_path.stem
        pdf = __fitz.Document(pdf_path)
        if isinstance(zoom, int):
            zoom = tuple((zoom, zoom))

        output_folder.mkdir(parents=True, exist_ok=True)

        for page_number, page in enumerate(pdf):
            """逐页遍历"""
            page: __fitz.Page
            mat: __fitz.Matrix = __fitz.Matrix(*zoom).prerotate(0)  # 页面大小属性
            rect: __fitz.Rect = page.rect  # 页面总范围
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
                    clip = __fitz.Rect(
                        delta_width * x_index,
                        delta_height * y_index,
                        delta_width * (x_index + 1),
                        delta_height * (y_index + 1),
                    )  # 裁剪范围(x0, y0, x1, y1)
                    pix: __fitz.Pixmap = page.get_pixmap(matrix=mat, clip=clip)
                    pix.save(
                        output_folder / f"{name}_{page_number}_y{y_index}x{x_index}.png"
                    )

    @staticmethod
    def img2pdf(
        pic_iter: __typing.Iterable[
            __typing.Union[str, __pathlib.Path]
        ] = __pathlib.Path(".").iterdir(),
        pdf_name: __typing.Union[str, __pathlib.Path] = "images.pdf",
        filter_: __typing.Callable = __imghdr.what,
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
        with __fitz.Document() as doc:
            for img in pic_iter:
                imgdoc: __fitz.Document = __fitz.Document(img)  # 打开图片
                pdfbytes = imgdoc.convert_to_pdf()  # 使用图片创建单页的 PDF
                imgpdf = __fitz.open("pdf", pdfbytes)
                doc.insert_pdf(imgpdf)  # 将当前页插入文档

            pdf_name = __pathlib.Path(pdf_name)

            # 保存在图片文件夹下
            pdf_name.parent.mkdir(exist_ok=True, parents=True)
            doc.save(pdf_name)  # 保存pdf文件

    @staticmethod
    def split_pdf(
        origin_pdf: __typing.Union[str, __pathlib.Path],
        output_folder: __typing.Union[str, __pathlib.Path] = None,
    ):
        """
        分割pdf

        Parameters
        ---
        origin_pdf : str | pathlib.Path
            原始pdf
        output_folder: str | pathlib.Path
            输出文件夹(默认为原始pdf的文件夹)
        """
        origin_pdf = __pathlib.Path(origin_pdf)
        output_folder = output_folder or origin_pdf.parent
        output_folder = __pathlib.Path(output_folder)

        pdf = __fitz.Document(origin_pdf)
        output_pdf_path_list = []
        for i in range(len(pdf)):
            newpdf = __fitz.Document()
            newpdf.insert_pdf(pdf, i, i)

            output_pdf_path = output_folder / f"{origin_pdf.stem}_p{i+1}.pdf"
            output_pdf_path_list.append(output_pdf_path)

            newpdf.save(output_pdf_path)
        return output_pdf_path_list

    @staticmethod
    def combine_pdf(
        origin_pdf_iter: __typing.Iterable[__typing.Union[str, __pathlib.Path]],
        output_pdf_path: __typing.Union[str, __pathlib.Path],
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
        origin_pdf_iter = [__pathlib.Path(i) for i in origin_pdf_iter]
        output_pdf_path = __pathlib.Path(output_pdf_path)
        output_pdf = __fitz.Document()
        for i in origin_pdf_iter:
            pdf_tobe_insert = __fitz.Document(i)
            output_pdf.insert_pdf(pdf_tobe_insert)
        output_pdf.save(output_pdf_path)

    @staticmethod
    def pdf_to_a4(origin_pdf: __pathlib.Path, output_pdf: __pathlib.Path = None):
        # TODO可用, 但测试中
        origin_pdf = __pathlib.Path(origin_pdf)
        output_pdf = output_pdf or origin_pdf.with_stem(f"{origin_pdf.stem}_A4")

        src = __fitz.Document(origin_pdf)
        doc = __fitz.Document()
        for ipage in src:
            rotation = ipage.rotation
            if rotation in {90, 270}:
                fmt = __fitz.paper_rect("a4-l")
                ipage.set_rotation(0)
            else:
                fmt = __fitz.paper_rect("a4")
            page: __fitz.Page = doc.new_page(width=fmt.width, height=fmt.height)
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
        with __py7zr.SevenZipFile(zip_path, password=password, mode="r", **kwargs) as z:
            z.extractall(output_folder)

    @staticmethod
    def compression(
        zip_path: __pathlib.Path,
        input_folder: __pathlib.Path,
        password: str = None,
        **kwargs,
    ):
        """
        7z压缩——默认无压缩。若有密码则使用AES256且加密文件名。
        """
        password = password if password else None

        zip_path, input_folder = map(__pathlib.Path, (zip_path, input_folder))
        if password:
            crypyto_kwargs = {
                "header_encryption": True,
                "filters": [
                    {"id": __py7zr.FILTER_COPY},
                    {"id": __py7zr.FILTER_CRYPTO_AES256_SHA256},
                ],
            }
        else:
            crypyto_kwargs = {
                "header_encryption": False,
                "filters": [{"id": __py7zr.FILTER_COPY}],
            }
        with __py7zr.SevenZipFile(
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
            zipfile = __py7zr.SevenZipFile(zip_path, password=password, mode="r")
            zipfile.close()
            return True
        except Exception:
            return False

    @staticmethod
    def test(zip_path: str, password: str = None):
        """测试压缩包中各个文件的CRC值"""
        password = password if password else None
        with __py7zr.SevenZipFile(zip_path, password=password, mode="r") as z:
            return z.test()
