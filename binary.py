# 标准库
import os
import pathlib as _pathlib
from typing import Iterable, Any
from dataclasses import dataclass

# 第三方库
import pandas as pd  # pandas
import fitz  # fitz, PyMuPDF
import py7zr as _py7zr  # py7zr
import imghdr


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
    @dataclass
    class ExcelData:
        data: Any
        shape: tuple
        index: list[str]
        row_data: list  # 逐行数据(但不包括首行(列标题)):list[list]

    def __init__(self, file_path, sheet_name):
        self.file_path = file_path
        self.sheet_name = sheet_name

    def read(self):
        data = pd.read_excel(self.file_path, self.sheet_name)
        shape = data.shape  # 形状:tuple(行数:int, 列数:int)
        index = list(data.columns)  # 列标题(首行):list[str]
        rowData = data.values.tolist()
        return self.ExcelData(data, shape, index, rowData)

    def write(self, data, **kwargs):
        kwargs.setdefault("index", False)
        kwargs.setdefault("header", True)
        pd.DataFrame(data).to_excel(self.file_path, self.sheet_name, **kwargs)


class MuPDF:
    @staticmethod
    def pdf2png(path="a.pdf", zoom=2, division=(1, 1), output="image"):
        """
        将pdf转为png
        :params path: pdf路径
        :params zoom: 放大系数(与图片清晰度有关)
        :params division: 将每一页(水平方向, 垂直方向)分割为n份
        :params output: 输出文件夹
        """
        name = path[: path.rfind(".")]
        pdf = fitz.Document(path)
        zx, zy = zoom, zoom
        if not os.path.isdir(output):
            os.makedirs(output)

        for i, page in enumerate(pdf):
            """逐页遍历"""
            page: fitz.Page
            mat: fitz.Matrix = fitz.Matrix(zx, zy).preRotate(0)  # 页面大小属性
            rect: fitz.Rect = page.rect  # 页面总范围
            b = rect.width
            h = rect.height
            db = b / division[0]
            dh = h / division[1]
            for xb in range(division[0]):
                for yb in range(division[1]):
                    """逐个切图"""
                    print(xb, yb)
                    clip = fitz.Rect(
                        db * xb, dh * yb, db * (xb + 1), dh * (yb + 1)
                    )  # 裁剪范围(x0, y0, x1, y1)
                    pix: fitz.Pixmap = page.getPixmap(matrix=mat, clip=clip)
                    pix.writePNG(os.path.join(output, f"{name}_{i}_y{yb}x{xb}.png"))

    @staticmethod
    def img2pdf(
        pic_iter=_pathlib.Path(".").iterdir(),
        pdf_name="images.pdf",
        filter_=imghdr.what,
    ):
        """
        图片转pdf
        :params pic_iter: 图片列表
        :params pdf_name: 输出的pdf名字(pdf保存在图片文件夹中)
        :params filter_: 过滤器
        """
        pic_iter = filter(filter_, pic_iter)
        with fitz.Document() as doc:
            for img in pic_iter:
                imgdoc: fitz.Document = fitz.Document(img)  # 打开图片
                pdfbytes = imgdoc.convertToPDF()  # 使用图片创建单页的 PDF
                imgpdf = fitz.open("pdf", pdfbytes)
                doc.insertPDF(imgpdf)  # 将当前页插入文档

            pdf_name = _pathlib.Path(pdf_name)

            # 保存在图片文件夹下
            pdf_name.parent.mkdir(exist_ok=True, parents=True)
            doc.save(pdf_name)  # 保存pdf文件

    @staticmethod
    def split_pdf(origin_pdf: _pathlib.Path, output_folder: _pathlib.Path = None):
        """
        分割pdf
        :param origin_pdf: 原始pdf
        :param output_folder: 输出文件夹
        """
        origin_pdf = _pathlib.Path(origin_pdf)
        if output_folder == None:
            output_folder = origin_pdf.parent

        pdf = fitz.Document(origin_pdf)
        output_pdf_path_list = []
        for i in range(len(pdf)):
            newpdf = fitz.Document()
            newpdf.insert_pdf(pdf, i, i)

            output_pdf_path = output_folder / f"{origin_pdf.stem}_p{i+1}.pdf"
            output_pdf_path_list.append(output_pdf_path)

            newpdf.save(output_pdf_path)
        return output_pdf_path_list

    @staticmethod
    def combine_pdf(
        origin_pdf_list: Iterable[_pathlib.Path], output_pdf_path: _pathlib.Path
    ):
        """
        合并pdf
        :param origin_pdf_list: (Iterable)原始pdf文件列表
        :param output_pdf_path: 输出pdf文件
        """
        origin_pdf_list = [_pathlib.Path(i) for i in origin_pdf_list]
        output_pdf_path = _pathlib.Path(output_pdf_path)
        output_pdf = fitz.Document()
        for i in origin_pdf_list:
            pdf_tobe_insert = fitz.Document(i)
            output_pdf.insert_pdf(pdf_tobe_insert)
        output_pdf.save(output_pdf_path)

    @staticmethod
    def pdf_to_a4(origin_pdf: _pathlib.Path, output_pdf: _pathlib.Path = None):
        # TODO可用, 但测试中
        origin_pdf = _pathlib.Path(origin_pdf)
        output_pdf = output_pdf or origin_pdf.with_stem(f"{origin_pdf.stem}_A4")

        src = fitz.Document(origin_pdf)
        doc = fitz.Document()
        for ipage in src:
            rotation = ipage.rotation
            if rotation in {90, 270}:
                fmt = fitz.paper_rect("a4-l")
                ipage.set_rotation(0)
            else:
                fmt = fitz.paper_rect("a4")
            page: fitz.Page = doc.new_page(width=fmt.width, height=fmt.height)
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
    def compression(zip_path: str, input_folder: str, password: str = None, **kwargs):
        """
        7z压缩——默认无压缩。若有密码则使用AES256且加密文件名。
        """
        password = password if password else None
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
            z.writeall(input_folder)

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
