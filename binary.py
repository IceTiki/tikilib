# 标准库
import pathlib as _pathlib
import typing as _typing
import sqlite3 as _sqlite3

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


class DbOperator(_sqlite3.Connection):
    """
    python中SQL语句特性
    ---
    - 表、列名不能用?占位符
    - select中, 列名如果用""括起来, 就会被识别为字符串值, 返回结果时不会返回列对应的值, 而是该字符串。填入其他值同理。
    """

    SQLITE_KEYWORD_SET = set(
        "ABORT ACTION ADD AFTER ALL ALTER ANALYZE AND AS ASC ATTACH AUTOINCREMENT BEFORE BEGIN BETWEEN BY CASCADE CASE CAST CHECK COLLATE COLUMN COMMIT CONFLICT CONSTRAINT CREATE CROSS CURRENT_DATE CURRENT_TIME CURRENT_TIMESTAMP DATABASE DEFAULT DEFERRABLE DEFERRED DELETE DESC DETACH DISTINCT DROP EACH ELSE END ESCAPE EXCEPT EXCLUSIVE EXISTS EXPLAIN FAIL FOR FOREIGN FROM FULL GLOB GROUP HAVING IF IGNORE IMMEDIATE IN INDEX INDEXED INITIALLY INNER INSERT INSTEAD INTERSECT INTO IS ISNULL JOIN KEY LEFT LIKE LIMIT MATCH NATURAL NO NOT NOTNULL NULL OF OFFSET ON OR ORDER OUTER PLAN PRAGMA PRIMARY QUERY RAISE RECURSIVE REFERENCES REGEXP REINDEX RELEASE RENAME REPLACE RESTRICT RIGHT ROLLBACK ROW SAVEPOINT SELECT SET TABLE TEMP TEMPORARY THEN TO TRANSACTION TRIGGER UNION UNIQUE UPDATE USING VACUUM VALUES VIEW VIRTUAL WHEN WHERE WITH WITHOUT".split(
            " "
        )
    )

    @classmethod
    def check_name_normal(cls, name: str):
        """检查名字仅含[a-zA-Z0-9_]且并非关键字"""
        if name.upper() in cls.SQLITE_KEYWORD_SET:
            return False
        if not _re.match(r"^\w+$", name):
            return False
        return True

    def __init__(
        self,
        database: str | bytes | _os.PathLike[str] | _os.PathLike[bytes],
        *args,
        **kwargs,
    ):
        """
        database: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        timeout: float = ...,
        detect_types: int = ...,
        isolation_level: str | None = ...,
        check_same_thread: bool = ...,
        factory: type[sqlite3.Connection] | None = ...,
        cached_statements: int = ...,
        uri: bool = ...,
        """
        super().__init__(database, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def table_list(self) -> list[str]:
        """
        Returns
        ---
        表名列表
        """
        sentence = "SELECT NAME FROM SQLITE_MASTER WHERE TYPE='table' ORDER BY NAME"  # SQLITE_MASTER不区分大小写, table必须为小写
        return [i[0] for i in self.execute(sentence)]

    def get_table_info(self, tbl_name: str):
        """
        获取表详情

        Warning: 没有对传入参数进行检查, 有sql注入风险
        """
        res = self.execute(f"PRAGMA table_info('{tbl_name}')")
        return list(res.fetchall())

    def try_exe(self, sql: str, parameters: _typing.Iterable = None) -> _sqlite3.Cursor:
        """execute的自动commit版本, 如果出错会自动rollback"""
        try:
            if parameters is None:
                result = self.execute(sql)
            else:
                result = self.execute(sql, parameters)
            self.commit()
            return result
        except Exception as e:
            self.rollback()
            raise e

    def try_exemany(self, sql: str, parameters: _typing.Iterable) -> _sqlite3.Cursor:
        """executemany的自动commit版本, 如果出错会自动rollback"""
        try:
            result = self.executemany(sql, parameters)
            self.commit()
            return result
        except Exception as e:
            self.rollback()
            raise e

    def create_table(self, table: str, columns: list[tuple[str]]) -> _sqlite3.Cursor:
        """
        创建表(如果表已存在, 则不执行创建)

        Warning: 没有对传入参数进行检查, 有sql注入风险

        Parameters
        ---
        table : str
            表名
        columns : list[tuple[str]]
            列属性, 应为(name, type, *constraints)
        """

        def fcolumn(column: tuple[str]):
            column = tuple(column)
            return f"'{column[0]}' " + " ".join(column[1:])

        columns = ",\n".join(map(fcolumn, columns))

        sentence = f"CREATE TABLE IF NOT EXISTS '{table}' ({columns});"
        return self.try_exe(sentence)

    def select(
        self,
        table: str,
        column_name: str | _typing.Iterable[str] = "*",
        clause: str = "",
        parameters=None,
    ) -> _sqlite3.Cursor:
        """
        查询

        Warning: 没有对传入参数进行检查, 有sql注入风险

        Parameters
        ---
        table : str
            表名
        column_name : str | typing.Iterable[str], default = "*"
            列名
        clause : str
            子句(比如WHERE ORDER等)
        parameters : SupportsLenAndGetItem[_AdaptedInputData] | Mapping[str, _AdaptedInputData]
            格式化参数
        """
        if isinstance(column_name, str):
            if column_name == "*":
                pass
            else:
                column_name = f"{column_name}"
        elif isinstance(column_name, _typing.Iterable):
            column_name = ", ".join((f"{i}" for i in column_name))

        sentence = f"""SELECT {column_name} FROM {table} {clause};"""

        if parameters is None:
            return self.execute(sentence)
        return self.execute(sentence, parameters)

    def insert_many(
        self,
        table: str,
        column_name: str | _typing.Iterable[str],
        data: list[tuple[_typing.Any]],
        clause: str = "",
    ) -> _sqlite3.Cursor:
        """
        插入

        Warning: 没有对传入参数进行检查, 有sql注入风险

        Parameters
        ---
        table : str
            表名
        column_name : str | Iterable[str]
            列名
        data : list[tuple[Any]]
            tuple[Any]代表单行数据包装为一个元组
        clause : str
            子句
        """
        if isinstance(column_name, str):
            column_name = f"('{column_name}')"
            placeholder = "(?)"
        elif isinstance(column_name, _typing.Iterable):
            placeholder = "(" + ", ".join(map(lambda x: "?", column_name)) + ")"
            column_name = ", ".join((f"'{i}'" for i in column_name))
            column_name = f"({column_name})"

        sentence = f"INSERT INTO '{table}' {column_name} VALUES {placeholder} {clause};"
        return self.try_exemany(sentence, data)

    _update = "UPDATE table SET column_name1 = ? where column_name2 = ?;"
