# 标准库
import os
import uuid
import shutil
import pathlib


class Path:
    """文件、文件夹、路径相关"""

    @staticmethod
    def traversing_generator(
        path, iter_file=True, topdown=False, path_filter=lambda x: True
    ):
        """
        遍历路径中的文件或文件夹的生成器(生成绝对路径)
        :param path: 遍历的路径
        :param iter_file: True遍历文件|False遍历文件夹
        :param topdown: 是否从根文件夹开始遍历
        :param path_filter: 返回绝对路径之前, 先用该过滤器过滤
            过滤器: 传入绝对路径(pathlib.Path), 传出布尔值(bool)
        :yield pathlib.Path: 绝对路径
        """
        for root, dirs, files in os.walk(path, topdown=topdown):
            root = pathlib.Path(root)
            if iter_file is True:
                for name in files:
                    file_dir: pathlib.Path = root / name
                    file_dir = file_dir.absolute()
                    if path_filter(file_dir):
                        yield file_dir
            if iter_file is False:
                for name in dirs:
                    folder_dir: pathlib.Path = root / name
                    folder_dir = folder_dir.absolute()
                    if path_filter(folder_dir):
                        yield folder_dir

    @staticmethod
    def file_copy(from_path, to_path):
        """文件复制(如果目标路径不存在, 则创建目录)"""
        to_dir = os.path.dirname(to_path)
        if os.path.isfile(to_dir) or os.path.isfile(to_path):
            raise OSError(f"「{to_path}」路径已被占用")
        if not os.path.isdir(to_dir):
            os.makedirs(to_dir)
        shutil.copyfile(from_path, to_path)

    @staticmethod
    def before_write(file_dir: str):
        """在写入文件之前, 检查路径是否存在(如果不存在则创建路径)"""
        file_dir: pathlib.Path = pathlib.Path(file_dir).absolute()
        folder_dir: pathlib.Path = file_dir.parent
        if file_dir.is_dir() or folder_dir.is_file():
            raise OSError(f"「{folder_dir}」路径已被占用")
        if not folder_dir.exists():
            folder_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def transfer_generator(
        cla,
        from_dir,
        to_dir,
        path_filter=lambda x: True,
        iter_file: bool = True,
        mkdir: bool = True,
    ):
        """
        文件夹转换器, 本生成器将不断生成一个元组(path1 : pathlib.Path, path2 : pathlib.Path)
            分别是在from_dir文件夹中路径和在to_dire文件夹中的路径, 两个路径在from_dir和to_dir的相对路径相同
        :param from_dir, to_dir: 蓝本文件夹, 新文件夹
        :param path_filter: 过滤器(传入绝对路径(pathlib.Path), 返回布尔值(bool)), 判断是否处理该蓝本文件夹内的路径
        :param iter_file: True: 遍历文件路径|False: 遍历文件夹路径
        :param mkdir: 当路径不存在时候, 自动创建路径(如果返回文件路径, 则保证文件所在文件夹存在)(如果返回文件夹, 则保证文件夹存在)
        """
        from_dir = pathlib.Path(from_dir).absolute()
        to_dir = pathlib.Path(to_dir).absolute()

        if iter_file:
            for from_file in cla.traversing_generator(
                from_dir, True, path_filter=path_filter
            ):
                from_file = pathlib.Path(from_file)
                to_file = to_dir / from_file.relative_to(from_dir)

                if mkdir:
                    cla.before_write(to_file)

                yield (from_file, to_file)
        else:
            for from_folder in cla.traversing_generator(
                from_dir, False, path_filter=path_filter
            ):
                from_folder = pathlib.Path(from_folder)
                to_folder = to_dir / from_folder.relative_to(from_dir)

                if mkdir:
                    to_folder.mkdir(parents=True, exist_ok=True)

                yield (from_folder, to_folder)


class Device:
    @staticmethod
    def get_device_id():
        return uuid.getnode()

    @staticmethod
    def get_mac_address():
        mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
        mac = ":".join([mac[e : e + 2] for e in range(0, 11, 2)])
        return mac
