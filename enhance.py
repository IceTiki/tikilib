# 标准库
import io as _io
import os as _os
import sys as _sys
import imp as _imp
import codecs as _codecs
import traceback as _traceback
import pathlib as _pathlib


def enhance_init(work_dir=__file__, python_version_require=0, check_module_list=[]):
    """
    :param pythonVersionRequire(int): 最低python所需版本
    :param checkModuleList: 示例: ("requests", "requests_toolbelt", "urllib3", "bs4", "Crypto", "pyDes", "yaml", "lxml", "rsa")
    """
    # ==========检查python版本==========
    if not (
        _sys.version_info[0] == 3 and _sys.version_info[1] >= python_version_require
    ):
        raise Exception(
            "!!!!!!!!!!!!!!Python版本错误!!!!!!!!!!!!!!\n请使用python3.%d及以上版本，而不是[python %s]"
            % (python_version_require, _sys.version)
        )
    # ==========环境变量初始化==========
    try:
        print("==========开始初始化==========")
    except UnicodeEncodeError:
        # 设置默认输出编码为utf-8, 但是会影响腾讯云函数日志输出。
        _sys.stdout = _codecs.getwriter("utf-8")(_sys.stdout.detach())
        print("==========开始初始化(utf-8输出)==========")
    script_abs_path = _pathlib.Path(work_dir).absolute().parent
    _os.chdir(script_abs_path)  # 将工作路径设置为脚本位置
    if _os.name == "posix":
        # 如果是linux系统, 增加TZ环境变量
        _os.environ["TZ"] = "Asia/Shanghai"
    _sys.path.append(script_abs_path)  # 将脚本路径加入模块搜索路径
    # ==========检查第三方模块==========
    try:
        for i in check_module_list:
            _imp.find_module(i)
    except ImportError as e:  # 腾讯云函数在初始化过程中print运作不正常，所以将信息丢入异常中
        raise ImportError(
            f"""!!!!!!!!!!!!!!缺少第三方模块(依赖)!!!!!!!!!!!!!!
    请使用pip3命令安装或者手动将依赖拖入文件夹
    错误信息: [{e}]"""
        )


class Decorators:
    @staticmethod
    def catch_exception(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception:
                print(_traceback.format_exc())
                return None

        return wrapper


class FileOut:
    """
    代替stdout和stderr, 使print同时输出到文件和终端中。
    start()方法可以直接用自身(self)替换stdout和stderr
    close()方法可以还原stdout和stderr
    """

    stdout = _sys.stdout
    stderr = _sys.stderr
    log: str = ""  # 同时将所有输出记录到log字符串中
    logFile: _io.TextIOWrapper = None

    @classmethod
    def set_file_out(cla, path: str = None):
        """
        设置日志输出文件
        :params path: 日志输出文件路径, 如果为空则取消日志文件输出
        """
        # 关闭旧文件
        if cla.logFile:
            cla.logFile.close()
            cla.logFile = None

        # 更新日志文件输出
        if path:
            try:
                path = _os.path.abspath(path)
                logDir = _os.path.dirname(path)
                if not _os.path.isdir(logDir):
                    _os.makedirs(logDir)
                cla.logFile = open(path, "w+", encoding="utf-8")
                cla.logFile.write(cla.log)
                cla.logFile.flush()
                return
            except Exception as e:
                print(2, f"设置日志文件输出失败, 错误信息: [{e}]")
                cla.logFile = None
                return
        else:
            cla.logFile = None
            return

    @classmethod
    def start(cla):
        """开始替换stdout和stderr"""
        if type(_sys.stdout) != cla and type(_sys.stderr) != cla:
            _sys.stdout = cla
            _sys.stderr = cla
        else:
            print("sysout/syserr已被替换为FileOut")

    @classmethod
    def write(cla, str_):
        r"""
        :params str: print传来的字符串
        :print(s)等价于sys.stdout.write(s+"\n")
        """
        str_ = str(str_)
        cla.log += str_
        if cla.logFile:
            cla.logFile.write(str_)
        cla.stdout.write(str_)
        cla.flush()

    @classmethod
    def flush(cla):
        """刷新缓冲区"""
        cla.stdout.flush()
        if cla.logFile:
            cla.logFile.flush()

    @classmethod
    def close(cla):
        """关闭"""
        if cla.logFile:
            cla.logFile.close()
        cla.log = ""
        _sys.stdout = cla.stdout
        _sys.stderr = cla.stderr
