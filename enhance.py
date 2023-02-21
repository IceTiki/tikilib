# 标准库
from io import TextIOWrapper
import os
import sys
import imp
import codecs
import traceback

# 第三方库
import requests

# 本库
from .text import Random


def enhance_init(workDir=__file__, python_version_require=0, check_module_list=[]):
    """
    :param pythonVersionRequire(int): 最低python所需版本
    :param checkModuleList: 示例: ("requests", "requests_toolbelt", "urllib3", "bs4", "Crypto", "pyDes", "yaml", "lxml", "rsa")
    """
    # ==========检查python版本==========
    if not (sys.version_info[0] == 3 and sys.version_info[1] >= python_version_require):
        raise Exception(
            "!!!!!!!!!!!!!!Python版本错误!!!!!!!!!!!!!!\n请使用python3.%d及以上版本，而不是[python %s]"
            % (python_version_require, sys.version)
        )
    # ==========环境变量初始化==========
    try:
        print("==========开始初始化==========")
    except UnicodeEncodeError:
        # 设置默认输出编码为utf-8, 但是会影响腾讯云函数日志输出。
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
        print("==========开始初始化(utf-8输出)==========")
    absScriptDir = os.path.dirname(os.path.abspath(workDir))
    os.chdir(absScriptDir)  # 将工作路径设置为脚本位置
    if os.name == "posix":
        # 如果是linux系统, 增加TZ环境变量
        os.environ["TZ"] = "Asia/Shanghai"
    sys.path.append(absScriptDir)  # 将脚本路径加入模块搜索路径
    # ==========检查第三方模块==========
    try:
        for i in check_module_list:
            imp.find_module(i)
    except ImportError as e:  # 腾讯云函数在初始化过程中print运作不正常，所以将信息丢入异常中
        raise ImportError(
            f"""!!!!!!!!!!!!!!缺少第三方模块(依赖)!!!!!!!!!!!!!!
    请使用pip3命令安装或者手动将依赖拖入文件夹
    错误信息: [{e}]"""
        )


def set_system_proxy(
    http: str = "http://127.0.0.1:7890", https: str = "http://127.0.0.1:7890"
):
    os.environ["http_proxy"] = http
    os.environ["https_proxy"] = https


class Decorators:
    @staticmethod
    def except_all_error(func):
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(traceback.format_exc())

        return new_func


class ExResponse(requests.Response):
    """requests.reqResponse的子类"""

    def __init__(self, res: requests.Response):
        self.__dict__.update(res.__dict__)

    def json(self, *args, **kwargs):
        """当解析失败的时候, 会print出响应内容"""
        try:
            return super(ExResponse, self).json(*args, **kwargs)
        except Exception as e:
            raise Exception(f"响应内容以json格式解析失败({e})，响应内容:\n\n{self.text}")


class ExSession(requests.Session):
    """requests.Session的子类"""

    def request(self, *args, **kwargs):
        """增添了请求的默认超时时间, 将返回值转换为reqResponse"""
        kwargs.setdefault("timeout", (10, 30))
        res = super(ExSession, self).request(*args, **kwargs)
        return ExResponse(res)

    def random_user_agent(self):
        """随机生成User-Agent"""
        self.headers["User-Agent"] = Random.random_user_agents()


class FileOut:
    """
    代替stdout和stderr, 使print同时输出到文件和终端中。
    start()方法可以直接用自身(self)替换stdout和stderr
    close()方法可以还原stdout和stderr
    """

    stdout = sys.stdout
    stderr = sys.stderr
    log: str = ""  # 同时将所有输出记录到log字符串中
    logFile: TextIOWrapper = None

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
                path = os.path.abspath(path)
                logDir = os.path.dirname(path)
                if not os.path.isdir(logDir):
                    os.makedirs(logDir)
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
        if type(sys.stdout) != cla and type(sys.stderr) != cla:
            sys.stdout = cla
            sys.stderr = cla
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
        sys.stdout = cla.stdout
        sys.stderr = cla.stderr
