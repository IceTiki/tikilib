if __name__ == "__main__":
    # 标准库
    import os as _os
    import random as _random

    # 第三方库
    import requests as _requests
else:
    from . import BatchLazyImport

    BatchLazyImport(
        globals(),
        locals(),
        """
    # 标准库
    import os as _os
    import random as _random

    # 第三方库
    import requests as _requests
    """,
    )

# 本库
from . import library as _t_library


class ExResponse(_requests.Response):
    """requests.reqResponse的子类"""

    def __init__(self, res: _requests.Response):
        self.__dict__.update(res.__dict__)

    def json(self, *args, **kwargs):
        """当解析失败的时候, 会print出响应内容"""
        try:
            return super(ExResponse, self).json(*args, **kwargs)
        except Exception as e:
            raise Exception(f"响应内容以json格式解析失败({e})，响应内容:\n\n{self.text}")


class ExSession(_requests.Session):
    """requests.Session的子类"""

    def request(self, *args, **kwargs):
        """增添了请求的默认超时时间, 将返回值转换为reqResponse"""
        kwargs.setdefault("timeout", (10, 30))
        res = super(ExSession, self).request(*args, **kwargs)
        return ExResponse(res)

    def random_user_agent(self):
        """随机生成User-Agent"""
        agents = _t_library.json_data["user_agents"]
        self.headers["User-Agent"] = _random.choice(agents)


def set_system_proxy(
    http: str = "http://127.0.0.1:7890", https: str = "http://127.0.0.1:7890"
):
    _os.environ["http_proxy"] = http
    _os.environ["https_proxy"] = https
