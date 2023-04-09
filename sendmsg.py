# 标准库
import smtplib
from email.mime.text import MIMEText
from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from urllib import parse
import re

# 第三方库
import requests  # requests


class Smtp:
    """Smtp发送类"""

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        key: str,
        sender: str,
        sender_name: str,
        use_ssl=True,
    ):
        """
        :param host: SMTP的域名
        :param port: SMTP的端口(一般是465或25)
        :param user: 用户名
        :param key: 用户的密钥
        :param sender: 邮件发送者(邮箱)
        :param sender_name: 发送者名称(可以随便填)
        :param use_ssl: 是否使用SSL(一般使用SSL的端口是465, 不使用是25)
        """
        self.host = host
        self.port = port
        self.user = user
        self.key = key
        self.sender = sender
        self.sender_name = sender_name
        self.use_ssl = use_ssl

    @property
    def __is_config_correst(self):
        # 简单检查邮箱地址或API地址是否合法
        for item in [self.host, self.user, self.key, self.sender]:
            if type(item) != str:
                return 0
            if len(item) == 0:
                return 0
            if "*" in item:
                return 0
        return 1

    def send(
        self,
        receivers: list,
        msg: str,
        title: str = "no title",
        subtype: str = "html",
        attachments=(),
    ):  # sourcery skip: remove-unnecessary-cast
        """
        发送邮件
        :param receivers: 邮件接收者列表(邮箱)
        :param msg: 要发送的消息(自动转为字符串类型)
        :param title: 邮件标题(自动转为字符串类型)
        :param subtype: 邮件类型
        :params attachment: 附件元组，形式为((blob二进制文件,filename文件名),(blob,filename),...)
        """
        msg = str(msg)
        title = str(title)
        if not self.__is_config_correst:
            print("邮件配置出错")
            return "邮件配置出错"

        mail = MIMEMultipart()
        # 添加正文
        mail.attach(MIMEText(msg, subtype, "utf-8"))
        # 添加标题
        mail["Subject"] = Header(title, "utf-8")
        # 添加发送者
        mail["From"] = formataddr((self.sender_name, self.sender), "utf-8")
        # 添加附件
        for att_full in attachments:
            att = MIMEText(att_full[0], "base64", "utf-8")
            att["Content-Type"] = "application/octet-stream"
            att["Content-Disposition"] = f'attachment; filename="{att_full[1]}"'
            mail.attach(att)
        # 发送邮件
        smtp_obj = (
            smtplib.SMTP_SSL(self.host, self.port)
            if self.use_ssl
            else smtplib.SMTP(self.host, self.port)
        )
        smtp_obj.login(self.user, self.key)
        smtp_obj.sendmail(self.sender, receivers, mail.as_string())
        print("邮件发送成功")


class Qmsg:
    """Qmsg发送类"""

    def __init__(self, key: str, qq: str, isGroup: bool = False):
        """
        :param key: qmsg密钥
        :param qq: 接收消息的qq(多个qq以","分隔)
        :param isGroup: 接收者是否为群
        """
        self.key = key
        self.qq = qq
        self.isGroup = isGroup

    @property
    def __is_config_correct(self):
        """简单检查配置是否合法"""
        if type(self.key) != str:
            return 0
        elif type(self.qq) != str:
            return 0
        elif not re.match("^[0-9a-f]{32}$", self.key):
            return 0
        elif not re.match("^\d+(,\d+)*$", self.qq):
            return 0
        else:
            return 1

    def send(self, msg):
        """发送消息
        :param msg: 要发送的消息(自动转为字符串类型)"""
        # msg：要发送的信息|消息推送函数
        msg = str(msg)
        # 简单检查配置
        if not self.__is_config_correct:
            print("Qmsg配置错误，信息取消发送")
        else:
            sendtype = "group/" if self.isGroup else "send/"
            res = requests.post(
                url=f"https://qmsg.zendee.cn/{sendtype}{self.key}",
                data={"msg": msg, "qq": self.qq},
            )
            return str(res)


class Pushplus:
    """Pushplus推送类"""

    def __init__(self, parameters: str):
        """
        :param parameters: "xxx"形式的令牌 或者 "token=xxx&topic=xxx&yyy=xxx"形式参数列表
        """
        self.parameters = parameters

    @property
    def __is_config_correct(self):
        # 简单检查邮箱地址或API地址是否合法
        if type(self.parameters) != str:
            return 0
        return 1 if self.parameters else 0

    def send_pushplus(self, msg: str, title: str):
        # sourcery skip: remove-unnecessary-cast
        msg = str(msg)
        title = str(title)
        msg = msg.replace("\n", "</br>")
        if not self.__is_config_correct:
            return "pushplus的令牌填写错误，已取消发送！"

        if "=" in self.parameters:
            params = parse.parse_qs(parse.urlparse(self.parameters).path)
            for k in params:
                params[k] = params[k][0]
            params.update({"title": title, "content": msg})
        else:
            params = {
                "token": self.parameters,
                "title": title,
                "content": msg,
            }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:92.0) Gecko/20100101 Firefox/92.0"
        }
        res = requests.post(
            "https://pushplus.hxtrip.com/send", headers=headers, params=params
        )
        return "发送成功" if res.status_code == 200 else "发送失败"
