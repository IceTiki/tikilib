# 标准库
import smtplib as _smtplib
from email.mime.text import MIMEText as _MIMEText
from email.mime.base import MIMEBase as _MIMEBase
from email.header import Header as _Header
from email.mime.multipart import MIMEMultipart as _MIMEMultipart
from email.utils import formataddr as _formataddr
from email.utils import quote as _quote
from email import encoders as _encoders
from urllib import parse as _parse
import re as _re
import typing as _typing
import secrets as _secrets
import pathlib as _pathlib

# 第三方库
import requests  # requests
from loguru import logger as _logger  # loguru
from lxml import etree as _etree


class SmtpSender:
    """
    Examples
    ---
    ```
    sender = SmtpSender(...)
    mail = SmtpSender.Mail()
    mail.attach_text("hello world!")
    sender.send_mail(mail, receiver_mail=..., title="Title of mail")
    ```
    """

    class SenderConfig:
        def __init__(
            self,
            host: str,
            port: int,
            account: str,
            password: str,
            sender_mail: str,
            sender_name: str,
            use_ssl: bool = True,
        ):
            """
            Parameters
            ---
            host : host
                SMTP服务器域名
            port : int
                SMTP服务器端口 (常用: 465/25)
            account : str
                账号 (一般是邮箱)
            password : str
                密码
            sender_mail : str
                发送使用的邮箱
            sender_name : str
                发送使用的昵称 (一般可以随便填)
            use_ssl : bool
                是否使用SSL(一般使用SSL的端口是465, 不使用是25)
            """
            self.host = host
            self.port = port
            self.account = account
            self.password = password
            self.sender_mail = sender_mail
            self.sender_name = sender_name
            self.use_ssl = use_ssl

            self.smtp_obj: _typing.Union[_smtplib.SMTP, _smtplib.SMTP_SSL] = (
                _smtplib.SMTP_SSL(self.host, self.port)
                if self.use_ssl
                else _smtplib.SMTP(self.host, self.port)
            )

        @property
        def _check_config(self):
            """简单检查邮箱地址或API地址是否合法"""
            for item in [self.host, self.account, self.password, self.sender_mail]:
                if type(item) != str:
                    return 0
            return 1

        def __enter__(self):
            self.login()
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.logout()
            return False

        def login(self):
            self.smtp_obj.login(self.account, self.password)

        def logout(self):
            self.smtp_obj.quit()

        def send(
            self,
            receiver_mail: _typing.Union[str, _typing.Sequence[str]],
            content: _MIMEBase,
        ):
            """
            Parameters
            ---
            receiver_mail : str | Sequence[str]
                接收者邮箱
            content: _MIMEBase
                邮件
            """
            self.login()
            self.smtp_obj.sendmail(self.sender_mail, receiver_mail, content.as_string())
            self.logout()

    class Mail:
        IMG_SUFFIX = {
            "jpg",
            "jpeg",
            "png",
            "gif",
            "bmp",
            "tiff",
            "tif",
            "webp",
            "ico",
            "svg",
            "heif",
            "heic",
            "avif",
            "jfif",
            "raw",
            "psd",
        }

        @staticmethod
        def pack_file(
            file: bytes,
            file_name: _typing.Optional[str] = None,
            content_id: _typing.Optional[str] = None,
        ) -> _typing.Tuple[_MIMEBase, str, str]:
            """
            Returns
            ---
            attachment, file_name, content_id : _MIMEBase, str, str
                - 打包好的file
                - 文件名
                - 供引用的content_id
            """
            file_name = (
                _secrets.token_hex(16) if file_name is None else _quote(file_name)
            )  # quote防止特殊符号
            content_id = _secrets.token_hex(16) if content_id is None else content_id

            attachment = _MIMEBase("application", "octet-stream")
            attachment.set_payload(file)

            _encoders.encode_base64(attachment)

            attachment["Content-Disposition"] = f'attachment; filename="{file_name}"'
            attachment["Content-ID"] = f"<{content_id}>"  # 允许 HTML 引用

            return attachment, file_name, content_id

        @classmethod
        def pack_file_inline(
            cls,
            file: bytes,
            file_name: _typing.Optional[str] = None,
            content_id: _typing.Optional[str] = None,
        ) -> _typing.Tuple[_MIMEBase, str, str]:
            """
            Returns
            ---
            attachment, file_name, content_id : _MIMEBase, str, str
                - 打包好的file
                - 文件名
                - 供引用的content_id
            """
            file_name = (
                _secrets.token_hex(16) if file_name is None else _quote(file_name)
            )  # quote防止特殊符号
            content_id = _secrets.token_hex(16) if content_id is None else content_id

            if len(file_name.split(".")) == 1:
                suffix = ""
            else:
                suffix = file_name.split(".")[-1]

            if suffix in cls.IMG_SUFFIX:
                main_type = "image"
                sub_type = suffix
            else:
                main_type = "application"
                sub_type = "octet-stream"

            attachment = _MIMEBase(main_type, sub_type)
            attachment.set_payload(file)

            _encoders.encode_base64(attachment)

            attachment["Content-Disposition"] = f'inline; filename="{file_name}"'
            attachment["Content-ID"] = f"<{content_id}>"  # 允许 HTML 引用

            return attachment, file_name, content_id

        def __init__(self):
            self.mail = _MIMEMultipart("related")

        def __setitem__(self, *args, **kwargs):
            self.mail.__setitem__(*args, **kwargs)

        def __getitem__(self, *args, **kwargs):
            self.mail.__getitem__(*args, **kwargs)

        def attach_text(self, text: str, subtype: str = "html"):
            self.mail.attach(_MIMEText(text, subtype, "utf-8"))

        def attach(self, item: _MIMEBase):
            self.mail.attach(item)

        def set_header(
            self,
            title: str,
            receiver_mail: _typing.Union[str, _typing.Sequence[str]],
            sender_name: str,
            sender_mail: str,
        ):
            self.mail["Subject"] = _Header(title, "utf-8")
            self.mail["To"] = (
                receiver_mail
                if isinstance(receiver_mail, str)
                else ", ".join(receiver_mail)
            )
            self.mail["From"] = (
                f"{sender_name} <{sender_mail}>"  # !如果<>里面的内容删掉，会隐藏发送者
            )

        @staticmethod
        def _read_local_file(
            root: _pathlib.Path, link: str
        ) -> _typing.Optional[_typing.Tuple[_pathlib.Path, bytes]]:
            """
            Returns
            ---
            tuple[pathlib.Path, bytes] | None
                如果读取文件正常, 返回 (文件路径, 文件bytes)
                否则返回None。
            """
            file_path = _pathlib.Path(link)
            try:
                if not file_path.absolute():
                    file_path = (root.absolute() / file_path).resolve()
                file = file_path.read_bytes()
            except Exception as e:
                _logger.debug(f"read {link} failed.")
                return None

            return (file_path, file)

        def attach_html_with_local_files(
            self, html_path: _typing.Union[str, _pathlib.Path], encoding: str = "utf-8"
        ):
            """
            自动将html中的本地附件 (比如图片) 上传。
            Notes
            ---
            当前解析a的href和img的src, 检查是否指向文件。相对路径的根目录使用html文件的目录。
            """
            html_path = _pathlib.Path(html_path)
            html = html_path.read_text(encoding)
            parser = _etree.HTMLParser()
            tree = _etree.fromstring(html, parser)
            path_cache: _typing.Dict[_pathlib.Path, _typing.Tuple[str, _MIMEBase]] = (
                {}
            )  # content_id, attachment

            def deal_link(link: str):
                res = self._read_local_file(html_path.parent, link)
                if res is None:
                    return link
                file_path, file = res

                # 处理重复引用的文件
                if file_path not in path_cache:
                    attachment, file_name, content_id = self.pack_file_inline(
                        file, file_path.name
                    )
                    path_cache[file_path] = (content_id, attachment)

                content_id, attachment = path_cache[file_path]
                return f"cid:{content_id}"

            # 处理链接
            for a in tree.xpath("//a"):
                href: str = a.get("href", "")
                a.set("href", deal_link(href))
            for img in tree.xpath("//img"):
                src: str = img.get("src", "")
                img.set("src", deal_link(src))

            modified_html = _etree.tostring(tree, pretty_print=True, encoding="utf-8")

            self.attach_text(modified_html, subtype="html")
            for i in path_cache.values():
                content_id, attachment = i
                self.attach(attachment)

    def __init__(
        self,
        host: str,
        port: int,
        account: str,
        password: str,
        sender_mail: str,
        sender_name: str,
        use_ssl: bool = True,
    ):
        """
        Parameters
        ---
        host : host
            SMTP服务器域名
        port : int
            SMTP服务器端口 (常用: 465/25)
        account : str
            账号 (一般是邮箱)
        password : str
            密码
        sender_mail : str
            发送使用的邮箱
        sender_name : str
            发送使用的昵称 (一般可以随便填)
        use_ssl : bool
            是否使用SSL(一般使用SSL的端口是465, 不使用是25)
        """
        self.sender_config = self.SenderConfig(
            host, port, account, password, sender_mail, sender_name, use_ssl
        )

    def send_mail(
        self,
        mail: Mail,
        receiver_mail: _typing.Union[str, _typing.Sequence[str]],
        title: str,
    ):
        """
        Notes
        ---
        会自动设置邮件的From, To, Subject (标题)
        """
        mail.set_header(
            title,
            receiver_mail,
            self.sender_config.sender_name,
            self.sender_config.sender_mail,
        )

        self.sender_config.send(receiver_mail, content=mail.mail)


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
        elif not _re.match("^[0-9a-f]{32}$", self.key):
            return 0
        elif not _re.match("^\\d+(,\\d+)*$", self.qq):
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
            params = _parse.parse_qs(_parse.urlparse(self.parameters).path)
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
