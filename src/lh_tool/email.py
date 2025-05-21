# -*- encoding: utf-8 -*-
"""
@File    :   email.py
@Time    :   2024/03/14 10:59:46
@Author  :   lihao57
@Version :   1.0
@Contact :   lihao57@baidu.com
"""

import os
import smtplib
from yacs.config import CfgNode
from email import encoders
from email.header import Header
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.utils import formataddr


class Email:
    """
    Email

    Parameters:
        cfg (dict): config

    Example:
        ```python
        Email(cfg).send()
        ```
    """

    def __init__(self, cfg):
        self.from_name = cfg.from_name
        self.from_addr = cfg.from_addr
        self.password = cfg.password
        self.to_name = cfg.to_name
        self.to_addr = cfg.to_addr
        self.subject = cfg.subject
        self.content = cfg.content
        self.smtp_server = cfg.smtp_server
        self.port = cfg.port
        self.encoding = cfg.encoding
        self.file_list = cfg.file_list

    def send(self):
        # 配置基础信息
        msg = MIMEMultipart()
        msg["From"] = formataddr(
            (
                Header(self.from_name, charset=self.encoding).encode(),
                self.from_addr,
            )
        )
        msg["To"] = formataddr(
            (
                Header(self.to_name, charset=self.encoding).encode(),
                self.to_addr,
            )
        )
        msg["Subject"] = Header(self.subject, charset=self.encoding).encode()
        msg.attach(MIMEText(self.content, _charset=self.encoding))

        # 添加附件
        if self.file_list is not None:
            for filename in self.file_list:
                with open(filename, "rb") as f:
                    _, extension = os.path.splitext(filename)
                    mime = MIMEBase(extension, extension, filename=filename)
                    mime.add_header("Content-Disposition", "attachment", filename=filename)
                    mime.add_header("Content-ID", "<0>")
                    mime.add_header("X-Attachment-Id", "0")
                    mime.set_payload(f.read())
                    encoders.encode_base64(mime)
                    msg.attach(mime)

        # 发送邮件
        server = smtplib.SMTP(self.smtp_server, self.port)
        server.login(self.from_addr, self.password)
        server.sendmail(self.from_addr, [self.to_addr], msg.as_string())
        server.quit()
        print("发送成功")


class CustomEmail(Email):
    """
    CustomEmail

    Parameters:
        from_name (str): from name
        from_addr (str): from address
        password (str): password
        to_name (str): to name
        to_addr (str): to address
        subject (str): subject
        content (str): content

    Example:
        ```python
        CustomEmail(from_name, from_addr, password, to_name, to_addr, subject, content).send()
        ```
    """

    def __init__(
        self,
        from_name,
        from_addr,
        password,
        to_name,
        to_addr,
        subject="",
        content="",
    ):
        cfg = CfgNode()
        cfg.from_name = from_name
        cfg.from_addr = from_addr
        cfg.password = password
        cfg.to_name = to_name
        cfg.to_addr = to_addr
        cfg.subject = subject
        cfg.content = content
        cfg.smtp_server = "smtp.whu.edu.cn"
        cfg.port = 25
        cfg.encoding = "utf-8"
        cfg.file_list = None
        cfg.freeze()

        super().__init__(cfg)
