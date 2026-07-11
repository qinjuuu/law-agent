"""
邮件发送工具模块
通过 SMTP 发送维权文档（投诉信、审查报告等）到指定邮箱
支持附件发送，发送前需用户确认

确认机制:
1. Agent 首次调用时 confirm=False，工具返回邮件预览信息，不实际发送
2. Agent 将预览展示给用户，询问用户是否确认发送
3. 用户确认后，Agent 再次调用工具 confirm=True，实际发送邮件
"""
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import Optional, List

from langchain.tools import tool

from config import (
    SMTP_SERVER,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASSWORD,
    SMTP_ENABLED,
    WORD_REPORTS_DIR,
)


@tool
def send_email(
    to_email: str,
    subject: str,
    body: str,
    attachments: Optional[List[str]] = None,
    confirm: bool = False,
) -> str:
    """
    发送邮件（支持附件）— 发送前必须获得用户确认

    参数:
        to_email: 收件人邮箱地址，如 "consumer@example.com"
        subject: 邮件主题
        body: 邮件正文（纯文本）
        attachments: 附件文件名列表（仅文件名，如 ["complaint_20260710.docx"]），
                     文件需位于 workspace/reports/ 目录下。可选参数。
        confirm: 是否确认发送。必须为 True 才会实际发送邮件。
                 首次调用时设为 False，工具会返回邮件预览信息供用户确认。

    返回:
        confirm=False 时: 返回邮件预览信息，提示用户确认
        confirm=True 时: 返回发送结果（成功/失败）

    确认流程:
        1. 首次调用 confirm=False → 返回邮件预览，包含收件人、主题、正文摘要、附件列表
        2. Agent 将预览展示给用户，询问"是否确认发送？"
        3. 用户回复确认后，Agent 再次调用，confirm=True → 实际发送邮件

    适用场景:
        - 将生成的投诉信 Word 文档通过邮件发送给商家或监管机构
        - 将审查报告发送给用户自己留存
        - 将维权材料包通过邮件发送
    """
    # 检查 SMTP 配置
    if not SMTP_ENABLED:
        return (
            "邮件功能未启用：SMTP 配置不完整。\n"
            "请在 .env 文件中配置以下变量：\n"
            "  SMTP_SERVER (如 smtp.qq.com)\n"
            "  SMTP_PORT (如 587)\n"
            "  SMTP_USER (发件人邮箱)\n"
            "  SMTP_PASSWORD (邮箱授权码)\n"
            "配置完成后重启系统即可使用邮件功能。"
        )

    # 校验收件人邮箱
    if not to_email or "@" not in to_email:
        return f"收件人邮箱地址无效: {to_email}"

    # 处理附件路径（带安全校验，防止路径越界读取任意文件）
    attachment_paths = []
    if attachments:
        for fname in attachments:
            # 安全校验：禁止包含 .. 或绝对路径
            if ".." in fname or os.path.isabs(fname):
                return f"附件路径不合法: {fname}（仅允许 workspace/reports/ 目录下的文件）"
            # 从 WORD_REPORTS_DIR 查找
            path_in_reports = os.path.normpath(os.path.join(WORD_REPORTS_DIR, fname))
            # 严格检查路径不越界
            root_norm = os.path.normpath(WORD_REPORTS_DIR)
            if os.path.commonpath([path_in_reports, root_norm]) != root_norm:
                return f"附件路径越界: {fname}"
            if os.path.exists(path_in_reports):
                attachment_paths.append(path_in_reports)
            else:
                return f"附件文件不存在: {fname}（请在 workspace/reports/ 目录下查找）"

    # ========== 未确认状态：返回预览 ==========
    if not confirm:
        preview = f"""📧 邮件预览（尚未发送）
══════════════════════════════════
发件人: {SMTP_USER}
收件人: {to_email}
主题: {subject}
附件: {', '.join(attachments) if attachments else '无'}
──────────────────────────────────
正文预览（前200字）:
{body[:200]}{'...' if len(body) > 200 else ''}
══════════════════════════════════
⚠️ 邮件尚未发送！
请向用户展示以上预览信息，询问用户是否确认发送。
用户确认后，请将 confirm 参数设为 True 再次调用本工具完成发送。"""
        return preview

    # ========== 已确认：实际发送邮件 ==========
    try:
        # 构建邮件
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = to_email
        msg["Subject"] = subject

        # 邮件正文
        msg.attach(MIMEText(body, "plain", "utf-8"))

        # 添加附件
        for att_path in attachment_paths:
            with open(att_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition",
                f'attachment; filename= "{os.path.basename(att_path)}"',
            )
            msg.attach(part)

        # 发送
        # 端口465用 SMTP_SSL（隐式SSL），端口587等用 SMTP+STARTTLS（显式SSL）
        context = ssl.create_default_context()
        if SMTP_PORT == 465:
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=context)
        else:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
        with server:
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, to_email, msg.as_string())

        att_info = f"，附件 {len(attachment_paths)} 个" if attachment_paths else ""
        return (
            f"✅ 邮件发送成功！\n"
            f"发件人: {SMTP_USER}\n"
            f"收件人: {to_email}\n"
            f"主题: {subject}{att_info}"
        )

    except smtplib.SMTPAuthenticationError:
        return (
            "❌ 邮件发送失败：SMTP 认证失败。\n"
            "请检查 SMTP_USER 和 SMTP_PASSWORD（授权码）是否正确。\n"
            "QQ邮箱授权码在 设置→账户→POP3/SMTP服务 中获取。"
        )
    except smtplib.SMTPException as e:
        return f"❌ 邮件发送失败（SMTP错误）: {str(e)}"
    except Exception as e:
        return f"❌ 邮件发送失败: {str(e)}"
