"""
文件操作工具模块
包含 create_file / read_file / delete_file / zip_files
全部带 _safe_path 路径越界校验
"""
import os
import sys
import zipfile
from typing import List

from langchain.tools import tool

from config import WORK_ROOT

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def _safe_path(relative_path: str, root: str = WORK_ROOT) -> str:
    """
    确保路径在指定根目录范围内，避免越界

    参数 relative_path 为相对路径
    返回绝对路径
    如果路径越界，抛出 ValueError 异常
    """
    abs_file_path = os.path.normpath(os.path.join(root, relative_path))
    if not abs_file_path.startswith(os.path.normpath(root)):
        raise ValueError(f"路径越界: {relative_path}")
    return abs_file_path


@tool
def create_file(filename: str, content: str = "", confirm: bool = False) -> str:
    """
    在工作目录下创建文件并写入内容

    参数:
        filename: 文件名，仅限工作目录下的相对路径
        content: 文件内容，默认为空字符串
        confirm: 是否确认覆盖已存在的文件

    返回:
        操作结果信息

    适用场景:
        用户需要创建文本文件、保存投诉信草稿时调用
    """
    try:
        file_path = _safe_path(filename)
        if os.path.exists(file_path) and not confirm:
            return f"文件 {filename} 已存在，如需覆盖请设置 confirm 为 True"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"文件已创建: {filename}，内容长度 {len(content)} 字符"
    except ValueError:
        return f"安全问题: {filename}"
    except Exception as e:
        return f"创建失败: {str(e)}"


@tool
def read_file(filename: str) -> str:
    """
    读取工作目录下指定文件的内容

    参数:
        filename: 文件名，仅限工作目录下的文件

    返回:
        文件内容字符串，文件不存在时返回错误信息

    适用场景:
        用户询问文件内容、需查看已保存的文件时调用
    """
    try:
        file_path = _safe_path(filename)
        if not os.path.exists(file_path):
            return f"文件不存在: {filename}"
        if not os.path.isfile(file_path):
            return f"错误: {filename} 不是一个文件"
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except ValueError:
        return f"安全问题: {filename}"
    except Exception as e:
        return f"读取失败: {str(e)}"


@tool
def delete_file(filename: str, confirm: bool = False) -> str:
    """
    删除工作目录下的指定文件

    参数:
        filename: 文件名，仅限工作目录下的文件
        confirm: 是否确认删除，删除不可恢复

    返回:
        操作结果信息

    适用场景:
        用户需要删除文件时调用
    """
    try:
        file_path = _safe_path(filename)
        if not os.path.exists(file_path):
            return f"文件不存在: {filename}"
        if not confirm:
            return f"请确认删除文件: {filename}，操作不可恢复，如确认请将 confirm 设置为 True"
        os.remove(file_path)
        return f"文件删除成功: {filename}"
    except ValueError:
        return f"安全问题: {filename}"
    except Exception as e:
        return f"删除失败: {str(e)}"


@tool
def zip_files(file_list: List[str], output_zip: str) -> str:
    """
    将指定的文件或文件夹打包为 ZIP 压缩包

    参数:
        file_list: 要打包的文件或文件夹名称列表
        output_zip: 输出的 zip 文件名

    返回:
        操作结果信息，包含压缩包路径和文件大小

    适用场景:
        用户需要批量打包维权证据文件、备份资料时调用
    """
    try:
        if not output_zip.endswith(".zip"):
            output_zip += ".zip"
        output_zip = _safe_path(output_zip)

        valid_items = []
        for file in file_list:
            item_path = _safe_path(file)
            if os.path.exists(item_path):
                valid_items.append(item_path)
            else:
                return f"错误: {item_path} 路径不存在"

        if not valid_items:
            return f"错误，没有有效的文件或文件夹: {output_zip}"

        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in valid_items:
                if os.path.isfile(item):
                    zf.write(item, arcname=os.path.basename(item))
                elif os.path.isdir(item):
                    for root, dirs, files in os.walk(item):
                        for file in files:
                            full_path = os.path.join(root, file)
                            arcname = os.path.relpath(full_path, WORK_ROOT)
                            zf.write(full_path, arcname=arcname)

        size = os.path.getsize(output_zip)
        return f"压缩成功: {output_zip}，{size} 字节"
    except ValueError:
        return f"安全问题: {output_zip}"
    except Exception as e:
        return f"压缩失败: {str(e)}"
