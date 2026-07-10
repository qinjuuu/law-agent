"""
工具模块
供 LangChain Agent 调用的工具函数集合
"""
from tools.file_tools import create_file, read_file, delete_file, zip_files
from tools.word_tools import create_complaint_report, create_review_report
from tools.search_tools import search_law, search_case
from tools.innovation_tools import (
    estimate_compensation,
    generate_evidence_checklist,
    plan_rights_path,
    merchant_tactics_response,
    rights_deadline_reminder,
    multi_platform_complaint,
    package_evidence,
    trap_warning,
    check_merchant_reputation,
)

# 工具列表，供 Agent 注册使用
all_tools = [
    create_file,
    read_file,
    delete_file,
    zip_files,
    create_complaint_report,
    create_review_report,
    search_law,
    search_case,
    estimate_compensation,
    generate_evidence_checklist,
    plan_rights_path,
    merchant_tactics_response,
    rights_deadline_reminder,
    multi_platform_complaint,
    package_evidence,
    trap_warning,
    check_merchant_reputation,
]

__all__ = [
    "create_file",
    "read_file",
    "delete_file",
    "zip_files",
    "create_complaint_report",
    "create_review_report",
    "search_law",
    "search_case",
    "estimate_compensation",
    "generate_evidence_checklist",
    "plan_rights_path",
    "merchant_tactics_response",
    "rights_deadline_reminder",
    "multi_platform_complaint",
    "package_evidence",
    "trap_warning",
    "check_merchant_reputation",
    "all_tools",
]
