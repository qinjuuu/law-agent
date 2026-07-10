"""
配置管理模块
从 .env 读取所有配置，集中管理大模型、RAG、工作目录等参数
"""
import os
import sys

# ============================================================
# HuggingFace 镜像配置（必须在任何 HF 相关 import 之前设置）
# ============================================================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["HF_HUB_OFFLINE"] = "0"

# 清理 sys.path 中可能存在的全局 Python312 用户目录（避免与 venv 冲突）
# 系统 PYTHONPATH 可能指向 Python312 的 site-packages，导致 torch 等库版本冲突
sys.path = [
    p for p in sys.path
    if not ("python312" in p.lower() and "site-packages" in p.lower())
]

from dotenv import load_dotenv

# Windows 控制台默认 GBK，强制使用 UTF-8 避免中文乱码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 加载 .env 文件
load_dotenv()

# ============================================================
# 大模型配置（火山引擎 ARK / OpenAI 兼容接口）
# ============================================================
ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
MODEL_ID = os.getenv("MODEL_ID", "doubao-seed-2-0-pro-260215")

# 嵌入模型配置（用于 RAG 向量化）
# 优先使用本地 sentence-transformers 模型，无需额外 API 调用
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

# ============================================================
# RAG 配置
# ============================================================
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
LAWS_DIR = os.path.join(DATA_DIR, "laws")
CASES_DIR = os.path.join(DATA_DIR, "cases")
TEMPLATES_DIR = os.path.join(DATA_DIR, "templates")
VECTORS_DIR = os.path.join(DATA_DIR, "vectors")

# 分块参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# 检索参数
TOP_K = int(os.getenv("TOP_K", "5"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))

# ============================================================
# 工作目录配置（Agent 文件操作沙箱）
# ============================================================
WORK_ROOT = os.path.join(_PROJECT_ROOT, "workspace", "files")
WORD_REPORTS_DIR = os.path.join(_PROJECT_ROOT, "workspace", "reports")

# ============================================================
# 数据库配置（MySQL）
# ============================================================
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "consumer_rights")

# ============================================================
# SMTP 邮件配置（发送投诉信/审查报告等）
# ============================================================
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.qq.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# 邮件功能是否已配置（所有字段都有值时才启用）
SMTP_ENABLED = bool(SMTP_SERVER and SMTP_PORT and SMTP_USER and SMTP_PASSWORD)

for dir_path in [WORK_ROOT, WORD_REPORTS_DIR, VECTORS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================
# 提示词目录
# ============================================================
PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")

# ============================================================
# 配置检查
# ============================================================
def check_config():
    errors = []
    if not ARK_API_KEY:
        errors.append("ARK_API_KEY 未配置")
    if not ARK_BASE_URL:
        errors.append("ARK_BASE_URL 未配置")
    if not MODEL_ID:
        errors.append("MODEL_ID 未配置")
    if errors:
        raise RuntimeError("配置错误:\n" + "\n".join(errors))
    print("[OK] 配置检查通过")
    print(f"  模型: {MODEL_ID}")
    print(f"  嵌入: {EMBEDDING_MODEL}")
    print(f"  工作目录: {WORK_ROOT}")
    print(f"  报告目录: {WORD_REPORTS_DIR}")


if __name__ == "__main__":
    check_config()
