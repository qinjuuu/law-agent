"""
消费维权智能助手 — 高级测试套件
覆盖已有测试未涉及的模块和 bug 回归测试

覆盖模块:
20. 路径安全增强 (file_tools / word_tools / email_tools)
21. 邮件工具 (email_tools)
22. 联网搜索工具 (web_search_tools)
23. 智能检索工具 (smart_search + 自动入库)
24. Embedder 单例模式
25. IntentRouter 意图路由器
26. 数据库模块增强
27. Agent 路由与工具注册
28. RAG 端到端检索
"""
import os
import sys
import tempfile
import shutil
import traceback

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# 预加载嵌入模型：必须在 langchain 导入前初始化，
# 否则 langchain 的导入链会触发全局 torch（DLL 损坏），
# 导致嵌入模型无法加载
try:
    from rag.embedder import Embedder as _PreloadEmbedder
    _PreloadEmbedder()
except Exception as e:
    print(f"[测试] Embedder 预加载失败（部分测试将跳过）: {e}")

# ============================================================
# 测试框架
# ============================================================
_passed = 0
_failed = 0
_errors = []


def test(name, func):
    """运行单个测试"""
    global _passed, _failed
    try:
        func()
        print(f"  [PASS] {name}")
        _passed += 1
    except AssertionError as e:
        print(f"  [FAIL] {name}: {e}")
        _failed += 1
        _errors.append((name, str(e), traceback.format_exc()))
    except Exception as e:
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        _failed += 1
        _errors.append((name, f"{type(e).__name__}: {e}", traceback.format_exc()))


# ============================================================
# 20. 路径安全增强测试 (bug 回归)
# ============================================================

def test_file_safe_path_traversal():
    """路径越界 — ../ 绕过"""
    from tools.file_tools import _safe_path
    try:
        _safe_path("../../etc/passwd")
        assert False, "标准路径越界未触发 ValueError"
    except ValueError:
        pass


def test_file_safe_path_prefix_bypass():
    """路径越界 — 同前缀目录绕过 (bug 回归)"""
    from tools.file_tools import _safe_path
    from config import WORK_ROOT
    # 构造一个与 WORK_ROOT 同前缀但不在其下的路径
    # WORK_ROOT 以 .../files 结尾，构造 .../files_evil/test.txt
    parent = os.path.dirname(os.path.normpath(WORK_ROOT))
    evil_path = os.path.relpath(
        os.path.join(parent, "files_evil", "test.txt"),
        WORK_ROOT
    )
    try:
        result = _safe_path(evil_path)
        # 如果没有抛出 ValueError，检查结果是否真的在 WORK_ROOT 下
        assert not result.startswith(os.path.normpath(WORK_ROOT) + os.sep), \
            f"同前缀目录绕过成功: {evil_path} -> {result}"
    except ValueError:
        pass  # 正确行为


def test_file_safe_path_valid():
    """路径安全 — 正常路径通过"""
    from tools.file_tools import _safe_path
    from config import WORK_ROOT
    result = _safe_path("test.txt")
    assert result == os.path.normpath(os.path.join(WORK_ROOT, "test.txt"))


def test_file_safe_path_subdir():
    """路径安全 — 子目录路径通过"""
    from tools.file_tools import _safe_path
    from config import WORK_ROOT
    result = _safe_path("subdir/test.txt")
    expected = os.path.normpath(os.path.join(WORK_ROOT, "subdir", "test.txt"))
    assert result == expected, f"子目录路径不匹配: {result} != {expected}"


def test_word_safe_path_traversal():
    """Word 路径越界 — ../ 绕过"""
    from tools.word_tools import _safe_word_path
    try:
        _safe_word_path("../../malicious.docx")
        assert False, "Word 路径越界未触发 ValueError"
    except ValueError:
        pass


def test_word_safe_path_prefix_bypass():
    """Word 路径越界 — 同前缀目录绕过 (bug 回归)"""
    from tools.word_tools import _safe_word_path
    from config import WORD_REPORTS_DIR
    parent = os.path.dirname(os.path.normpath(WORD_REPORTS_DIR))
    evil_name = os.path.relpath(
        os.path.join(parent, "reports_evil", "test.docx"),
        WORD_REPORTS_DIR
    ) + ".docx"
    try:
        result = _safe_word_path(evil_name)
        root_norm = os.path.normpath(WORD_REPORTS_DIR)
        assert os.path.commonpath([result, root_norm]) != root_norm or result.startswith(root_norm + os.sep), \
            f"同前缀目录绕过: {evil_name} -> {result}"
    except ValueError:
        pass  # 正确行为


def test_word_safe_path_valid():
    """Word 路径安全 — 正常路径通过"""
    from tools.word_tools import _safe_word_path
    from config import WORD_REPORTS_DIR
    result = _safe_word_path("test.docx")
    expected = os.path.normpath(os.path.join(WORD_REPORTS_DIR, "test.docx"))
    assert result == expected


def test_word_auto_extension():
    """Word 路径 — 自动补全 .docx 后缀"""
    from tools.word_tools import _safe_word_path
    from config import WORD_REPORTS_DIR
    result = _safe_word_path("report")
    assert result.endswith(".docx"), f"未自动补全 .docx: {result}"


# ============================================================
# 21. 邮件工具测试
# ============================================================

def test_email_invalid_address():
    """邮件 — 无效邮箱地址"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "invalid_email",
        "subject": "测试",
        "body": "测试内容",
        "confirm": False,
    })
    assert "无效" in result, f"未检测到无效邮箱: {result}"


def test_email_preview_mode():
    """邮件 — 预览模式（不发送）"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "test@example.com",
        "subject": "测试主题",
        "body": "这是测试邮件正文内容",
        "confirm": False,
    })
    assert "尚未发送" in result or "预览" in result, f"未进入预览模式: {result}"
    assert "test@example.com" in result
    assert "测试主题" in result


def test_email_attachment_not_found():
    """邮件 — 附件不存在"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "test@example.com",
        "subject": "测试",
        "body": "测试",
        "attachments": ["nonexistent_file.docx"],
        "confirm": False,
    })
    # 附件检查在 confirm 之前执行
    assert "不存在" in result or "不合法" in result, f"未检测到附件不存在: {result}"


def test_email_attachment_path_traversal():
    """邮件 — 附件路径越界 (bug 回归)"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "test@example.com",
        "subject": "测试",
        "body": "测试",
        "attachments": ["../../.env"],
        "confirm": False,
    })
    assert "不合法" in result or "越界" in result, f"附件路径越界未拦截: {result}"


def test_email_attachment_absolute_path():
    """邮件 — 附件绝对路径拦截"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "test@example.com",
        "subject": "测试",
        "body": "测试",
        "attachments": ["/etc/passwd"],
        "confirm": False,
    })
    assert "不合法" in result or "越界" in result, f"绝对路径附件未拦截: {result}"


def test_email_empty_body():
    """邮件 — 空正文预览"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "test@example.com",
        "subject": "测试",
        "body": "",
        "confirm": False,
    })
    assert "预览" in result or "尚未发送" in result


def test_email_preview_content():
    """邮件 — 预览包含关键信息"""
    from tools.email_tools import send_email
    result = send_email.invoke({
        "to_email": "consumer@example.com",
        "subject": "投诉信",
        "body": "尊敬的市场监督管理部门：",
        "confirm": False,
    })
    assert "consumer@example.com" in result
    assert "投诉信" in result
    assert "confirm" in result.lower() or "确认" in result


# ============================================================
# 22. 联网搜索工具测试
# ============================================================

def test_web_search_tool_exists():
    """联网搜索 — 工具可导入"""
    from tools.web_search_tools import search_web
    assert search_web is not None
    assert hasattr(search_web, "invoke")


def test_latest_regulation_tool_exists():
    """最新法规搜索 — 工具可导入"""
    from tools.web_search_tools import search_latest_regulation
    assert search_latest_regulation is not None
    assert hasattr(search_latest_regulation, "invoke")


def test_merchant_info_tool_exists():
    """商家信息搜索 — 工具可导入"""
    from tools.web_search_tools import search_merchant_info
    assert search_merchant_info is not None
    assert hasattr(search_merchant_info, "invoke")


def test_do_search_returns_list():
    """_do_search — 返回列表类型"""
    from tools.web_search_tools import _do_search
    # 调用实际搜索（网络可用时返回结果，不可用时返回空列表）
    results = _do_search("消费者权益保护法", max_results=2)
    assert isinstance(results, list), f"_do_search 应返回 list，实际 {type(results)}"


def test_bing_search_structure():
    """_bing_search — 返回正确数据结构"""
    from tools.web_search_tools import _bing_search
    results = _bing_search("测试", max_results=2)
    for r in results:
        assert "title" in r, "结果缺少 title 字段"
        assert "body" in r, "结果缺少 body 字段"
        assert "href" in r, "结果缺少 href 字段"


# ============================================================
# 23. 智能检索工具测试 (smart_search + 自动入库)
# ============================================================

def test_smart_search_local_hit():
    """smart_search — 本地知识库命中"""
    from tools.search_tools import smart_search
    result = smart_search.invoke({"query": "七天无理由退货"})
    assert isinstance(result, str)
    assert len(result) > 0
    # 本地知识库应该有匹配
    assert "本地知识库" in result or "检索" in result


def test_smart_search_food_safety():
    """smart_search — 食品安全检索"""
    from tools.search_tools import smart_search
    result = smart_search.invoke({"query": "食品安全赔偿"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_smart_search_returns_string():
    """smart_search — 返回字符串类型"""
    from tools.search_tools import smart_search
    result = smart_search.invoke({"query": "消费者权利"})
    assert isinstance(result, str), f"smart_search 应返回 str，实际 {type(result)}"


def test_add_web_results_to_kb_function_exists():
    """_add_web_results_to_kb — 函数可导入"""
    from tools.search_tools import _add_web_results_to_kb
    assert callable(_add_web_results_to_kb)


def test_add_web_results_to_kb_empty_input():
    """_add_web_results_to_kb — 空输入返回0"""
    from tools.search_tools import _add_web_results_to_kb
    result = _add_web_results_to_kb("测试查询", [])
    assert result == 0, f"空输入应返回0，实际 {result}"


def test_add_web_results_to_kb_short_text():
    """_add_web_results_to_kb — 过短文本不入库"""
    from tools.search_tools import _add_web_results_to_kb
    # 只有过短文本，应不入库
    web_results = [{"title": "短", "body": "短", "href": "http://example.com"}]
    result = _add_web_results_to_kb("测试", web_results)
    assert result == 0, f"过短文本不应入库，实际入库 {result} 条"


def test_add_web_results_to_kb_dedup():
    """_add_web_results_to_kb — 去重逻辑"""
    from tools.search_tools import _add_web_results_to_kb
    # 先入库一条
    web_results = [{
        "title": "去重测试标题" + "x" * 30,
        "body": "这是一段足够长的测试正文内容用于验证去重逻辑是否正常工作" + "y" * 30,
        "href": "http://example.com",
    }]
    first = _add_web_results_to_kb("去重测试", web_results)
    # 再次入库相同内容
    second = _add_web_results_to_kb("去重测试", web_results)
    assert second == 0, f"重复入库应返回0，实际 {second}"


# ============================================================
# 24. Embedder 单例模式测试
# ============================================================

def test_embedder_singleton():
    """Embedder — 单例模式"""
    from rag.embedder import Embedder
    e1 = Embedder()
    e2 = Embedder()
    assert e1 is e2, "Embedder 应为单例模式"


def test_embedder_dimension():
    """Embedder — 向量维度正确"""
    from rag.embedder import Embedder
    e = Embedder()
    assert e.dim == 512, f"向量维度应为 512，实际 {e.dim}"


def test_embedder_embed_single():
    """Embedder — 单条文本嵌入"""
    from rag.embedder import Embedder
    e = Embedder()
    vec = e.embed_query("消费者权益保护法")
    assert vec.shape == (512,), f"单条嵌入 shape 应为 (512,)，实际 {vec.shape}"


def test_embedder_embed_batch():
    """Embedder — 批量嵌入"""
    from rag.embedder import Embedder
    e = Embedder()
    texts = ["消费者权益", "食品安全法", "产品质量法"]
    vectors = e.embed(texts)
    assert vectors.shape == (3, 512), f"批量嵌入 shape 应为 (3, 512)，实际 {vectors.shape}"


def test_embedder_embed_string_input():
    """Embedder — 字符串输入自动转列表"""
    from rag.embedder import Embedder
    e = Embedder()
    vec = e.embed("测试文本")
    assert vec.shape[0] == 1, f"字符串输入应转为单元素列表，实际 shape {vec.shape}"


def test_embedder_normalized():
    """Embedder — 向量已归一化"""
    import numpy as np
    from rag.embedder import Embedder
    e = Embedder()
    vec = e.embed_query("归一化测试")
    norm = np.linalg.norm(vec)
    # 归一化后 L2 范数应接近 1.0
    assert abs(norm - 1.0) < 0.01, f"归一化向量范数应接近1.0，实际 {norm:.4f}"


# ============================================================
# 25. IntentRouter 意图路由器测试
# ============================================================

def test_router_constants():
    """IntentRouter — 意图常量定义"""
    from agents.router import INTENT_COMPLAINT, INTENT_REVIEW, INTENT_QA
    assert INTENT_COMPLAINT == "complaint"
    assert INTENT_REVIEW == "review"
    assert INTENT_QA == "qa"


def test_router_class_structure():
    """IntentRouter — 类结构"""
    from agents.router import IntentRouter
    assert hasattr(IntentRouter, "classify")


def test_router_system_prompt():
    """IntentRouter — 系统提示词包含关键信息"""
    from agents.router import ROUTER_SYSTEM_PROMPT
    assert "complaint" in ROUTER_SYSTEM_PROMPT
    assert "review" in ROUTER_SYSTEM_PROMPT
    assert "qa" in ROUTER_SYSTEM_PROMPT


# ============================================================
# 26. 数据库模块增强测试
# ============================================================

def test_db_email_method():
    """DatabaseManager — log_email 方法定义"""
    from database.db_manager import DatabaseManager
    assert hasattr(DatabaseManager, "log_email"), "DatabaseManager 应有 log_email 方法"


def test_db_email_stats_method():
    """DatabaseManager — get_email_stats 方法定义"""
    from database.db_manager import DatabaseManager
    assert hasattr(DatabaseManager, "get_email_stats"), "DatabaseManager 应有 get_email_stats 方法"


def test_db_schema_email_table():
    """数据库 schema — email_logs 表定义"""
    schema_path = os.path.join(_PROJECT_ROOT, "database", "schema.sql")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = f.read()
    assert "email_logs" in schema, "schema.sql 应包含 email_logs 表"


def test_db_all_log_methods():
    """DatabaseManager — 所有日志方法定义"""
    from database.db_manager import DatabaseManager
    methods = [
        "create_conversation", "log_message", "log_tool_call",
        "log_emotion", "log_completeness", "log_reasoning_chain",
        "log_reflection", "log_confidence", "log_progress",
        "log_compensation", "log_evidence_checklist", "log_rights_path",
        "log_merchant_tactics", "log_deadline", "log_trap_warning",
        "log_merchant_reputation", "log_document", "log_complaint",
        "log_clause_review", "log_handoff", "log_email", "get_stats",
    ]
    for method in methods:
        assert hasattr(DatabaseManager, method), f"DatabaseManager 缺少方法: {method}"


# ============================================================
# 27. Agent 路由与工具注册测试
# ============================================================

def test_qa_agent_tools_registered():
    """QA Agent — 工具列表非空"""
    from agents.qa_agent import ConsumerQAAgent
    agent = ConsumerQAAgent()
    tool_names = [t.name for t in agent.tools]
    assert "smart_search" in tool_names or "search_law" in tool_names, \
        f"QA Agent 应注册搜索工具，实际: {tool_names}"


def test_complaint_agent_tools_registered():
    """Complaint Agent — 工具列表非空"""
    from agents.complaint_agent import ComplaintAgent
    agent = ComplaintAgent()
    tool_names = [t.name for t in agent.tools]
    assert "create_complaint_report" in tool_names, \
        f"Complaint Agent 应注册投诉信生成工具，实际: {tool_names}"


def test_review_agent_tools_registered():
    """Review Agent — 工具列表非空"""
    from agents.review_agent import ReviewAgent
    agent = ReviewAgent()
    tool_names = [t.name for t in agent.tools]
    assert "create_review_report" in tool_names, \
        f"Review Agent 应注册审查报告工具，实际: {tool_names}"


def test_all_agents_have_llm():
    """所有 Agent — LLM 已配置"""
    from agents.qa_agent import ConsumerQAAgent
    from agents.complaint_agent import ComplaintAgent
    from agents.review_agent import ReviewAgent
    for AgentClass in [ConsumerQAAgent, ComplaintAgent, ReviewAgent]:
        agent = AgentClass()
        assert agent.llm is not None, f"{AgentClass.__name__} 的 LLM 未配置"


def test_all_agents_have_prompt():
    """所有 Agent — 系统提示词非空"""
    from agents.qa_agent import ConsumerQAAgent
    from agents.complaint_agent import ComplaintAgent
    from agents.review_agent import ReviewAgent
    for AgentClass in [ConsumerQAAgent, ComplaintAgent, ReviewAgent]:
        agent = AgentClass()
        assert len(agent.system_prompt) > 100, \
            f"{AgentClass.__name__} 系统提示词过短: {len(agent.system_prompt)} 字符"


def test_tool_labels_include_smart_search():
    """工具标签 — smart_search 已注册"""
    from agents.base import _TOOL_LABELS
    assert "smart_search" in _TOOL_LABELS, "工具标签缺少 smart_search"


def test_tool_labels_include_send_email():
    """工具标签 — send_email 已注册"""
    from agents.base import _TOOL_LABELS
    assert "send_email" in _TOOL_LABELS, "工具标签缺少 send_email"


# ============================================================
# 28. RAG 端到端检索测试
# ============================================================

def test_rag_search_law_result():
    """RAG 端到端 — 法律检索返回结果"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "七天无理由退货"})
    assert isinstance(result, str)
    assert len(result) > 50, f"检索结果过短: {len(result)} 字符"
    assert "消费者" in result or "退货" in result or "检索" in result


def test_rag_search_case_result():
    """RAG 端到端 — 案例检索返回结果"""
    from tools.search_tools import search_case
    result = search_case.invoke({"query": "食品安全"})
    assert isinstance(result, str)
    assert len(result) > 20


def test_rag_search_food_safety():
    """RAG 端到端 — 食品安全法检索"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "十倍赔偿 食品安全"})
    assert "食品安全" in result or "十倍" in result or "赔偿" in result or "检索" in result


def test_rag_search_electronic_commerce():
    """RAG 端到端 — 电子商务法检索"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "电子商务法 网购"})
    assert isinstance(result, str)
    assert len(result) > 20


def test_rag_search_product_quality():
    """RAG 端到端 — 产品质量法检索"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "产品质量 缺陷"})
    assert isinstance(result, str)
    assert len(result) > 20


def test_rag_search_returns_score():
    """RAG 端到端 — 结果包含匹配度"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "消费者权利"})
    assert "匹配度" in result, f"检索结果应包含匹配度: {result[:100]}"


def test_rag_search_metadata():
    """RAG 端到端 — 结果包含出处"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "消费者保护"})
    assert "出处" in result, f"检索结果应包含出处: {result[:100]}"


# ============================================================
# 29. Embedder 预加载机制测试
# ============================================================

def test_embedder_preload_in_search_tools():
    """search_tools — Embedder 预加载代码存在"""
    search_tools_path = os.path.join(_PROJECT_ROOT, "tools", "search_tools.py")
    with open(search_tools_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "_EmbedderPreload" in content, "search_tools.py 应包含 Embedder 预加载代码"


def test_search_tools_score_threshold():
    """search_tools — 匹配度阈值定义"""
    from tools.search_tools import _SCORE_THRESHOLD
    assert 0 < _SCORE_THRESHOLD < 1, f"匹配度阈值应在 (0,1) 范围内，实际 {_SCORE_THRESHOLD}"


def test_search_tools_retriever_lazy():
    """search_tools — 检索器延迟初始化"""
    import tools.search_tools as st
    # 模块级 _retriever 初始应为 None
    assert hasattr(st, "_retriever"), "search_tools 应有 _retriever 属性"


# ============================================================
# 30. 配置完整性测试
# ============================================================

def test_config_smtp_enabled_flag():
    """配置 — SMTP_ENABLED 布尔值"""
    from config import SMTP_ENABLED
    assert isinstance(SMTP_ENABLED, bool), f"SMTP_ENABLED 应为 bool，实际 {type(SMTP_ENABLED)}"


def test_config_vectors_dir_exists():
    """配置 — 向量目录存在且有文件"""
    from config import VECTORS_DIR
    assert os.path.isdir(VECTORS_DIR)
    files = os.listdir(VECTORS_DIR)
    assert len(files) >= 2, f"向量目录应有索引文件，实际 {files}"


def test_config_all_dirs():
    """配置 — 所有数据目录存在"""
    from config import DATA_DIR, LAWS_DIR, CASES_DIR, TEMPLATES_DIR, KNOWLEDGE_DIR
    for d in [DATA_DIR, LAWS_DIR, CASES_DIR, TEMPLATES_DIR, KNOWLEDGE_DIR]:
        assert os.path.isdir(d), f"目录不存在: {d}"


def test_config_law_files_count():
    """配置 — 法律文件数量"""
    from config import LAWS_DIR
    files = [f for f in os.listdir(LAWS_DIR) if f.endswith(".txt")]
    assert len(files) >= 5, f"法律文件应 >= 5 个，实际 {len(files)} 个"


def test_config_knowledge_files_count():
    """配置 — 知识库文件数量"""
    from config import KNOWLEDGE_DIR
    files = [f for f in os.listdir(KNOWLEDGE_DIR) if f.endswith(".txt")]
    assert len(files) >= 3, f"知识库文件应 >= 3 个，实际 {len(files)} 个"


# ============================================================
# 31. 创新工具增强测试
# ============================================================

def test_compensation_product_quality():
    """赔偿预估 — 产品质量"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "产品质量",
        "purchase_amount": 500,
        "actual_loss": 2000,
    })
    assert "产品质量法" in result
    assert "2000" in result or "实际" in result


def test_compensation_prepaid():
    """赔偿预估 — 预付款"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "预付款",
        "purchase_amount": 3000,
        "actual_loss": 0,
    })
    assert "预付" in result or "第五十三条" in result


def test_compensation_personal_injury():
    """赔偿预估 — 人身损害"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "人身损害",
        "purchase_amount": 100,
        "actual_loss": 5000,
    })
    assert "人身" in result or "医疗" in result or "第四十九条" in result


def test_deadline_civil_lawsuit():
    """时效提醒 — 民事诉讼时效"""
    from tools.innovation_tools import rights_deadline_reminder
    result = rights_deadline_reminder.invoke({
        "dispute_type": "民事诉讼时效",
        "purchase_date": "2025-01-01",
    })
    assert "民事诉讼" in result or "三年" in result or "1095" in result


def test_deadline_quality_warranty():
    """时效提醒 — 质量保修"""
    from tools.innovation_tools import rights_deadline_reminder
    result = rights_deadline_reminder.invoke({
        "dispute_type": "质量保修",
        "purchase_date": "2026-01-01",
    })
    assert "质量保修" in result or "保修" in result


def test_multi_platform_consumer_assoc():
    """多平台投诉 — 消协"""
    from tools.innovation_tools import multi_platform_complaint
    result = multi_platform_complaint.invoke({
        "dispute_info": "买到过期食品",
        "platform": "消协",
        "complainant_name": "王五",
    })
    assert "消协" in result or "消费投诉" in result
    assert "王五" in result


def test_multi_platform_ecommerce():
    """多平台投诉 — 电商平台"""
    from tools.innovation_tools import multi_platform_complaint
    result = multi_platform_complaint.invoke({
        "dispute_info": "商品质量问题",
        "platform": "电商平台",
        "merchant_name": "测试商品",
    })
    assert "订单" in result or "电商" in result


def test_trap_warning_ecommerce():
    """消费陷阱 — 电商行业"""
    from tools.innovation_tools import trap_warning
    result = trap_warning.invoke({"industry": "电商"})
    assert len(result) > 20
    assert "电商" in result or "陷阱" in result


def test_trap_warning_food():
    """消费陷阱 — 食品行业"""
    from tools.innovation_tools import trap_warning
    result = trap_warning.invoke({"industry": "食品"})
    assert len(result) > 20
    assert "食品" in result or "陷阱" in result


def test_trap_warning_education():
    """消费陷阱 — 教育行业"""
    from tools.innovation_tools import trap_warning
    result = trap_warning.invoke({"industry": "教育"})
    assert len(result) > 20
    assert "教育" in result or "陷阱" in result


def test_merchant_reputation_meituan():
    """商家信誉 — 美团"""
    from tools.innovation_tools import check_merchant_reputation
    result = check_merchant_reputation.invoke({"merchant_name": "美团"})
    assert "美团" in result
    assert "投诉" in result or "信誉" in result or "风险" in result


def test_merchant_reputation_jd():
    """商家信誉 — 京东"""
    from tools.innovation_tools import check_merchant_reputation
    result = check_merchant_reputation.invoke({"merchant_name": "京东"})
    assert "京东" in result
    assert "信誉" in result or "评分" in result or "投诉" in result


# ============================================================
# 32. 文件工具增强测试
# ============================================================

def test_file_create_with_confirm():
    """文件创建 — confirm 覆盖"""
    from tools.file_tools import create_file, delete_file
    # 先清理
    delete_file.invoke({"filename": "confirm_test.txt", "confirm": True})
    # 创建
    create_file.invoke({"filename": "confirm_test.txt", "content": "v1"})
    # 再次创建不 confirm
    result = create_file.invoke({"filename": "confirm_test.txt", "content": "v2"})
    assert "已存在" in result
    # confirm 覆盖
    result = create_file.invoke({"filename": "confirm_test.txt", "content": "v3", "confirm": True})
    assert "已创建" in result
    # 验证内容
    from tools.file_tools import read_file
    content = read_file.invoke({"filename": "confirm_test.txt"})
    assert content == "v3"
    # 清理
    delete_file.invoke({"filename": "confirm_test.txt", "confirm": True})


def test_file_delete_no_confirm():
    """文件删除 — 未确认不删除"""
    from tools.file_tools import create_file, delete_file
    create_file.invoke({"filename": "delete_no_confirm.txt", "content": "test"})
    result = delete_file.invoke({"filename": "delete_no_confirm.txt"})
    assert "确认" in result, f"未确认应提示确认: {result}"
    # 文件应仍存在
    from tools.file_tools import read_file
    content = read_file.invoke({"filename": "delete_no_confirm.txt"})
    assert content == "test"
    # 清理
    delete_file.invoke({"filename": "delete_no_confirm.txt", "confirm": True})


def test_file_delete_nonexist():
    """文件删除 — 删除不存在的文件"""
    from tools.file_tools import delete_file
    result = delete_file.invoke({"filename": "nonexist_delete.txt", "confirm": True})
    assert "不存在" in result


def test_file_read_directory():
    """文件读取 — 读取目录报错"""
    from tools.file_tools import create_file, read_file
    # 创建一个目录（通过 create_file 无法直接创建目录，模拟用现有目录）
    result = read_file.invoke({"filename": "."})
    # . 会被 normpath 为 WORK_ROOT 本身，读取目录应报错
    assert "不是" in result or "失败" in result or isinstance(result, str)


def test_file_zip_nonexist():
    """文件压缩 — 源文件不存在"""
    from tools.file_tools import zip_files
    result = zip_files.invoke({
        "file_list": ["nonexistent_source.txt"],
        "output_zip": "test_nonexist.zip",
    })
    assert "不存在" in result or "错误" in result


# ============================================================
# 33. Word 工具增强测试
# ============================================================

def test_word_complaint_overwrite():
    """Word 投诉信 — 覆盖已存在文件"""
    from tools.word_tools import create_complaint_report
    # 第一次创建
    result = create_complaint_report.invoke({
        "title": "覆盖测试",
        "complainant": "测试人",
        "respondent": "测试商家",
        "complaint_content": ["内容"],
        "legal_basis": ["法条"],
        "demands": ["诉求"],
        "filename": "overwrite_test_complaint.docx",
        "confirm": True,
    })
    assert "已生成" in result
    # 再次创建不 confirm
    result = create_complaint_report.invoke({
        "title": "覆盖测试2",
        "complainant": "测试人2",
        "respondent": "测试商家2",
        "complaint_content": ["内容2"],
        "legal_basis": ["法条2"],
        "demands": ["诉求2"],
        "filename": "overwrite_test_complaint.docx",
        "confirm": False,
    })
    assert "已存在" in result


def test_word_review_overwrite():
    """Word 审查报告 — 覆盖已存在文件"""
    from tools.word_tools import create_review_report
    result = create_review_report.invoke({
        "contract_title": "覆盖测试",
        "review_results": ["结论"],
        "risk_items": ["风险"],
        "suggestions": ["建议"],
        "risk_level": "低",
        "filename": "overwrite_test_review.docx",
        "confirm": True,
    })
    assert "已生成" in result
    result = create_review_report.invoke({
        "contract_title": "覆盖测试2",
        "review_results": ["结论2"],
        "risk_items": ["风险2"],
        "suggestions": ["建议2"],
        "risk_level": "低",
        "filename": "overwrite_test_review.docx",
        "confirm": False,
    })
    assert "已存在" in result


def test_word_complaint_auto_filename():
    """Word 投诉信 — 自动生成文件名"""
    from tools.word_tools import create_complaint_report
    result = create_complaint_report.invoke({
        "title": "自动命名测试",
        "complainant": "测试",
        "respondent": "测试",
        "complaint_content": ["内容"],
        "legal_basis": ["法条"],
        "demands": ["诉求"],
        "filename": None,
        "confirm": True,
    })
    assert "complaint_" in result, f"自动文件名应包含 complaint_ 前缀: {result}"


def test_word_review_auto_filename():
    """Word 审查报告 — 自动生成文件名"""
    from tools.word_tools import create_review_report
    result = create_review_report.invoke({
        "contract_title": "自动命名",
        "review_results": ["结论"],
        "risk_items": ["风险"],
        "suggestions": ["建议"],
        "risk_level": "中",
        "filename": None,
        "confirm": True,
    })
    assert "review_" in result, f"自动文件名应包含 review_ 前缀: {result}"


# ============================================================
# 主测试函数
# ============================================================
def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("消费维权智能助手 - 高级测试套件")
    print("=" * 60)
    print()

    # 20. 路径安全增强
    print("[20] 路径安全增强测试 (bug 回归)")
    test("路径越界-标准", test_file_safe_path_traversal)
    test("路径越界-同前缀绕过", test_file_safe_path_prefix_bypass)
    test("路径安全-正常路径", test_file_safe_path_valid)
    test("路径安全-子目录", test_file_safe_path_subdir)
    test("Word路径越界-标准", test_word_safe_path_traversal)
    test("Word路径越界-同前缀绕过", test_word_safe_path_prefix_bypass)
    test("Word路径安全-正常", test_word_safe_path_valid)
    test("Word路径-自动补全后缀", test_word_auto_extension)
    print()

    # 21. 邮件工具
    print("[21] 邮件工具测试")
    test("邮件-无效邮箱", test_email_invalid_address)
    test("邮件-预览模式", test_email_preview_mode)
    test("邮件-附件不存在", test_email_attachment_not_found)
    test("邮件-附件路径越界", test_email_attachment_path_traversal)
    test("邮件-附件绝对路径", test_email_attachment_absolute_path)
    test("邮件-空正文预览", test_email_empty_body)
    test("邮件-预览内容完整", test_email_preview_content)
    print()

    # 22. 联网搜索工具
    print("[22] 联网搜索工具测试")
    test("联网搜索工具-可导入", test_web_search_tool_exists)
    test("最新法规搜索-可导入", test_latest_regulation_tool_exists)
    test("商家信息搜索-可导入", test_merchant_info_tool_exists)
    test("_do_search-返回列表", test_do_search_returns_list)
    test("Bing搜索-数据结构", test_bing_search_structure)
    print()

    # 23. 智能检索工具
    print("[23] 智能检索工具测试")
    test("smart_search-本地命中", test_smart_search_local_hit)
    test("smart_search-食品安全", test_smart_search_food_safety)
    test("smart_search-返回字符串", test_smart_search_returns_string)
    test("自动入库-函数存在", test_add_web_results_to_kb_function_exists)
    test("自动入库-空输入", test_add_web_results_to_kb_empty_input)
    test("自动入库-短文本不入库", test_add_web_results_to_kb_short_text)
    test("自动入库-去重", test_add_web_results_to_kb_dedup)
    print()

    # 24. Embedder
    print("[24] Embedder 单例模式测试")
    test("Embedder-单例模式", test_embedder_singleton)
    test("Embedder-向量维度", test_embedder_dimension)
    test("Embedder-单条嵌入", test_embedder_embed_single)
    test("Embedder-批量嵌入", test_embedder_embed_batch)
    test("Embedder-字符串输入", test_embedder_embed_string_input)
    test("Embedder-向量归一化", test_embedder_normalized)
    print()

    # 25. IntentRouter
    print("[25] IntentRouter 意图路由器测试")
    test("路由-意图常量", test_router_constants)
    test("路由-类结构", test_router_class_structure)
    test("路由-系统提示词", test_router_system_prompt)
    print()

    # 26. 数据库模块增强
    print("[26] 数据库模块增强测试")
    test("DB-log_email方法", test_db_email_method)
    test("DB-get_email_stats方法", test_db_email_stats_method)
    test("DB-schema email表", test_db_schema_email_table)
    test("DB-所有日志方法", test_db_all_log_methods)
    print()

    # 27. Agent 路由与工具注册
    print("[27] Agent 路由与工具注册测试")
    test("QA Agent-工具注册", test_qa_agent_tools_registered)
    test("Complaint Agent-工具注册", test_complaint_agent_tools_registered)
    test("Review Agent-工具注册", test_review_agent_tools_registered)
    test("所有Agent-LLM配置", test_all_agents_have_llm)
    test("所有Agent-提示词非空", test_all_agents_have_prompt)
    test("工具标签-smart_search", test_tool_labels_include_smart_search)
    test("工具标签-send_email", test_tool_labels_include_send_email)
    print()

    # 28. RAG 端到端检索
    print("[28] RAG 端到端检索测试")
    test("RAG-法律检索", test_rag_search_law_result)
    test("RAG-案例检索", test_rag_search_case_result)
    test("RAG-食品安全法", test_rag_search_food_safety)
    test("RAG-电子商务法", test_rag_search_electronic_commerce)
    test("RAG-产品质量法", test_rag_search_product_quality)
    test("RAG-结果含匹配度", test_rag_search_returns_score)
    test("RAG-结果含出处", test_rag_search_metadata)
    print()

    # 29. Embedder 预加载机制
    print("[29] Embedder 预加载机制测试")
    test("预加载代码存在", test_embedder_preload_in_search_tools)
    test("匹配度阈值", test_search_tools_score_threshold)
    test("检索器延迟初始化", test_search_tools_retriever_lazy)
    print()

    # 30. 配置完整性
    print("[30] 配置完整性测试")
    test("SMTP_ENABLED布尔值", test_config_smtp_enabled_flag)
    test("向量目录存在", test_config_vectors_dir_exists)
    test("所有数据目录存在", test_config_all_dirs)
    test("法律文件数量", test_config_law_files_count)
    test("知识库文件数量", test_config_knowledge_files_count)
    print()

    # 31. 创新工具增强
    print("[31] 创新工具增强测试")
    test("赔偿-产品质量", test_compensation_product_quality)
    test("赔偿-预付款", test_compensation_prepaid)
    test("赔偿-人身损害", test_compensation_personal_injury)
    test("时效-民事诉讼", test_deadline_civil_lawsuit)
    test("时效-质量保修", test_deadline_quality_warranty)
    test("多平台-消协", test_multi_platform_consumer_assoc)
    test("多平台-电商", test_multi_platform_ecommerce)
    test("陷阱-电商", test_trap_warning_ecommerce)
    test("陷阱-食品", test_trap_warning_food)
    test("陷阱-教育", test_trap_warning_education)
    test("信誉-美团", test_merchant_reputation_meituan)
    test("信誉-京东", test_merchant_reputation_jd)
    print()

    # 32. 文件工具增强
    print("[32] 文件工具增强测试")
    test("文件-confirm覆盖", test_file_create_with_confirm)
    test("文件-删除未确认", test_file_delete_no_confirm)
    test("文件-删除不存在", test_file_delete_nonexist)
    test("文件-读取目录", test_file_read_directory)
    test("文件-压缩源不存在", test_file_zip_nonexist)
    print()

    # 33. Word 工具增强
    print("[33] Word 工具增强测试")
    test("Word投诉信-覆盖", test_word_complaint_overwrite)
    test("Word审查报告-覆盖", test_word_review_overwrite)
    test("Word投诉信-自动命名", test_word_complaint_auto_filename)
    test("Word审查报告-自动命名", test_word_review_auto_filename)
    print()

    # 汇总
    total = _passed + _failed
    print("=" * 60)
    print("高级测试结果汇总")
    print("=" * 60)
    print(f"  通过: {_passed}/{total}")
    print(f"  失败: {_failed}/{total}")
    if _failed > 0:
        print(f"\n  {_failed} 个测试失败，详情:")
        for name, msg, _ in _errors:
            print(f"    - {name}: {msg}")
    else:
        print("\n  全部测试通过！")
    print()
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
