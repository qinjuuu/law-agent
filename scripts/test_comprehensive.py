"""
消费维权智能助手 - 综合测试套件
覆盖 15 个模块的 82+ 项测试，包括:
1. 配置模块      2. 文本分块      3. BM25检索器
4. 向量存储      5. 混合检索器    6. 文件工具
7. Word工具     8. 创新工具      9. 搜索工具
10. Agent工具   11. 用户画像    12. 信息完整性
13. 思维链      14. 置信度评估   15. 维权进度
"""
import os
import sys
import time
import traceback
import tempfile
import shutil

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"

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
# 1. 配置模块测试
# ============================================================
def test_config_import():
    """配置模块可导入"""
    import config
    assert config is not None


def test_config_syspath_clean():
    """sys.path 清理了全局 Python312"""
    import config  # noqa: F811
    for p in sys.path:
        if "site-packages" in p.lower():
            assert "python312" not in p.lower(), f"未清理 Python312 路径: {p}"


def test_config_dirs_exist():
    """工作目录和报告目录存在"""
    from config import WORK_ROOT, WORD_REPORTS_DIR, VECTORS_DIR
    assert os.path.isdir(WORK_ROOT), f"工作目录不存在: {WORK_ROOT}"
    assert os.path.isdir(WORD_REPORTS_DIR), f"报告目录不存在: {WORD_REPORTS_DIR}"
    assert os.path.isdir(VECTORS_DIR), f"向量目录不存在: {VECTORS_DIR}"


def test_config_model_id():
    """MODEL_ID 配置正确"""
    from config import MODEL_ID
    assert MODEL_ID, "MODEL_ID 未配置"
    assert len(MODEL_ID) > 5, f"MODEL_ID 过短: {MODEL_ID}"


def test_config_embedding_model():
    """EMBEDDING_MODEL 配置正确"""
    from config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL, "EMBEDDING_MODEL 未配置"
    assert "bge" in EMBEDDING_MODEL.lower() or "bert" in EMBEDDING_MODEL.lower(), \
        f"EMBEDDING_MODEL 疑似不正确: {EMBEDDING_MODEL}"


def test_config_rag_params():
    """RAG 参数在合理范围"""
    from config import TOP_K, VECTOR_WEIGHT, BM25_WEIGHT, CHUNK_SIZE, CHUNK_OVERLAP
    assert 1 <= TOP_K <= 20, f"TOP_K 超出范围: {TOP_K}"
    assert 0.0 <= VECTOR_WEIGHT <= 1.0, f"VECTOR_WEIGHT 超出范围: {VECTOR_WEIGHT}"
    assert 0.0 <= BM25_WEIGHT <= 1.0, f"BM25_WEIGHT 超出范围: {BM25_WEIGHT}"
    assert CHUNK_SIZE > 0, f"CHUNK_SIZE 应为正数: {CHUNK_SIZE}"
    assert 0 <= CHUNK_OVERLAP < CHUNK_SIZE, f"CHUNK_OVERLAP 应小于 CHUNK_SIZE"


# ============================================================
# 2. 文本分块测试
# ============================================================
def test_chunk_dataclass():
    """Chunk 数据类基本操作"""
    from rag.chunking import Chunk
    c = Chunk(text="测试文本", metadata={"source": "test.txt"})
    assert len(c) == 4
    assert c.metadata["source"] == "test.txt"


def test_chunker_semantic_basic():
    """语义分块 — 按法律条文边界"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    text = (
        "第一条 为保护消费者合法权益，制定本法。\n"
        "第二条 消费者为生活消费需要购买商品，其权益受保护。\n"
        "第三条 经营者提供商品或服务应当遵守本法。"
    )
    chunks = chunker.chunk_text(text, source="test_law.txt", strategy="semantic")
    assert len(chunks) >= 2, f"分块数应>=2，实际{len(chunks)}"
    for c in chunks:
        assert len(c.text) >= 20 or c.metadata.get("type") == "preamble", \
            f"分块过短: {len(c.text)} 字符"


def test_chunker_semantic_article_metadata():
    """语义分块 — 条文元数据正确"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    text = (
        "消费者权益保护法\n"
        "第一条 为保护消费者合法权益，制定本法，本法规定了消费者的权利和经营者的义务。\n"
        "第二条 消费者为生活消费需要购买、使用商品或接受服务，其权益受本法保护。\n"
        "第三条 经营者为消费者提供商品或服务应当遵守本法。"
    )
    chunks = chunker.chunk_text(text, source="消法.txt", strategy="semantic")
    article_chunks = [c for c in chunks if c.metadata.get("type") == "article"]
    assert len(article_chunks) >= 2, f"条文分块数应>=2，实际{len(article_chunks)}"
    for c in article_chunks:
        assert "article" in c.metadata, "条文分块应有 article 元数据"
        assert c.metadata["article"].startswith("第"), \
            f"article 元数据应以'第'开头: {c.metadata['article']}"


def test_chunker_sentence():
    """句子分块策略"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。"
    chunks = chunker.chunk_text(text, source="test.txt", strategy="sentence")
    assert len(chunks) >= 1
    combined = " ".join(c.text for c in chunks)
    assert "第一句" in combined


def test_chunker_fixed():
    """固定长度分块"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "A" * 200
    chunks = chunker.chunk_text(text, source="test.txt", strategy="fixed")
    assert len(chunks) >= 3
    for c in chunks:
        assert len(c.text) <= 50


def test_chunker_empty_text():
    """空文本分块"""
    from rag.chunking import TextChunker
    chunker = TextChunker()
    assert chunker.chunk_text("", source="empty.txt") == []
    assert chunker.chunk_text("   \n  ", source="whitespace.txt") == []


def test_chunker_short_filter():
    """短文本块过滤"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=512)
    text = "短。\n这是一个足够长的段落用于通过过滤，确保不会被短文本过滤器去掉。"
    chunks = chunker.chunk_text(text, source="test.txt", strategy="sentence")
    for c in chunks:
        assert len(c.text) >= 20, f"短文本块未被过滤: {len(c.text)} 字符"


def test_chunker_long_article_split():
    """超长条文按句切分"""
    from rag.chunking import TextChunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=10)
    long_article = "第一条 " + "这是一段很长的法律条文内容。" * 20
    chunks = chunker.chunk_text(long_article, source="test.txt", strategy="semantic")
    assert len(chunks) >= 2, "超长条文应被切分为多个块"


# ============================================================
# 3. BM25 检索器测试
# ============================================================
def test_bm25_basic():
    """BM25 基本检索"""
    from rag.bm25_retriever import BM25Retriever
    from rag.chunking import Chunk
    bm25 = BM25Retriever()
    chunks = [
        Chunk(text="消费者权益保护法规定退货制度", metadata={}),
        Chunk(text="食品安全法规定了十倍赔偿", metadata={}),
        Chunk(text="产品质量法涉及缺陷责任", metadata={}),
    ]
    bm25.add_chunks(chunks)
    results = bm25.search("消费者退货", top_k=2)
    assert len(results) > 0, "BM25 应返回结果"
    assert results[0][0].text == chunks[0].text, "最相关结果应为第一条"


def test_bm25_empty_query():
    """BM25 空查询"""
    from rag.bm25_retriever import BM25Retriever
    from rag.chunking import Chunk
    bm25 = BM25Retriever()
    bm25.add_chunks([Chunk(text="测试内容", metadata={})])
    results = bm25.search("", top_k=5)
    assert results == [], "空查询应返回空列表"


def test_bm25_empty_index():
    """BM25 空索引"""
    from rag.bm25_retriever import BM25Retriever
    bm25 = BM25Retriever()
    assert bm25.size == 0
    results = bm25.search("测试", top_k=5)
    assert results == [], "空索引应返回空列表"


def test_bm25_save_load():
    """BM25 持久化"""
    from rag.bm25_retriever import BM25Retriever
    from rag.chunking import Chunk
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "bm25_test.pkl")
        bm25 = BM25Retriever()
        bm25.add_chunks([Chunk(text="持久化测试数据", metadata={})])
        bm25.save(path)
        assert os.path.exists(path)
        bm25_2 = BM25Retriever()
        assert bm25_2.load(path)
        assert bm25_2.size == 1
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 4. 向量存储测试
# ============================================================
def test_vector_store_basic():
    """向量存储基本操作"""
    import numpy as np
    from rag.vector_store import VectorStore
    from rag.chunking import Chunk
    dim = 64
    store = VectorStore(dim=dim)
    vectors = np.random.rand(3, dim).astype(np.float32)
    chunks = [Chunk(text=f"文本块{i}", metadata={"idx": i}) for i in range(3)]
    store.add(chunks, vectors)
    assert store.size == 3
    query = np.random.rand(dim).astype(np.float32)
    results = store.search(query, top_k=2)
    assert len(results) == 2


def test_vector_store_empty():
    """空向量存储搜索"""
    import numpy as np
    from rag.vector_store import VectorStore
    store = VectorStore(dim=64)
    query = np.random.rand(64).astype(np.float32)
    results = store.search(query, top_k=5)
    assert results == []


def test_vector_store_save_load():
    """向量存储持久化"""
    import numpy as np
    from rag.vector_store import VectorStore
    from rag.chunking import Chunk
    tmpdir = tempfile.mkdtemp()
    try:
        store = VectorStore(dim=32)
        vectors = np.random.rand(2, 32).astype(np.float32)
        chunks = [Chunk(text=f"测试{i}", metadata={}) for i in range(2)]
        store.add(chunks, vectors)
        store.save(os.path.join(tmpdir, "test_index"))
        store2 = VectorStore(dim=32)
        assert store2.load(os.path.join(tmpdir, "test_index"))
        assert store2.size == 2
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ============================================================
# 5. 混合检索器测试
# ============================================================
def test_hybrid_retriever_empty_query():
    """混合检索器 — 空查询"""
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.search("   ")
    assert results == [], "空查询应返回空列表"


def test_hybrid_retriever_no_data():
    """混合检索器 — 无数据"""
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    results = retriever.search("消费者权益")
    assert results == [], "无数据时应返回空列表"


def test_hybrid_search_for_prompt_empty():
    """检索格式化 — 无结果"""
    from rag.retriever import HybridRetriever
    retriever = HybridRetriever()
    text = retriever.search_for_prompt("无匹配查询")
    assert isinstance(text, str)
    assert "未检索到" in text or len(text) > 0


# ============================================================
# 6. 文件工具测试
# ============================================================
def test_file_create_read():
    """文件创建和读取"""
    from tools.file_tools import create_file, read_file, delete_file
    # 清理上次测试残留文件
    delete_file.invoke({"filename": "test_file.txt", "confirm": True})
    result = create_file.invoke({"filename": "test_file.txt", "content": "测试内容"})
    assert "已创建" in result, f"创建失败: {result}"
    content = read_file.invoke({"filename": "test_file.txt"})
    assert content == "测试内容", f"内容不符: {content}"


def test_file_overwrite_protection():
    """文件覆盖保护"""
    from tools.file_tools import create_file
    create_file.invoke({"filename": "overwrite_test.txt", "content": "原始"})
    result = create_file.invoke({"filename": "overwrite_test.txt", "content": "新内容"})
    assert "已存在" in result, f"未触发覆盖保护: {result}"


def test_file_delete():
    """文件删除"""
    from tools.file_tools import create_file, delete_file
    create_file.invoke({"filename": "delete_test.txt", "content": "待删除"})
    result = delete_file.invoke({"filename": "delete_test.txt", "confirm": True})
    assert "删除成功" in result, f"删除失败: {result}"


def test_file_safe_path():
    """路径越界保护"""
    from tools.file_tools import _safe_path
    try:
        _safe_path("../../etc/passwd")
        assert False, "路径越界未触发 ValueError"
    except ValueError:
        pass


def test_file_read_nonexist():
    """读取不存在的文件"""
    from tools.file_tools import read_file
    result = read_file.invoke({"filename": "nonexistent_file.txt"})
    assert "不存在" in result


def test_file_zip():
    """文件压缩"""
    from tools.file_tools import create_file, zip_files
    create_file.invoke({"filename": "zip_source.txt", "content": "压缩测试"})
    result = zip_files.invoke({"file_list": ["zip_source.txt"], "output_zip": "test_output.zip"})
    assert "压缩成功" in result


# ============================================================
# 7. Word 文档工具测试
# ============================================================
def test_word_complaint_report():
    """生成投诉信"""
    from tools.word_tools import create_complaint_report
    result = create_complaint_report.invoke({
        "title": "测试投诉信",
        "complainant": "张三 13800138000",
        "respondent": "某某商家",
        "complaint_content": ["纠纷内容1", "纠纷内容2"],
        "legal_basis": ["《消费者权益保护法》第二十四条"],
        "demands": ["退款", "赔偿"],
        "filename": "test_complaint.docx",
        "confirm": True,
    })
    assert "已生成" in result, f"投诉信生成失败: {result}"


def test_word_review_report():
    """生成审查报告"""
    from tools.word_tools import create_review_report
    result = create_review_report.invoke({
        "contract_title": "测试合同",
        "review_results": ["审查结论1"],
        "risk_items": ["风险条款1"],
        "suggestions": ["修改建议1"],
        "risk_level": "中",
        "filename": "test_review.docx",
        "confirm": True,
    })
    assert "已生成" in result, f"审查报告生成失败: {result}"


def test_word_safe_path():
    """Word 路径越界保护"""
    from tools.word_tools import _safe_word_path
    try:
        _safe_word_path("../../malicious.docx")
        assert False, "路径越界未触发 ValueError"
    except ValueError:
        pass


def test_word_confirm_required():
    """Word 确认机制"""
    from tools.word_tools import create_complaint_report
    result = create_complaint_report.invoke({
        "title": "测试",
        "complainant": "测试",
        "respondent": "测试",
        "complaint_content": ["测试"],
        "legal_basis": ["测试"],
        "demands": ["测试"],
        "filename": "confirm_test.docx",
        "confirm": False,
    })
    assert "确认" in result or "confirm" in result


# ============================================================
# 8. 创新工具测试
# ============================================================
def test_compensation_food_safety():
    """赔偿预估 — 食品安全"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "食品安全",
        "purchase_amount": 100,
        "actual_loss": 0,
    })
    assert "1000" in result or "100" in result, f"食品安全赔偿金额不正确: {result}"
    assert "食品安全法" in result


def test_compensation_fraud():
    """赔偿预估 — 欺诈"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "欺诈",
        "purchase_amount": 200,
        "actual_loss": 0,
    })
    assert "600" in result or "500" in result, f"欺诈赔偿金额不正确: {result}"
    assert "消费者权益保护法" in result


def test_compensation_unknown_type():
    """赔偿预估 — 未知类型"""
    from tools.innovation_tools import estimate_compensation
    result = estimate_compensation.invoke({
        "dispute_type": "未知类型",
        "purchase_amount": 100,
        "actual_loss": 0,
    })
    assert "不支持" in result or "暂不" in result


def test_evidence_checklist():
    """证据清单生成"""
    from tools.innovation_tools import generate_evidence_checklist
    result = generate_evidence_checklist.invoke({"dispute_type": "食品安全"})
    assert "购物小票" in result or "证据" in result
    assert len(result) > 50


def test_evidence_unknown_type():
    """证据清单 — 未知类型"""
    from tools.innovation_tools import generate_evidence_checklist
    result = generate_evidence_checklist.invoke({"dispute_type": "未知"})
    assert "不支持" in result or "暂不" in result


def test_rights_path_food():
    """维权路径 — 食品安全"""
    from tools.innovation_tools import plan_rights_path
    result = plan_rights_path.invoke({"dispute_description": "超市买到过期食品"})
    assert "食品安全" in result or "食品" in result
    assert "第1步" in result or "第2步" in result


def test_rights_path_online():
    """维权路径 — 网购"""
    from tools.innovation_tools import plan_rights_path
    result = plan_rights_path.invoke({"dispute_description": "网购退货被拒"})
    assert "网购" in result or "七天" in result


def test_rights_path_general():
    """维权路径 — 通用"""
    from tools.innovation_tools import plan_rights_path
    result = plan_rights_path.invoke({"dispute_description": "商家拒绝退款"})
    assert "第1步" in result
    assert "协商" in result or "投诉" in result


def test_merchant_tactics():
    """商家话术应对"""
    from tools.innovation_tools import merchant_tactics_response
    result = merchant_tactics_response.invoke({"merchant_statement": "特价商品概不退换"})
    assert len(result) > 20
    assert "消费者权益保护法" in result or "话术" in result or "建议" in result


def test_merchant_tactics_unknown():
    """商家话术 — 未匹配"""
    from tools.innovation_tools import merchant_tactics_response
    result = merchant_tactics_response.invoke({"merchant_statement": "这是一句完全无关的话"})
    assert len(result) > 20


def test_deadline_reminder():
    """时效提醒"""
    from tools.innovation_tools import rights_deadline_reminder
    result = rights_deadline_reminder.invoke({
        "dispute_type": "七天无理由退货",
        "purchase_date": "2026-07-01",
    })
    assert "七天无理由退货" in result
    assert "剩余" in result or "过期" in result


def test_deadline_invalid_date():
    """时效提醒 — 日期格式错误"""
    from tools.innovation_tools import rights_deadline_reminder
    result = rights_deadline_reminder.invoke({
        "dispute_type": "七天无理由退货",
        "purchase_date": "invalid",
    })
    assert "格式错误" in result or "YYYY" in result


def test_deadline_unknown_type():
    """时效提醒 — 未知类型"""
    from tools.innovation_tools import rights_deadline_reminder
    result = rights_deadline_reminder.invoke({
        "dispute_type": "未知时效",
        "purchase_date": "",
    })
    assert "不支持" in result or "暂不" in result


def test_multi_platform_12315():
    """多平台投诉 — 12315"""
    from tools.innovation_tools import multi_platform_complaint
    result = multi_platform_complaint.invoke({
        "dispute_info": "买到过期食品",
        "platform": "12315",
        "complainant_name": "张三",
        "contact": "13800138000",
        "merchant_name": "某超市",
        "demand": "退款赔偿",
    })
    assert "12315" in result
    assert "张三" in result


def test_multi_platform_court():
    """多平台投诉 — 法院"""
    from tools.innovation_tools import multi_platform_complaint
    result = multi_platform_complaint.invoke({
        "dispute_info": "消费欺诈",
        "platform": "法院",
        "complainant_name": "李四",
    })
    assert "民事起诉状" in result or "法院" in result


def test_multi_platform_unknown():
    """多平台投诉 — 未知平台"""
    from tools.innovation_tools import multi_platform_complaint
    result = multi_platform_complaint.invoke({
        "dispute_info": "测试",
        "platform": "未知平台",
    })
    assert "不支持" in result or "暂不" in result


def test_package_evidence():
    """证据打包"""
    from tools.innovation_tools import package_evidence
    result = package_evidence.invoke({
        "complaint_filename": "",
        "evidence_files": [],
        "output_name": "test_package.zip",
    })
    assert "已生成" in result or "失败" in result


def test_trap_warning():
    """消费陷阱查询"""
    from tools.innovation_tools import trap_warning
    result = trap_warning.invoke({"industry": "预付卡"})
    assert len(result) > 20
    assert "预付卡" in result or "陷阱" in result


def test_trap_warning_unknown():
    """消费陷阱 — 未知行业"""
    from tools.innovation_tools import trap_warning
    result = trap_warning.invoke({"industry": "未知行业"})
    assert len(result) > 0


def test_merchant_reputation_known():
    """商家信誉 — 已知商家"""
    from tools.innovation_tools import check_merchant_reputation
    result = check_merchant_reputation.invoke({"merchant_name": "拼多多"})
    assert "拼多多" in result
    assert "投诉" in result or "信誉" in result


def test_merchant_reputation_unknown():
    """商家信誉 — 未知商家"""
    from tools.innovation_tools import check_merchant_reputation
    result = check_merchant_reputation.invoke({"merchant_name": "完全未知的商家"})
    assert len(result) > 20


def test_tool_labels_completeness():
    """工具标签完整性"""
    from agents.base import _TOOL_LABELS
    required_tools = [
        "search_law", "search_case", "search_web", "estimate_compensation",
        "generate_evidence_checklist", "plan_rights_path", "merchant_tactics_response",
        "rights_deadline_reminder", "multi_platform_complaint", "package_evidence",
        "trap_warning", "check_merchant_reputation", "handoff_to_agent",
        "get_rights_progress", "create_complaint_report", "create_review_report",
        "create_file", "read_file", "delete_file", "zip_files",
    ]
    for tool_name in required_tools:
        assert tool_name in _TOOL_LABELS, f"工具 {tool_name} 缺少中文标签"


# ============================================================
# 9. 搜索工具测试
# ============================================================
def test_search_law_no_index():
    """法律检索 — 无索引"""
    from tools.search_tools import search_law
    result = search_law.invoke({"query": "七天无理由退货"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_search_case_no_index():
    """案例检索 — 无索引"""
    from tools.search_tools import search_case
    result = search_case.invoke({"query": "食品安全"})
    assert isinstance(result, str)
    assert len(result) > 0


# ============================================================
# 10. Agent 工具测试
# ============================================================
def test_handoff_to_agent():
    """Agent 交接工具"""
    from tools.agent_tools import handoff_to_agent
    result = handoff_to_agent.invoke({
        "target_agent": "complaint",
        "reason": "需要起草投诉信",
        "context_summary": "用户买到过期食品",
    })
    assert "投诉信起草专家" in result
    assert "过期食品" in result


def test_handoff_unknown():
    """Agent 交接 — 未知目标"""
    from tools.agent_tools import handoff_to_agent
    result = handoff_to_agent.invoke({
        "target_agent": "unknown",
        "reason": "测试",
    })
    assert len(result) > 0


def test_get_rights_progress():
    """维权进度查询工具"""
    from tools.agent_tools import get_rights_progress
    result = get_rights_progress.invoke({})
    assert "维权" in result
    assert "第1步" in result or "咨询" in result


# ============================================================
# 11. 用户画像自适应测试
# ============================================================
def test_profile_expert():
    """用户画像 — 专业用户"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("根据消法第五十五条退一赔三和食品安全法第一百四十八条惩罚性赔偿条款", None)
    assert result["legal_level"] == "expert", f"应为 expert，实际 {result['legal_level']}"


def test_profile_novice():
    """用户画像 — 新手用户"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("我不知道怎么办，能退吗，第一次遇到这种事", None)
    assert result["legal_level"] == "novice", f"应为 novice，实际 {result['legal_level']}"


def test_profile_intermediate():
    """用户画像 — 中等用户"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("我在网上买了个手机，有点质量问题", None)
    assert result["legal_level"] == "intermediate", f"应为 intermediate，实际 {result['legal_level']}"


def test_profile_urgent():
    """用户画像 — 紧迫"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("急！明天就过期了，来不及了", None)
    assert result["urgency"] == "urgent", f"应为 urgent，实际 {result['urgency']}"


def test_profile_normal():
    """用户画像 — 不紧迫"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("想咨询一下退货政策", None)
    assert result["urgency"] == "normal"


def test_profile_merchant():
    """用户画像 — 商家用户"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("我是商家，店铺经营中遇到消费者投诉", None)
    assert result["user_type"] == "merchant", f"应为 merchant，实际 {result['user_type']}"


def test_profile_empty_input():
    """用户画像 — 空输入"""
    from agents.enhancements import UserProfileDetector
    result = UserProfileDetector.detect("", None)
    assert result["legal_level"] in ("novice", "intermediate")
    assert result["urgency"] == "normal"


# ============================================================
# 12. 信息完整性测试
# ============================================================
def test_completeness_qa_full():
    """信息完整性 — QA 模式始终 100%"""
    from agents.enhancements import CompletenessTracker
    result = CompletenessTracker.track(None, "任何问题", "qa")
    assert result["completeness"] == 100
    assert result["should_ask"] is False


def test_completeness_complaint_complete():
    """信息完整性 — 完整投诉信息"""
    from agents.enhancements import CompletenessTracker
    text = "我叫张三，电话13800138000，在永辉超市昨天买的食品过期了，要求退款赔偿"
    result = CompletenessTracker.track(None, text, "complaint")
    assert result["completeness"] > 50, f"完整度应>50%，实际 {result['completeness']}"


def test_completeness_complaint_missing():
    """信息完整性 — 缺失信息"""
    from agents.enhancements import CompletenessTracker
    result = CompletenessTracker.track(None, "买了个东西不退", "complaint")
    assert result["completeness"] < 70, f"完整度应<70%，实际 {result['completeness']}"
    assert result["should_ask"] is True
    assert len(result["missing_fields"]) > 0


def test_completeness_review_complete():
    """信息完整性 — 审查模式"""
    from agents.enhancements import CompletenessTracker
    text = "这个合同条款写了最终解释权归商家所有，用于会员卡消费"
    result = CompletenessTracker.track(None, text, "review")
    assert result["completeness"] >= 50


def test_completeness_history_merge():
    """信息完整性 — 历史对话合并"""
    from agents.enhancements import CompletenessTracker
    history = [{"role": "user", "content": "我叫张三，电话是13800138000"}]
    current = "在美团买的食品过期了，要求赔偿"
    result = CompletenessTracker.track(history, current, "complaint")
    assert "投诉人姓名" in result["provided_fields"] or result["completeness"] > 30


# ============================================================
# 13. 思维链测试
# ============================================================
def test_reasoning_basic():
    """思维链提取 — 基本功能"""
    from agents.enhancements import ReasoningChainExtractor
    messages = []
    result = ReasoningChainExtractor.extract(messages, "退货问题", "neutral")
    assert "思维链" in result
    assert "理解意图" in result


def test_reasoning_with_emotion():
    """思维链提取 — 带情绪"""
    from agents.enhancements import ReasoningChainExtractor
    messages = []
    result = ReasoningChainExtractor.extract(messages, "气死我了", "anger")
    assert "情绪" in result or "愤怒" in result


def test_reasoning_with_tools():
    """思维链提取 — 带工具调用"""
    from agents.enhancements import ReasoningChainExtractor
    class MockMsg:
        tool_calls = [{"name": "search_law"}, {"name": "estimate_compensation"}]
    messages = [MockMsg()]
    result = ReasoningChainExtractor.extract(messages, "赔偿问题", "neutral")
    assert "检索法律条文" in result or "search_law" in result
    assert "预估赔偿金额" in result or "estimate_compensation" in result


# ============================================================
# 14. 置信度评估测试
# ============================================================
def test_confidence_high():
    """置信度 — 高（法条引用充分）"""
    from agents.enhancements import ConfidenceEvaluator
    answer = "根据《消费者权益保护法》第二十五条和《食品安全法》第一百四十八条的规定，消费者有权要求退货和赔偿。"
    result = ConfidenceEvaluator.evaluate("买到过期食品怎么办", answer, True)
    assert result["confidence"] == "high", f"应为 high，实际 {result['confidence']}"


def test_confidence_medium():
    """置信度 — 中"""
    from agents.enhancements import ConfidenceEvaluator
    answer = "您可以尝试联系商家协商解决，如果不行可以投诉。"
    result = ConfidenceEvaluator.evaluate("退货问题", answer, False)
    assert result["confidence"] == "medium", f"应为 medium，实际 {result['confidence']}"


def test_confidence_low():
    """置信度 — 低（无依据且短）"""
    from agents.enhancements import ConfidenceEvaluator
    answer = "不清楚。"
    result = ConfidenceEvaluator.evaluate("怎么办", answer, False)
    assert result["confidence"] == "low", f"应为 low，实际 {result['confidence']}"


def test_confidence_out_of_scope():
    """置信度 — 超出范围"""
    from agents.enhancements import ConfidenceEvaluator
    result = ConfidenceEvaluator.evaluate("我想离婚怎么办", "这个问题超出范围", False)
    assert result["is_out_of_scope"] is True
    assert result["confidence"] == "low"
    assert len(result["boundary_warning"]) > 0


def test_confidence_empty_answer():
    """置信度 — 空回答"""
    from agents.enhancements import ConfidenceEvaluator
    result = ConfidenceEvaluator.evaluate("问题", "", False)
    assert result["confidence"] == "low"


def test_confidence_emoji():
    """置信度 — emoji 正确"""
    from agents.enhancements import ConfidenceEvaluator
    high = ConfidenceEvaluator.evaluate("退货", "根据消费者权益保护法第二十五条和食品安全法第一百四十八条规定可退。", True)
    medium = ConfidenceEvaluator.evaluate("退货", "可以协商解决，不行再投诉。", False)
    low = ConfidenceEvaluator.evaluate("离婚", "不清楚。", False)
    assert high["confidence_emoji"] == "🟢"
    assert medium["confidence_emoji"] == "🟡"
    assert low["confidence_emoji"] == "🔴"


# ============================================================
# 15. 维权进度追踪测试
# ============================================================
def test_progress_initial():
    """维权进度 — 初始阶段"""
    from agents.enhancements import CaseProgressTracker
    result = CaseProgressTracker.track(None, "我想咨询一下退货怎么办")
    assert result["current_stage"] == 1, f"应在阶段1，实际在阶段{result['current_stage']}"
    assert "咨询" in result["current_label"]


def test_progress_platform():
    """维权进度 — 平台投诉阶段"""
    from agents.enhancements import CaseProgressTracker
    result = CaseProgressTracker.track(None, "我已经向12315平台投诉了")
    assert result["current_stage"] >= 3, f"应在阶段>=3，实际在阶段{result['current_stage']}"


def test_progress_resolved():
    """维权进度 — 已解决"""
    from agents.enhancements import CaseProgressTracker
    result = CaseProgressTracker.track(None, "问题已经解决了，商家退款了")
    assert result["current_stage"] == 6, f"应在阶段6，实际在阶段{result['current_stage']}"


def test_progress_format():
    """维权进度 — 格式化输出"""
    from agents.enhancements import CaseProgressTracker
    progress = CaseProgressTracker.track(None, "正在协商")
    text = CaseProgressTracker.format_progress(progress)
    assert "维权进度" in text
    assert "当前阶段" in text
    assert "下一步" in text or "已完成" in text


def test_progress_bar():
    """维权进度 — 进度条"""
    from agents.enhancements import CaseProgressTracker
    result = CaseProgressTracker.track(None, "咨询退货")
    assert "█" in result["progress_bar"] or "░" in result["progress_bar"]
    assert "%" in result["progress_bar"]


def test_progress_full_path():
    """维权进度 — 完整路径"""
    from agents.enhancements import CaseProgressTracker
    result = CaseProgressTracker.track(None, "协商过了")
    assert len(result["full_path"]) == 6
    has_current = any(s["status"] == "current" for s in result["full_path"])
    assert has_current, "full_path 应包含 current 状态"


# ============================================================
# 16. 情绪感知测试
# ============================================================
def test_emotion_anger():
    """情绪检测 — 愤怒"""
    from agents.base import _detect_emotion
    assert _detect_emotion("气死我了太坑了黑心商家") == "anger"


def test_emotion_anxiety():
    """情绪检测 — 焦虑"""
    from agents.base import _detect_emotion
    assert _detect_emotion("怎么办啊我不知道该怎么处理着急") == "anxiety"


def test_emotion_sad():
    """情绪检测 — 委屈"""
    from agents.base import _detect_emotion
    assert _detect_emotion("太委屈了倒霉透顶心寒失望") == "sad"


def test_emotion_neutral():
    """情绪检测 — 中性"""
    from agents.base import _detect_emotion
    assert _detect_emotion("我想了解一下退货政策") == "neutral"


def test_emotion_empty():
    """情绪检测 — 空输入"""
    from agents.base import _detect_emotion
    assert _detect_emotion("") == "neutral"


def test_emotion_prefix():
    """情绪前缀构建"""
    from agents.base import _build_emotion_prefix
    prefix = _build_emotion_prefix("anger")
    assert "愤怒" in prefix
    assert "😤" in prefix


def test_emotion_prefix_neutral():
    """情绪前缀 — 中性无前缀"""
    from agents.base import _build_emotion_prefix
    assert _build_emotion_prefix("neutral") == ""


# ============================================================
# 17. 对话历史转换测试
# ============================================================
def test_gradio_history_dict():
    """Gradio 历史 — dict 格式"""
    from agents.base import _gradio_history_to_messages
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "您好"},
    ]
    messages = _gradio_history_to_messages(history)
    assert len(messages) == 2


def test_gradio_history_tuple():
    """Gradio 历史 — tuple 格式"""
    from agents.base import _gradio_history_to_messages
    history = [["你好", "您好"]]
    messages = _gradio_history_to_messages(history)
    assert len(messages) == 2


def test_gradio_history_empty():
    """Gradio 历史 — 空"""
    from agents.base import _gradio_history_to_messages
    assert _gradio_history_to_messages(None) == []
    assert _gradio_history_to_messages([]) == []


def test_content_to_str():
    """内容转字符串"""
    from agents.base import _content_to_str
    assert _content_to_str("简单文本") == "简单文本"
    assert _content_to_str(None) == ""
    assert _content_to_str(["部分1", "部分2"]) == "部分1\n部分2"


def test_clean_agent_output():
    """清理 Agent 输出"""
    from agents.base import _clean_agent_output
    text = "> 😤 情绪感知: 愤怒\n\n---\n\n回答内容"
    clean = _clean_agent_output(text)
    assert "情绪感知" not in clean
    assert "---" not in clean
    assert "回答内容" in clean


# ============================================================
# 18. Agent 结构测试
# ============================================================
def test_qa_agent_init():
    """QA Agent 初始化"""
    from agents.qa_agent import ConsumerQAAgent
    agent = ConsumerQAAgent()
    assert agent.agent_type == "qa"
    assert len(agent.tools) > 0
    assert agent.system_prompt


def test_complaint_agent_init():
    """Complaint Agent 初始化"""
    from agents.complaint_agent import ComplaintAgent
    agent = ComplaintAgent()
    assert agent.agent_type == "complaint"
    assert len(agent.tools) > 0
    assert agent.system_prompt


def test_review_agent_init():
    """Review Agent 初始化"""
    from agents.review_agent import ReviewAgent
    agent = ReviewAgent()
    assert agent.agent_type == "review"
    assert len(agent.tools) > 0
    assert agent.system_prompt


def test_base_agent_class():
    """BaseAgent 类属性"""
    from agents.base import BaseAgent
    assert hasattr(BaseAgent, "chat")
    assert hasattr(BaseAgent, "_log_business_data")


# ============================================================
# 19. 数据库模块测试（非连接测试）
# ============================================================
def test_db_schema_exists():
    """数据库 schema 文件存在"""
    schema_path = os.path.join(_PROJECT_ROOT, "database", "schema.sql")
    assert os.path.exists(schema_path), "schema.sql 不存在"


def test_db_manager_class():
    """DatabaseManager 类定义"""
    from database.db_manager import DatabaseManager
    assert hasattr(DatabaseManager, "create_conversation")
    assert hasattr(DatabaseManager, "log_message")
    assert hasattr(DatabaseManager, "log_tool_call")
    assert hasattr(DatabaseManager, "get_stats")


def test_db_json_dumps():
    """数据库 JSON 序列化"""
    from database.db_manager import DatabaseManager
    # 创建一个不连接的实例来测试 _json_dumps
    result = DatabaseManager._json_dumps(DatabaseManager, {"key": "值"})
    assert isinstance(result, str)
    assert "值" in result


# ============================================================
# 主测试函数
# ============================================================
def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("消费维权智能助手 - 综合测试套件")
    print("=" * 60)
    print()

    # 1. 配置模块
    print("[1] 配置模块测试")
    test("配置模块导入", test_config_import)
    test("sys.path 清理", test_config_syspath_clean)
    test("工作目录存在", test_config_dirs_exist)
    test("MODEL_ID 配置", test_config_model_id)
    test("EMBEDDING_MODEL 配置", test_config_embedding_model)
    test("RAG 参数范围", test_config_rag_params)
    print()

    # 2. 文本分块
    print("[2] 文本分块测试")
    test("Chunk 数据类", test_chunk_dataclass)
    test("语义分块基本", test_chunker_semantic_basic)
    test("语义分块元数据", test_chunker_semantic_article_metadata)
    test("句子分块", test_chunker_sentence)
    test("固定长度分块", test_chunker_fixed)
    test("空文本分块", test_chunker_empty_text)
    test("短文本过滤", test_chunker_short_filter)
    test("超长条文切分", test_chunker_long_article_split)
    print()

    # 3. BM25 检索器
    print("[3] BM25 检索器测试")
    test("BM25 基本检索", test_bm25_basic)
    test("BM25 空查询", test_bm25_empty_query)
    test("BM25 空索引", test_bm25_empty_index)
    test("BM25 持久化", test_bm25_save_load)
    print()

    # 4. 向量存储
    print("[4] 向量存储测试")
    test("向量存储基本操作", test_vector_store_basic)
    test("空向量存储", test_vector_store_empty)
    test("向量存储持久化", test_vector_store_save_load)
    print()

    # 5. 混合检索器
    print("[5] 混合检索器测试")
    test("空查询检索", test_hybrid_retriever_empty_query)
    test("无数据检索", test_hybrid_retriever_no_data)
    test("格式化输出", test_hybrid_search_for_prompt_empty)
    print()

    # 6. 文件工具
    print("[6] 文件工具测试")
    test("文件创建读取", test_file_create_read)
    test("文件覆盖保护", test_file_overwrite_protection)
    test("文件删除", test_file_delete)
    test("路径越界保护", test_file_safe_path)
    test("读取不存在文件", test_file_read_nonexist)
    test("文件压缩", test_file_zip)
    print()

    # 7. Word 工具
    print("[7] Word 文档工具测试")
    test("投诉信生成", test_word_complaint_report)
    test("审查报告生成", test_word_review_report)
    test("Word 路径越界", test_word_safe_path)
    test("Word 确认机制", test_word_confirm_required)
    print()

    # 8. 创新工具
    print("[8] 创新工具测试")
    test("赔偿预估-食品安全", test_compensation_food_safety)
    test("赔偿预估-欺诈", test_compensation_fraud)
    test("赔偿预估-未知类型", test_compensation_unknown_type)
    test("证据清单生成", test_evidence_checklist)
    test("证据清单-未知类型", test_evidence_unknown_type)
    test("维权路径-食品安全", test_rights_path_food)
    test("维权路径-网购", test_rights_path_online)
    test("维权路径-通用", test_rights_path_general)
    test("商家话术应对", test_merchant_tactics)
    test("商家话术-未匹配", test_merchant_tactics_unknown)
    test("时效提醒", test_deadline_reminder)
    test("时效-日期格式错误", test_deadline_invalid_date)
    test("时效-未知类型", test_deadline_unknown_type)
    test("多平台投诉-12315", test_multi_platform_12315)
    test("多平台投诉-法院", test_multi_platform_court)
    test("多平台-未知平台", test_multi_platform_unknown)
    test("证据打包", test_package_evidence)
    test("消费陷阱查询", test_trap_warning)
    test("消费陷阱-未知行业", test_trap_warning_unknown)
    test("商家信誉-已知", test_merchant_reputation_known)
    test("商家信誉-未知", test_merchant_reputation_unknown)
    test("工具标签完整性", test_tool_labels_completeness)
    print()

    # 9. 搜索工具
    print("[9] 搜索工具测试")
    test("法律检索", test_search_law_no_index)
    test("案例检索", test_search_case_no_index)
    print()

    # 10. Agent 工具
    print("[10] Agent 工具测试")
    test("Agent 交接", test_handoff_to_agent)
    test("Agent 交接-未知", test_handoff_unknown)
    test("维权进度查询", test_get_rights_progress)
    print()

    # 11. 用户画像
    print("[11] 用户画像测试")
    test("画像-专业用户", test_profile_expert)
    test("画像-新手用户", test_profile_novice)
    test("画像-中等用户", test_profile_intermediate)
    test("画像-紧迫", test_profile_urgent)
    test("画像-不紧迫", test_profile_normal)
    test("画像-商家用户", test_profile_merchant)
    test("画像-空输入", test_profile_empty_input)
    print()

    # 12. 信息完整性
    print("[12] 信息完整性测试")
    test("完整性-QA 模式", test_completeness_qa_full)
    test("完整性-完整投诉", test_completeness_complaint_complete)
    test("完整性-缺失信息", test_completeness_complaint_missing)
    test("完整性-审查模式", test_completeness_review_complete)
    test("完整性-历史合并", test_completeness_history_merge)
    print()

    # 13. 思维链
    print("[13] 思维链测试")
    test("思维链-基本", test_reasoning_basic)
    test("思维链-带情绪", test_reasoning_with_emotion)
    test("思维链-带工具", test_reasoning_with_tools)
    print()

    # 14. 置信度评估
    print("[14] 置信度评估测试")
    test("置信度-高", test_confidence_high)
    test("置信度-中", test_confidence_medium)
    test("置信度-低", test_confidence_low)
    test("置信度-超出范围", test_confidence_out_of_scope)
    test("置信度-空回答", test_confidence_empty_answer)
    test("置信度-emoji", test_confidence_emoji)
    print()

    # 15. 维权进度
    print("[15] 维权进度测试")
    test("进度-初始阶段", test_progress_initial)
    test("进度-平台投诉", test_progress_platform)
    test("进度-已解决", test_progress_resolved)
    test("进度-格式化", test_progress_format)
    test("进度-进度条", test_progress_bar)
    test("进度-完整路径", test_progress_full_path)
    print()

    # 16. 情绪感知
    print("[16] 情绪感知测试")
    test("情绪-愤怒", test_emotion_anger)
    test("情绪-焦虑", test_emotion_anxiety)
    test("情绪-委屈", test_emotion_sad)
    test("情绪-中性", test_emotion_neutral)
    test("情绪-空输入", test_emotion_empty)
    test("情绪前缀构建", test_emotion_prefix)
    test("情绪前缀-中性", test_emotion_prefix_neutral)
    print()

    # 17. 对话历史转换
    print("[17] 对话历史转换测试")
    test("历史-dict 格式", test_gradio_history_dict)
    test("历史-tuple 格式", test_gradio_history_tuple)
    test("历史-空", test_gradio_history_empty)
    test("内容转字符串", test_content_to_str)
    test("清理 Agent 输出", test_clean_agent_output)
    print()

    # 18. Agent 结构
    print("[18] Agent 结构测试")
    test("QA Agent 初始化", test_qa_agent_init)
    test("Complaint Agent 初始化", test_complaint_agent_init)
    test("Review Agent 初始化", test_review_agent_init)
    test("BaseAgent 类属性", test_base_agent_class)
    print()

    # 19. 数据库模块
    print("[19] 数据库模块测试")
    test("schema 文件存在", test_db_schema_exists)
    test("DatabaseManager 类定义", test_db_manager_class)
    test("JSON 序列化", test_db_json_dumps)
    print()

    # 汇总
    total = _passed + _failed
    print("=" * 60)
    print("测试结果汇总")
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
