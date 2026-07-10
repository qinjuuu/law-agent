"""
系统自检脚本
验证项目结构、配置、RAG 索引状态，安装依赖后运行
用法: python scripts/test_system.py
"""
import sys
import os

# HuggingFace 镜像 + SSL 修复
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def check_imports():
    """检查核心模块导入"""
    print("[1/5] 检查模块导入...")
    try:
        import langchain
        print(f"  langchain: {langchain.__version__}")
    except ImportError:
        print("  [失败] langchain 未安装")
        return False

    try:
        import langchain_openai
        print(f"  langchain_openai: OK")
    except ImportError:
        print("  [失败] langchain_openai 未安装")
        return False

    try:
        import gradio as gr
        print(f"  gradio: {gr.__version__}")
    except ImportError:
        print("  [失败] gradio 未安装")
        return False

    try:
        import faiss
        print(f"  faiss: OK")
    except ImportError:
        print("  [失败] faiss-cpu 未安装")
        return False

    try:
        import sentence_transformers
        print(f"  sentence_transformers: OK")
    except ImportError:
        print("  [失败] sentence-transformers 未安装")
        return False

    try:
        import jieba
        print(f"  jieba: OK")
    except ImportError:
        print("  [失败] jieba 未安装")
        return False

    try:
        import docx
        print(f"  python-docx: OK")
    except ImportError:
        print("  [失败] python-docx 未安装")
        return False

    print("  所有依赖导入成功")
    return True


def check_config():
    """检查配置"""
    print("\n[2/5] 检查配置...")
    try:
        from config import ARK_API_KEY, ARK_BASE_URL, MODEL_ID, EMBEDDING_MODEL
        if not ARK_API_KEY:
            print("  [警告] ARK_API_KEY 未配置，请在 .env 文件中填写")
            return False
        print(f"  模型: {MODEL_ID}")
        print(f"  嵌入: {EMBEDDING_MODEL}")
        print(f"  API: {ARK_BASE_URL}")
        print("  配置检查通过")
        return True
    except Exception as e:
        print(f"  [失败] {e}")
        return False


def check_data():
    """检查知识库数据"""
    print("\n[3/5] 检查知识库数据...")
    from config import LAWS_DIR, CASES_DIR, TEMPLATES_DIR

    law_count = 0
    case_count = 0
    template_count = 0

    if os.path.exists(LAWS_DIR):
        law_count = len([f for f in os.listdir(LAWS_DIR) if f.endswith(".txt")])
    if os.path.exists(CASES_DIR):
        case_count = len([f for f in os.listdir(CASES_DIR) if f.endswith(".txt")])
    if os.path.exists(TEMPLATES_DIR):
        template_count = len([f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".txt")])

    print(f"  法律条文: {law_count} 个文件")
    print(f"  维权案例: {case_count} 个文件")
    print(f"  模板文档: {template_count} 个文件")

    if law_count == 0:
        print("  [警告] 未找到法律条文文件，请检查 data/laws/ 目录")
        return False

    print("  数据检查通过")
    return True


def check_rag():
    """检查 RAG 索引"""
    print("\n[4/5] 检查 RAG 索引...")
    from config import VECTORS_DIR

    faiss_path = os.path.join(VECTORS_DIR, "faiss_index.faiss")
    bm25_path = os.path.join(VECTORS_DIR, "bm25_index.pkl")

    faiss_ok = os.path.exists(faiss_path)
    bm25_ok = os.path.exists(bm25_path)

    if faiss_ok and bm25_ok:
        faiss_size = os.path.getsize(faiss_path)
        bm25_size = os.path.getsize(bm25_path)
        print(f"  FAISS 索引: {faiss_size:,} 字节")
        print(f"  BM25 索引: {bm25_size:,} 字节")
        print("  RAG 索引已构建")
        return True
    else:
        print("  [提示] RAG 索引未构建")
        print("  请先运行: python scripts/ingest_data.py")
        return False


def check_llm():
    """检查 LLM 连接"""
    print("\n[5/5] 检查 LLM 连接...")
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        from config import ARK_API_KEY, ARK_BASE_URL, MODEL_ID

        llm = ChatOpenAI(
            model=MODEL_ID,
            api_key=ARK_API_KEY,
            base_url=ARK_BASE_URL,
            temperature=0.0,
        )
        response = llm.invoke([HumanMessage(content="请回复'连接成功'四个字")])
        print(f"  LLM 回复: {response.content}")
        print("  LLM 连接正常")
        return True
    except Exception as e:
        print(f"  [失败] {e}")
        return False


def main():
    print("=" * 60)
    print("消费维权智能助手 - 系统自检")
    print("=" * 60)

    results = []
    results.append(("依赖导入", check_imports()))
    results.append(("配置检查", check_config()))
    results.append(("数据检查", check_data()))
    results.append(("RAG 索引", check_rag()))
    if all(r[1] for r in results[:2]):  # 只有依赖和配置通过才测 LLM
        results.append(("LLM 连接", check_llm()))

    print("\n" + "=" * 60)
    print("自检结果汇总")
    print("=" * 60)
    for name, passed in results:
        status = "通过" if passed else "未通过"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n所有检查通过! 可以启动应用: python app.py")
    else:
        print("\n部分检查未通过，请根据提示修复后重试")

    return all_passed


if __name__ == "__main__":
    main()
