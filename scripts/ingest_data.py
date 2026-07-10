"""
数据向量化入库脚本
将 data/ 目录下的法律条文、案例、模板文本进行分块、嵌入、存储
运行一次即可，后续启动时自动加载已有索引
"""
import sys
import os

# HuggingFace 镜像 + SSL 修复（入库时需要下载模型，用镜像）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

# 添加项目根目录到 path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from config import LAWS_DIR, CASES_DIR, TEMPLATES_DIR, KNOWLEDGE_DIR, VECTORS_DIR
from rag.chunking import TextChunker
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever


def load_text_files(directory: str) -> list:
    """
    加载目录下所有 .txt 文件

    返回:
        [(filename, content), ...]
    """
    results = []
    if not os.path.exists(directory):
        return results
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            results.append((filename, content))
    return results


def main():
    print("=" * 60)
    print("消费维权智能助手 - 数据向量化入库")
    print("=" * 60)

    # 1. 加载所有文本文件
    print("\n[1/5] 加载文本数据...")
    law_files = load_text_files(LAWS_DIR)
    case_files = load_text_files(CASES_DIR)
    template_files = load_text_files(TEMPLATES_DIR)
    knowledge_files = load_text_files(KNOWLEDGE_DIR)

    print(f"  法律条文: {len(law_files)} 个文件")
    print(f"  维权案例: {len(case_files)} 个文件")
    print(f"  模板文档: {len(template_files)} 个文件")
    print(f"  知识库: {len(knowledge_files)} 个文件")

    all_files = []
    for name, content in law_files:
        all_files.append((name, content, "law"))
    for name, content in case_files:
        all_files.append((name, content, "case"))
    for name, content in template_files:
        all_files.append((name, content, "template"))
    for name, content in knowledge_files:
        all_files.append((name, content, "knowledge"))

    if not all_files:
        print("  [警告] 未找到任何文本文件，请检查 data/ 目录")
        return

    # 2. 文本分块
    print("\n[2/5] 文本分块...")
    chunker = TextChunker(chunk_size=512, chunk_overlap=50)
    all_chunks = []
    for filename, content, doc_type in all_files:
        source_label = f"{doc_type}/{filename}"
        chunks = chunker.chunk_text(content, source=source_label)
        # 给每个 chunk 加上文档类型标记
        for chunk in chunks:
            chunk.metadata["doc_type"] = doc_type
        all_chunks.extend(chunks)
        print(f"  {source_label}: {len(chunks)} 个文本块")

    print(f"  总计: {len(all_chunks)} 个文本块")

    # 3. 嵌入向量化
    print("\n[3/5] 嵌入向量化...")
    embedder = Embedder()
    texts = [chunk.text for chunk in all_chunks]

    # 分批嵌入（避免内存溢出）
    batch_size = 64
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embedder.embed(batch)
        all_vectors.append(vectors)
        print(f"  嵌入进度: {min(i + batch_size, len(texts))}/{len(texts)}")

    import numpy as np
    all_vectors = np.vstack(all_vectors)
    print(f"  向量矩阵形状: {all_vectors.shape}")

    # 4. FAISS 向量存储
    print("\n[4/5] 构建 FAISS 索引...")
    vector_store = VectorStore(dim=embedder.dim)
    vector_store.add(all_chunks, all_vectors)
    vector_store.save()
    print(f"  FAISS 索引: {vector_store.size} 条向量")

    # 5. BM25 索引
    print("\n[5/5] 构建 BM25 索引...")
    bm25 = BM25Retriever()
    bm25.add_chunks(all_chunks)
    bm25.save()
    print(f"  BM25 索引: {bm25.size} 篇文档")

    print("\n" + "=" * 60)
    print(f"入库完成! 向量索引 {vector_store.size} 条, BM25 索引 {bm25.size} 篇")
    print(f"索引文件位于: {VECTORS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
