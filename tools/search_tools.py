"""
法律检索工具模块
供 Agent 调用的法律条文和案例检索工具
通过 RAG 混合检索器实现知识库查询
"""
from langchain.tools import tool

# 延迟初始化检索器，避免在模块加载时触发模型下载
_retriever = None


def _get_retriever():
    """懒加载混合检索器"""
    global _retriever
    if _retriever is None:
        from rag.retriever import HybridRetriever
        from rag.embedder import Embedder
        from rag.vector_store import VectorStore
        from rag.bm25_retriever import BM25Retriever

        embedder = Embedder()
        vector_store = VectorStore(dim=embedder.dim)
        vector_store.load()  # 尝试加载已有索引

        bm25 = BM25Retriever()
        bm25.load()  # 尝试加载已有索引

        _retriever = HybridRetriever(
            embedder=embedder,
            vector_store=vector_store,
            bm25=bm25,
        )
    return _retriever


@tool
def search_law(query: str) -> str:
    """
    检索与消费者维权相关的法律条文

    参数:
        query: 检索关键词或问题描述，如"七天无理由退货"、"食品安全赔偿"

    返回:
        相关法律条文列表，包含法条出处和匹配内容

    适用场景:
        用户询问消费者权益相关法律规定、需要法条依据时调用
        投诉信起草需要引用法律依据时调用
        格式条款审查需要判断条款合法性时调用
    """
    try:
        retriever = _get_retriever()
        results = retriever.search(query, top_k=5)
        if not results:
            return "未检索到相关法律条文，建议换个关键词试试"

        lines = ["检索到以下相关法律条文:\n"]
        for i, (chunk, score, source) in enumerate(results, 1):
            meta = chunk.metadata
            source_label = meta.get("source", "未知")
            article = meta.get("article", "")
            lines.append(f"--- 参考{i} ---")
            if article:
                lines.append(f"出处: {source_label} | {article} | 匹配度: {score:.2f}")
            else:
                lines.append(f"出处: {source_label} | 匹配度: {score:.2f}")
            lines.append(chunk.text)
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"检索失败: {str(e)}"


@tool
def search_case(query: str) -> str:
    """
    检索与消费维权相关的典型案例

    参数:
        query: 案例检索关键词，如"网购退款纠纷"、"食品安全十倍赔偿"

    返回:
        相关案例列表，包含案例描述和处理结果

    适用场景:
        用户想了解类似维权案例的处理方式和结果时调用
        投诉信起草需要参考类似案例时调用
    """
    try:
        retriever = _get_retriever()
        # 用案例相关关键词增强查询
        enhanced_query = f"案例 {query}"
        results = retriever.search(enhanced_query, top_k=3)
        if not results:
            return "未检索到相关案例"

        lines = ["检索到以下相关案例:\n"]
        for i, (chunk, score, source) in enumerate(results, 1):
            meta = chunk.metadata
            source_label = meta.get("source", "未知")
            lines.append(f"--- 案例{i} ---")
            lines.append(f"来源: {source_label} | 匹配度: {score:.2f}")
            lines.append(chunk.text)
            lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"检索失败: {str(e)}"
