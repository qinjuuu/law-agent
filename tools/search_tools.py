"""
法律检索工具模块
供 Agent 调用的法律条文和案例检索工具
通过 RAG 混合检索器实现知识库查询
smart_search 实现先查本地知识库、匹配度不足时自动回退联网搜索的策略
联网搜索结果会自动入库，实现知识库自增长
"""
# 预加载嵌入模型：必须在 langchain 导入前初始化，
# 否则 langchain 的导入链会触发全局 torch（可能 DLL 损坏），
# 导致嵌入模型无法加载
try:
    from rag.embedder import Embedder as _EmbedderPreload
    _EmbedderPreload()
except Exception:
    pass  # 延迟到 _get_retriever() 再尝试

from langchain.tools import tool

# 本地知识库匹配度阈值：融合分数低于此值视为"未找到匹配"
_SCORE_THRESHOLD = 0.15

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


@tool
def smart_search(query: str) -> str:
    """
    智能检索：先查本地知识库，匹配度不足时自动回退联网搜索

    这是主检索工具，自动处理本地→联网的回退逻辑，无需手动判断。
    返回结果会标注来源（本地知识库 / 联网搜索），联网结果仅供参考。

    参数:
        query: 检索关键词或问题描述，如"强制交易怎么维权"、"网购七天无理由退货条件"

    返回:
        检索结果文本，包含法条原文/案例/联网资讯，并标注来源和匹配度
    """
    # ===== 第一步：查询本地知识库 =====
    local_results = []
    try:
        retriever = _get_retriever()
        local_results = retriever.search(query, top_k=5)
    except Exception as e:
        print(f"[smart_search] 本地检索异常: {e}")

    # 判断本地匹配是否充分：最高分结果需超过阈值
    local_sufficient = False
    if local_results:
        top_score = local_results[0][1]  # (chunk, score, source) 中的 score
        local_sufficient = top_score >= _SCORE_THRESHOLD

    if local_sufficient:
        # 本地知识库命中，格式化返回
        lines = ["【本地知识库检索结果】\n"]
        for i, (chunk, score, source) in enumerate(local_results, 1):
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
        lines.append("（以上内容来自本地法律知识库，匹配度较高，可直接引用）")
        return "\n".join(lines)

    # ===== 第二步：本地匹配不足，回退联网搜索 =====
    lines = []

    # 先附上本地低匹配度结果（如有）
    if local_results:
        lines.append("【本地知识库初步匹配】（匹配度较低，仅供参考）")
        for i, (chunk, score, source) in enumerate(local_results[:2], 1):
            meta = chunk.metadata
            source_label = meta.get("source", "未知")
            article = meta.get("article", "")
            header = f"出处: {source_label}"
            if article:
                header += f" | {article}"
            header += f" | 匹配度: {score:.2f}"
            lines.append(f"--- 本地参考{i} ---")
            lines.append(header)
            lines.append(chunk.text)
            lines.append("")
        lines.append("---\n")

    lines.append("本地知识库匹配度不足，已自动联网搜索补充:\n")

    # 调用联网搜索
    try:
        from tools.web_search_tools import _do_search
        web_results = _do_search(query, max_results=5)
        if web_results:
            lines.append("【联网搜索结果】")
            for i, r in enumerate(web_results, 1):
                lines.append(f"--- 搜索结果{i} ---")
                lines.append(f"标题: {r.get('title', '')}")
                lines.append(f"摘要: {r.get('body', '')}")
                lines.append(f"来源: {r.get('href', '')}\n")
            lines.append("提示: 以上联网搜索结果来自互联网，请注意甄别准确性。")
            lines.append("法律法规引用请以本地知识库或官方发布版本为准。")

            # ===== 自动将联网搜索结果入库 =====
            added = _add_web_results_to_kb(query, web_results)
            if added > 0:
                lines.append(f"\n[知识库自增长] 已将 {added} 条联网搜索结果自动加入本地知识库，后续检索可直接命中。")
        else:
            lines.append("联网搜索也未找到相关结果，建议换个关键词试试。")
            if local_results:
                lines.append("可以参考上方本地知识库的低匹配度结果。")
    except Exception as e:
        lines.append(f"联网搜索失败: {str(e)}")
        if local_results:
            lines.append("请参考上方本地知识库的初步匹配结果。")

    return "\n".join(lines)


def _add_web_results_to_kb(query: str, web_results: list) -> int:
    """
    将联网搜索结果自动加入本地知识库（FAISS + BM25）
    实现知识库自增长：下次检索同样主题时可直接命中本地

    参数:
        query: 原始查询（用于记录 metadata）
        web_results: 联网搜索结果列表 [{title, body, href}, ...]

    返回:
        成功入库的文本块数量
    """
    try:
        retriever = _get_retriever()
        embedder = retriever.embedder
        vector_store = retriever.vector_store
        bm25 = retriever.bm25

        # 将搜索结果拼接为文本，使用 TextChunker 分块
        from rag.chunking import TextChunker, Chunk

        chunker = TextChunker(chunk_size=512, chunk_overlap=50)
        new_chunks = []

        for r in web_results:
            title = r.get("title", "").strip()
            body = r.get("body", "").strip()
            href = r.get("href", "").strip()
            if not body and not title:
                continue

            # 拼接为完整文本
            text_parts = []
            if title:
                text_parts.append(title)
            if body:
                text_parts.append(body)
            full_text = "\n".join(text_parts)

            if len(full_text) < 20:
                continue

            # 分块，source 标记为联网搜索来源
            source_label = f"联网搜索: {title[:30]}" if title else "联网搜索"
            chunks = chunker.chunk_text(full_text, source=source_label, strategy="sentence")

            # 给每个 chunk 补充 metadata
            for c in chunks:
                c.metadata["type"] = "web_search"
                c.metadata["url"] = href
                c.metadata["query"] = query

            new_chunks.extend(chunks)

        if not new_chunks:
            return 0

        # 去重：检查文本前100字是否已存在于知识库
        existing_keys = set()
        for existing_chunk in vector_store.chunks:
            existing_keys.add(existing_chunk.text[:100])

        unique_chunks = []
        for c in new_chunks:
            key = c.text[:100]
            if key not in existing_keys:
                existing_keys.add(key)
                unique_chunks.append(c)

        if not unique_chunks:
            print("[smart_search] 联网搜索结果已在知识库中，无需重复入库")
            return 0

        # 嵌入向量
        texts = [c.text for c in unique_chunks]
        vectors = embedder.embed(texts)

        # 添加到 FAISS 索引
        vector_store.add(unique_chunks, vectors)

        # 添加到 BM25 索引
        bm25.add_chunks(unique_chunks)

        # 持久化保存
        vector_store.save()
        bm25.save()

        print(f"[smart_search] 联网搜索结果已自动入库: +{len(unique_chunks)} 个文本块 (查询: {query[:30]})")
        return len(unique_chunks)

    except Exception as e:
        print(f"[smart_search] 联网结果入库失败（不影响检索结果）: {e}")
        return 0
