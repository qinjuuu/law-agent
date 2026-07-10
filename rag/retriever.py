"""
混合检索器模块
融合向量检索（语义相似）和 BM25 检索（关键词匹配），取长补短
"""
from typing import List, Tuple
import numpy as np

from rag.chunking import Chunk
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.bm25_retriever import BM25Retriever
from config import TOP_K, VECTOR_WEIGHT, BM25_WEIGHT


class HybridRetriever:
    """
    混合检索器
    同时执行向量检索和 BM25 检索，加权融合后返回最终结果
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        bm25: BM25Retriever = None,
        top_k: int = TOP_K,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.top_k = top_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight

    def search(self, query: str, top_k: int = None) -> List[Tuple[Chunk, float, str]]:
        """
        混合检索

        参数:
            query: 查询文本
            top_k: 返回结果数

        返回:
            (Chunk, fused_score, source) 列表
            source 标记结果来源: "vector" / "bm25" / "both"
        """
        top_k = top_k or self.top_k
        if not query.strip():
            return []

        # 结果收集: chunk_text -> {chunk, vector_score, bm25_score}
        results_map = {}

        # 1. 向量检索
        if self.vector_store and self.vector_store.size > 0 and self.embedder:
            query_vec = self.embedder.embed_query(query)
            vec_results = self.vector_store.search(query_vec, top_k=top_k * 2)
            # 归一化向量分数
            max_vec_score = max((s for _, s in vec_results), default=1.0)
            for chunk, score in vec_results:
                key = chunk.text[:100]  # 用文本前100字做去重 key
                normalized = score / max_vec_score if max_vec_score > 0 else 0
                results_map[key] = {"chunk": chunk, "vec_score": normalized, "bm25_score": 0}

        # 2. BM25 检索
        if self.bm25 and self.bm25.size > 0:
            bm25_results = self.bm25.search(query, top_k=top_k * 2)
            max_bm25_score = max((s for _, s in bm25_results), default=1.0)
            for chunk, score in bm25_results:
                key = chunk.text[:100]
                normalized = score / max_bm25_score if max_bm25_score > 0 else 0
                if key in results_map:
                    results_map[key]["bm25_score"] = normalized
                else:
                    results_map[key] = {"chunk": chunk, "vec_score": 0, "bm25_score": normalized}

        # 3. 加权融合
        fused = []
        for key, info in results_map.items():
            final_score = (
                self.vector_weight * info["vec_score"]
                + self.bm25_weight * info["bm25_score"]
            )
            # 标记来源
            if info["vec_score"] > 0 and info["bm25_score"] > 0:
                source = "both"
            elif info["vec_score"] > 0:
                source = "vector"
            else:
                source = "bm25"
            fused.append((info["chunk"], final_score, source))

        # 降序排序，取 top_k
        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def search_for_prompt(self, query: str, top_k: int = None) -> str:
        """
        检索并格式化为提示词上下文文本

        返回:
            格式化的法律条文/案例引用文本，可直接拼入 LLM prompt
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return "（未检索到相关法律条文）"

        lines = []
        for i, (chunk, score, source) in enumerate(results, 1):
            meta = chunk.metadata
            source_label = meta.get("source", "未知来源")
            article = meta.get("article", "")
            header = f"【参考{i}】来源: {source_label}"
            if article:
                header += f" | {article}"
            header += f" | 匹配度: {score:.2f} | 检索路径: {source}"
            lines.append(f"{header}\n{chunk.text}\n")

        return "\n".join(lines)
