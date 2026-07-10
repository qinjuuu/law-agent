"""
FAISS 向量存储模块
管理法律文本的向量索引，支持持久化保存与加载
"""
import os
import json
import pickle
import numpy as np
from typing import List, Tuple

from config import VECTORS_DIR
from rag.chunking import Chunk


class VectorStore:
    """
    基于 FAISS 的向量存储
    支持 add（批量写入）、search（最近邻搜索）、save/load（持久化）
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.index = None
        self.chunks: List[Chunk] = []
        self._init_index()

    def _init_index(self):
        """初始化 FAISS 索引（L2 距离）"""
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)  # 内积相似度（配合归一化向量）

    def add(self, chunks: List[Chunk], vectors: np.ndarray):
        """
        批量添加文本块和对应的向量

        参数:
            chunks: Chunk 列表
            vectors: 对应的向量矩阵 (n, dim)
        """
        if len(chunks) != vectors.shape[0]:
            raise ValueError(f"块数量 {len(chunks)} 与向量数量 {vectors.shape[0]} 不匹配")
        # 确保向量是 float32 连续内存
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        搜索最相似的文本块

        参数:
            query_vector: 查询向量 (dim,)
            top_k: 返回结果数

        返回:
            (Chunk, score) 列表，按相似度降序
        """
        if self.index.ntotal == 0:
            return []
        query_vector = np.ascontiguousarray(
            query_vector.astype(np.float32).reshape(1, -1)
        )
        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(score)))
        return results

    @property
    def size(self) -> int:
        """已存储的向量数量"""
        return self.index.ntotal if self.index else 0

    def save(self, path_prefix: str = None):
        """
        持久化保存索引和元数据

        参数:
            path_prefix: 文件路径前缀，默认使用 data/vectors/faiss_index
        """
        import faiss
        if path_prefix is None:
            path_prefix = os.path.join(VECTORS_DIR, "faiss_index")

        # 保存 FAISS 索引
        faiss.write_index(self.index, f"{path_prefix}.faiss")
        # 保存 Chunk 元数据
        with open(f"{path_prefix}.meta", "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"[RAG] 向量索引已保存: {self.size} 条向量 -> {path_prefix}")

    def load(self, path_prefix: str = None) -> bool:
        """
        从文件加载索引和元数据

        返回:
            是否加载成功
        """
        import faiss
        if path_prefix is None:
            path_prefix = os.path.join(VECTORS_DIR, "faiss_index")

        faiss_path = f"{path_prefix}.faiss"
        meta_path = f"{path_prefix}.meta"
        if not os.path.exists(faiss_path):
            return False

        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"[RAG] 向量索引已加载: {self.size} 条向量")
        return True
