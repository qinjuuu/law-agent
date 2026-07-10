"""
BM25 关键词检索模块
基于 jieba 中文分词的 BM25 算法，与向量检索互补
"""
import math
import pickle
import os
from typing import List, Tuple
from collections import Counter

import jieba

from rag.chunking import Chunk


class BM25Retriever:
    """
    BM25 关键词检索器
    使用 jieba 分词处理中文文本，支持添加文档、搜索、持久化
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[Chunk] = []
        self.doc_tokens: List[List[str]] = []
        self.doc_len: List[int] = []
        self.avg_doc_len: float = 0.0
        self.df: Counter = Counter()  # 文档频率
        self.idf: dict = {}

    def _tokenize(self, text: str) -> List[str]:
        """jieba 分词，过滤停用词和单字"""
        tokens = jieba.lcut(text)
        # 过滤标点、空白、单字
        return [t.strip() for t in tokens if len(t.strip()) > 1]

    def add_chunks(self, chunks: List[Chunk]):
        """批量添加文档"""
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self.doc_tokens.append(tokens)
            self.doc_len.append(len(tokens))
            self.chunks.append(chunk)
            # 更新文档频率
            for word in set(tokens):
                self.df[word] += 1

        self.avg_doc_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0
        self._compute_idf()

    def _compute_idf(self):
        """计算 IDF 值"""
        n = len(self.chunks)
        for word, df in self.df.items():
            self.idf[word] = math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        BM25 关键词搜索

        参数:
            query: 查询文本
            top_k: 返回结果数

        返回:
            (Chunk, score) 列表，按相关度降序
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for i, doc_tokens in enumerate(self.doc_tokens):
            score = self._bm25_score(query_tokens, i)
            scores.append((i, score))

        # 降序排序，取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def _bm25_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """计算单篇文档的 BM25 得分"""
        doc_tokens = self.doc_tokens[doc_idx]
        doc_len = self.doc_len[doc_idx]
        tf = Counter(doc_tokens)

        score = 0.0
        for word in query_tokens:
            if word not in self.idf:
                continue
            idf = self.idf[word]
            f = tf.get(word, 0)
            # BM25 公式
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * numerator / denominator if denominator > 0 else 0

        return score

    @property
    def size(self) -> int:
        return len(self.chunks)

    def save(self, path: str = None):
        """持久化保存"""
        if path is None:
            from config import VECTORS_DIR
            path = os.path.join(VECTORS_DIR, "bm25_index.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "doc_tokens": self.doc_tokens,
                "doc_len": self.doc_len,
                "avg_doc_len": self.avg_doc_len,
                "df": self.df,
                "idf": self.idf,
                "k1": self.k1,
                "b": self.b,
            }, f)
        print(f"[RAG] BM25 索引已保存: {self.size} 篇文档 -> {path}")

    def load(self, path: str = None) -> bool:
        """从文件加载"""
        if path is None:
            from config import VECTORS_DIR
            path = os.path.join(VECTORS_DIR, "bm25_index.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.doc_tokens = data["doc_tokens"]
        self.doc_len = data["doc_len"]
        self.avg_doc_len = data["avg_doc_len"]
        self.df = data["df"]
        self.idf = data["idf"]
        self.k1 = data["k1"]
        self.b = data["b"]
        print(f"[RAG] BM25 索引已加载: {self.size} 篇文档")
        return True
