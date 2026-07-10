"""
嵌入模型模块
封装 sentence-transformers 中文嵌入模型，将文本转为向量
"""
import os
import numpy as np
from typing import List

# 设置 HuggingFace 镜像（国内网络加速）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from config import EMBEDDING_MODEL


class Embedder:
    """
    中文文本嵌入器
    使用 BAAI/bge-small-zh-v1.5 模型（512 维，约 100MB）
    首次使用时自动下载到本地缓存
    """

    _instance = None  # 单例模式，避免重复加载模型

    def __new__(cls, model_name: str = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = None):
        if self._initialized:
            return
        model_name = model_name or EMBEDDING_MODEL
        print(f"[RAG] 正在加载嵌入模型 {model_name} ...")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_embedding_dimension()
            print(f"[RAG] 模型加载完成，向量维度 {self.dim}")
        except Exception as e:
            print(f"[RAG] 嵌入模型加载失败: {e}")
            print("[RAG] 请执行 pip install sentence-transformers 安装依赖")
            raise
        self._initialized = True

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        将文本列表转为向量矩阵

        参数:
            texts: 文本列表

        返回:
            numpy 数组，shape = (len(texts), dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(embeddings)

    def embed_query(self, text: str) -> np.ndarray:
        """嵌入单条查询文本，返回一维向量"""
        return self.embed([text])[0]
