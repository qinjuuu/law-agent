"""
文本分块模块
针对中文法律文本进行语义感知分块，尊重条款边界
"""
import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class Chunk:
    """文本块"""
    text: str
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.text)

    def __repr__(self):
        preview = self.text[:50].replace("\n", " ")
        return f"Chunk(text='{preview}...', meta={self.metadata})"


class TextChunker:
    """
    中文法律文本分块器
    支持三种策略: semantic（按条款分）、sentence（按句分）、fixed（固定长度）
    默认使用 semantic 策略，尊重法律条文的第X条边界
    """

    # 法律条文常见起始模式: 第一条、第二十条、第一百零一条 等
    ARTICLE_PATTERN = re.compile(r"第[一二三四五六七八九十百千零\d]+条")

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, source: str = "", strategy: str = "semantic") -> List[Chunk]:
        """
        对文本进行分块

        参数:
            text: 原始文本
            source: 来源标记（文件名等），写入 metadata
            strategy: 分块策略 semantic / sentence / fixed

        返回:
            Chunk 列表
        """
        text = text.strip()
        if not text:
            return []

        if strategy == "semantic":
            chunks = self._chunk_semantic(text, source)
        elif strategy == "sentence":
            chunks = self._chunk_sentence(text, source)
        else:
            chunks = self._chunk_fixed(text, source)

        # 过滤过短的块（少于 20 字的碎片跳过）
        chunks = [c for c in chunks if len(c.text) >= 20]
        return chunks

    def _chunk_semantic(self, text: str, source: str) -> List[Chunk]:
        """按法律条文边界分块，超长条款再按句切分"""
        # 尝试按"第X条"切分
        splits = self.ARTICLE_PATTERN.split(text)
        matches = self.ARTICLE_PATTERN.findall(text)

        chunks = []
        if len(splits) > 1 and matches:
            # 有条文结构，按条组装
            # splits[0] 是第一条之前的内容（通常是标题或说明）
            if splits[0].strip():
                chunks.append(Chunk(
                    text=splits[0].strip(),
                    metadata={"source": source, "type": "preamble"}
                ))
            # 每个条文 = matches[i] + splits[i+1]
            for i, match in enumerate(matches):
                if i + 1 < len(splits):
                    article_text = (match + splits[i + 1]).strip()
                    if len(article_text) > self.chunk_size:
                        # 超长条文再按句切
                        sub_chunks = self._split_long_article(article_text, source, match)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(Chunk(
                            text=article_text,
                            metadata={"source": source, "article": match, "type": "article"}
                        ))
        else:
            # 无条文结构，按段落分
            chunks = self._chunk_sentence(text, source)

        return chunks

    def _split_long_article(self, text: str, source: str, article_label: str) -> List[Chunk]:
        """将超长条文按句号切分为多个块"""
        sentences = re.split(r"(?<=[。；！？])", text)
        chunks = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    metadata={"source": source, "article": article_label, "type": "article"}
                ))
                current = sent
            else:
                current += sent
        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                metadata={"source": source, "article": article_label, "type": "article"}
            ))
        return chunks

    def _chunk_sentence(self, text: str, source: str) -> List[Chunk]:
        """按句号/换行分块"""
        sentences = re.split(r"(?<=[。；！？\n])", text)
        chunks = []
        current = ""
        for sent in sentences:
            if not sent.strip():
                continue
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(Chunk(
                    text=current.strip(),
                    metadata={"source": source, "type": "paragraph"}
                ))
                current = sent
            else:
                current += sent
        if current.strip():
            chunks.append(Chunk(
                text=current.strip(),
                metadata={"source": source, "type": "paragraph"}
            ))
        return chunks

    def _chunk_fixed(self, text: str, source: str) -> List[Chunk]:
        """固定长度分块，带重叠"""
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                metadata={"source": source, "chunk_idx": idx, "type": "fixed"}
            ))
            start = end - self.chunk_overlap
            idx += 1
        return chunks
