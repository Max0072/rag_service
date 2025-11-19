from abc import ABC, abstractmethod
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np
import re


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass

# 1 -- 1 -- 1 -- 1
class RecursiveCharacterTextChunker(BaseChunker):
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunker = chunker_recursive = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    def chunk(self, text: str):
        chunks = self.chunker.split_text(text)
        return chunks

# 2 -- 2 -- 2 -- 2
class TokenChunker(BaseChunker):
    def __init__(self, model: str = "gpt-4o-mini", chunk_size=512, overlap=64):
        self.enc = tiktoken.encoding_for_model(model)
        self.chunk_size = chunk_size
        self.overlap = overlap
    def chunk(self, text: str):
        tokens = self.enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.chunk_size
            chunk_tokens = tokens[start:end]
            chunks.append(self.enc.decode(chunk_tokens))
            start += self.chunk_size - self.overlap
        return chunks

# 3 -- 3 -- 3 -- 3
class SemanticChunker(BaseChunker):
    def __init__(self, model_name="all-MiniLM-L6-v2", max_sentences=10, sim_threshold=0.6):
        self.model = SentenceTransformer(model_name)
        self.max_sentences = max_sentences
        self.sim_threshold = sim_threshold

    def chunk(self, text: str):
        sentences = text.split(". ")
        embs = self.model.encode(sentences)
        chunks, current, current_embs = [], [], []

        for sent, emb in zip(sentences, embs):
            if not current:
                current = [sent]
                current_embs = [emb]
                continue

            avg_emb = np.mean(current_embs, axis=0)
            sim = np.dot(emb, avg_emb) / (
                np.linalg.norm(emb) * np.linalg.norm(avg_emb) + 1e-9
            )

            if sim < self.sim_threshold or len(current) >= self.max_sentences:
                chunks.append(". ".join(current))
                current = [sent]
                current_embs = [emb]
            else:
                current.append(sent)
                current_embs.append(emb)

        if current:
            chunks.append(". ".join(current))
        return chunks




class CodeChunker(BaseChunker):
    def __init__(self, max_lines: int = 80):
        self.max_lines = max_lines

    def chunk(self, code: str):
        lines = code.splitlines()
        chunks = []
        current = []

        for line in lines:
            # Новая функция или класс — начинаем новый чанк
            if re.match(r'^\s*(def |class )', line) and current:
                chunks.append("\n".join(current))
                current = [line]
            else:
                current.append(line)

            if len(current) >= self.max_lines:
                chunks.append("\n".join(current))
                current = []

        if current:
            chunks.append("\n".join(current))

        return chunks
