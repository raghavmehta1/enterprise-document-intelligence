from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    source: str
    section: str
    chunk_id: int
    text: str


@dataclass(frozen=True)
class Citation:
    source: str
    section: str
    chunk_id: int
    quote: str
    score: float


@dataclass(frozen=True)
class SearchResult:
    answer: str
    citations: tuple[Citation, ...]
    insufficient_evidence: bool


class DocumentIntelligenceModel:
    """Offline retrieval and extractive QA baseline with traceable citations."""

    def __init__(self, *, threshold: float = 0.12, chunk_words: int = 90, overlap: int = 20):
        if overlap >= chunk_words:
            raise ValueError("overlap must be smaller than chunk_words")
        self.threshold = threshold
        self.chunk_words = chunk_words
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = None

    def fit_directory(self, directory: str | Path) -> "DocumentIntelligenceModel":
        directory = Path(directory)
        self.chunks = []
        for path in sorted(directory.rglob("*.txt")):
            self.chunks.extend(self._chunk_document(path.name, path.read_text(encoding="utf-8")))
        if not self.chunks:
            raise ValueError(f"No .txt documents found in {directory}")
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)
        return self

    def ask(self, question: str, *, top_k: int = 3) -> SearchResult:
        if self.matrix is None:
            raise RuntimeError("Call fit_directory before ask")
        query = self.vectorizer.transform([question])
        scores = cosine_similarity(query, self.matrix).ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        citations = tuple(
            Citation(
                source=self.chunks[i].source,
                section=self.chunks[i].section,
                chunk_id=self.chunks[i].chunk_id,
                quote=self._best_sentence(self.chunks[i].text, question),
                score=round(float(scores[i]), 4),
            )
            for i in ranked
            if scores[i] >= self.threshold
        )
        if not citations:
            return SearchResult(
                answer="The documents do not contain enough evidence to answer reliably.",
                citations=(),
                insufficient_evidence=True,
            )
        return SearchResult(answer=citations[0].quote, citations=citations, insufficient_evidence=False)

    def _chunk_document(self, source: str, text: str) -> list[Chunk]:
        sections: list[tuple[str, list[str]]] = []
        heading = "Document"
        buffer: list[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if buffer:
                    sections.append((heading, buffer))
                heading, buffer = line.lstrip("# ").strip(), []
            else:
                buffer.extend(line.split())
        if buffer:
            sections.append((heading, buffer))

        chunks: list[Chunk] = []
        for section, words in sections:
            step = self.chunk_words - self.overlap
            for start in range(0, len(words), step):
                part = words[start : start + self.chunk_words]
                if part:
                    chunks.append(Chunk(source, section, len(chunks), " ".join(part)))
                if start + self.chunk_words >= len(words):
                    break
        return chunks

    @staticmethod
    def _best_sentence(text: str, question: str) -> str:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        return max(
            sentences,
            key=lambda sentence: len(terms & set(re.findall(r"[a-z0-9]+", sentence.lower()))),
        )

