"""Unit tests for RAG chunking and utilities (no Ollama/ChromaDB required)."""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.rag import chunk_text, chunk_source, format_retrieved_context


class TestChunking:
    def test_short_text_returns_single_chunk(self):
        text = "This is a short text."
        chunks = chunk_text(text, chunk_size=300)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_splits_into_chunks(self):
        words = ["word"] * 700
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=300, overlap=50)
        assert len(chunks) > 1
        # Each chunk should be at most chunk_size words
        for chunk in chunks:
            assert len(chunk.split()) <= 300

    def test_overlap_creates_shared_content(self):
        words = [f"word{i}" for i in range(400)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=200, overlap=50)
        # Last word of chunk N should appear in start of chunk N+1
        if len(chunks) >= 2:
            c1_words = set(chunks[0].split())
            c2_words = set(chunks[1].split())
            overlap = c1_words & c2_words
            assert len(overlap) > 0

    def test_empty_text_returns_single_empty_chunk(self):
        chunks = chunk_text("")
        assert chunks == [""]


class TestChunkSource:
    def test_source_without_snippet_returns_empty(self):
        source = {"id": 1, "url": "https://example.com", "title": "Test", "snippet": ""}
        result = chunk_source(source)
        assert result == []

    def test_source_with_snippet_returns_chunks(self):
        source = {
            "id": 1, "url": "https://example.com",
            "title": "Test Article", "snippet": "This is the content. " * 50
        }
        result = chunk_source(source)
        assert len(result) >= 1
        assert all(c["source_id"] == 1 for c in result)
        assert all(c["url"] == "https://example.com" for c in result)
        assert all("chunk_id" in c for c in result)

    def test_chunk_ids_are_unique(self):
        source = {
            "id": 5, "url": "https://test.com",
            "title": "Title", "snippet": "word " * 400
        }
        chunks = chunk_source(source)
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


class TestFormatContext:
    def test_empty_chunks_returns_empty_string(self):
        result = format_retrieved_context([])
        assert result == ""

    def test_chunks_formatted_with_relevance(self):
        chunks = [
            {"text": "AI is transforming industries.", "source_id": "1",
             "url": "https://example.com", "title": "AI Article", "relevance": 0.92},
            {"text": "Machine learning advances rapidly.", "source_id": "2",
             "url": "https://other.com", "title": "ML News", "relevance": 0.78},
        ]
        result = format_retrieved_context(chunks)
        assert "0.92" in result
        assert "0.78" in result
        assert "AI Article" in result
        assert "ML News" in result
