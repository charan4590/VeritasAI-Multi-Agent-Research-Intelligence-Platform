"""Shared test fixtures."""

import os
import pytest
import tempfile

# Use temp DB for all tests
os.environ.setdefault("DB_PATH", ":memory:")
os.environ.setdefault("TAVILY_API_KEY", "fake-key-for-testing")
os.environ.setdefault("CHROMA_PATH", tempfile.mkdtemp())


@pytest.fixture
def sample_sources():
    return {
        1: {
            "id": 1,
            "url": "https://arxiv.org/abs/123",
            "title": "AI Paper",
            "snippet": "Deep learning advances rapidly.",
        },
        2: {
            "id": 2,
            "url": "https://bbc.com/news/ai",
            "title": "BBC AI News",
            "snippet": "AI is transforming industries worldwide.",
        },
        3: {
            "id": 3,
            "url": "https://example.com/blog",
            "title": "Blog Post",
            "snippet": "Some thoughts on machine learning.",
        },
    }


@pytest.fixture
def sample_report():
    return """
## Introduction
AI agents are becoming increasingly capable [1].

## Key Developments
Recent advances show significant progress [2]. Multiple research groups have
contributed to this field [1][2].

## Conclusion
The future looks promising [3].

---

**References**

[1] "AI Paper," [Online]. Available: https://arxiv.org/abs/123
[2] "BBC AI News," [Online]. Available: https://bbc.com/news/ai
[3] "Blog Post," [Online]. Available: https://example.com/blog
"""
