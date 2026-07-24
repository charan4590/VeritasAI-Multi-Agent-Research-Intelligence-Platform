"""
Source credibility scoring — extended with more academic domains.

Changes: added sciencedirect.com, researchgate.net, semanticscholar.org,
plos.org, frontiersin.org, mdpi.com to TIER_1.
Added compute_research_confidence() that weights methodology/citation
coverage, not just domain credibility.
"""

import re
from typing import Dict
from urllib.parse import urlparse

TIER_1 = {
    # Core academic
    "arxiv.org",
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "scholar.google.com",
    "ieee.org",
    "acm.org",
    "springer.com",
    "sciencedirect.com",
    "researchgate.net",
    "semanticscholar.org",
    "cell.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
    "plos.org",
    "frontiersin.org",
    "mdpi.com",
    "bmj.com",
    "wiley.com",
    "tandfonline.com",
    "sagepub.com",
    # Reference
    "wikipedia.org",
    "britannica.com",
}

TIER_2 = {
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "wsj.com",
    "bloomberg.com",
    "ft.com",
    "economist.com",
    "forbes.com",
    "techcrunch.com",
    "wired.com",
    "arstechnica.com",
    "theverge.com",
    "zdnet.com",
    "mit.edu",
    "stanford.edu",
    "harvard.edu",
    "oxford.ac.uk",
    "github.com",
    "stackoverflow.com",
    "docs.python.org",
    "openai.com",
    "anthropic.com",
    "deepmind.com",
    "huggingface.co",
    "kaggle.com",
    "towardsdatascience.com",
    # Bug fix: this list was entirely academic/tech-and-news, so a
    # perfectly legitimate general-interest question (e.g. "top 10 car
    # engines produced between 1990 and 2000") had every one of its real,
    # authoritative sources fall through to the generic 50/"Low" default
    # below purely because their domain wasn't on the list — not because
    # the sources were actually low quality. Added recognized authorities
    # for a handful of other common everyday research categories so
    # confidence scoring isn't implicitly academic-topics-only.
    # Automotive
    "caranddriver.com",
    "motortrend.com",
    "edmunds.com",
    "autoblog.com",
    "roadandtrack.com",
    "topgear.com",
    "autoweek.com",
    "hagerty.com",
    "motor1.com",
    "autoevolution.com",
    # General reference / explainers
    "history.com",
    "howstuffworks.com",
    "smithsonianmag.com",
    "nationalgeographic.com",
    # Finance
    "investopedia.com",
    "cnbc.com",
    "marketwatch.com",
    # Health (mainstream, non-academic)
    "mayoclinic.org",
    "webmd.com",
    "healthline.com",
    # Sports
    "espn.com",
}

TIER_3_TLDS = {".edu", ".gov", ".org", ".ac.uk", ".ac.in"}


def get_domain(url: str) -> str:
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def score_url(url: str) -> int:
    domain = get_domain(url)
    if not domain:
        return 40
    if domain in TIER_1:
        return 95
    if domain in TIER_2:
        return 80
    for tld in TIER_3_TLDS:
        if domain.endswith(tld):
            return 70
    if any(x in domain for x in ["reddit.com", "quora.com"]):
        return 45
    if "medium.com" in domain or "substack.com" in domain:
        return 52
    # Bug fix: this used to be 50, which score_label() classifies as
    # "Low" — meaning ANY domain not explicitly whitelisted above (most
    # of the internet, for any topic outside academia/tech/mainstream
    # news) was automatically treated as low-credibility by default, not
    # because of anything about the source itself. 55 lands an unknown-
    # but-otherwise-normal .com in "Medium" instead — still clearly below
    # a recognized authority, but not punished as if it were a forum
    # post. TIER_2 above is expanded per-category as real gaps show up;
    # this default is the fallback for everything category-list still
    # doesn't cover, so it shouldn't sit at the bottom by default.
    return 55


def score_label(score: int) -> str:
    if score >= 90:
        return "High"
    if score >= 70:
        return "Good"
    if score >= 55:
        return "Medium"
    return "Low"


def compute_confidence(sources: Dict, citations_used: list) -> int:
    """Original confidence — domain credibility × citation coverage.
    Used for general/technical intent (see main.py) — compute_research_confidence
    below is the academic-intent equivalent."""
    if not citations_used:
        return 0
    scores = [score_url(sources[i]["url"]) for i in citations_used if i in sources]
    if not scores:
        return 0
    avg_credibility = sum(scores) / len(scores)
    # Bug fix: this divisor was 5, meaning a well-answered general
    # question citing 2-3 genuinely good sources (which is normal — not
    # every everyday question needs or has 5 independent citable sources
    # the way an academic survey does) got capped at 40-60% of its
    # deserved credibility score for no reason related to source
    # quality. 3 still rewards citing more sources, just without
    # penalizing questions that are legitimately answered well with fewer.
    citation_factor = min(1.0, len(citations_used) / 3)
    return int(avg_credibility * citation_factor)


def compute_research_confidence(
    sources: Dict,
    citations_used: list,
    report: str,
) -> int:
    """
    F: Improved confidence for research reports.
    Factors:
      - Source quality (40%): avg credibility of cited sources
      - Citation density (25%): citations per paragraph
      - Academic source ratio (20%): % of cited sources from TIER_1
      - Content coverage (15%): presence of methodology/results keywords in report
    """
    if not citations_used or not sources:
        return 0

    # Factor 1: source quality
    cited = [sources[i] for i in citations_used if i in sources]
    scores = [score_url(s["url"]) for s in cited]
    avg_quality = sum(scores) / len(scores) if scores else 50

    # Factor 2: citation density
    paragraphs = [p for p in report.split("\n\n") if len(p.strip()) > 40]
    cited_paras = sum(1 for p in paragraphs if re.search(r"\[\d+\]", p))
    density = (cited_paras / len(paragraphs) * 100) if paragraphs else 0

    # Factor 3: academic source ratio
    academic_count = sum(1 for s in cited if score_url(s["url"]) >= 90)
    academic_ratio = (academic_count / len(cited) * 100) if cited else 0

    # Factor 4: content coverage — check for research sections in report
    report_lower = report.lower()
    coverage_keywords = [
        "architecture",
        "dataset",
        "accuracy",
        "results",
        "methodology",
        "experiment",
        "proposed",
        "model",
    ]
    coverage_hits = sum(1 for kw in coverage_keywords if kw in report_lower)
    coverage_score = min(100, coverage_hits * 12.5)

    # Weighted combination
    final = int(avg_quality * 0.40 + density * 0.25 + academic_ratio * 0.20 + coverage_score * 0.15)
    return min(100, final)
