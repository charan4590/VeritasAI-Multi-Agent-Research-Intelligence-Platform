"""
Research agent graph — production version with academic research optimization.

Pipeline:
  planner → search → reflect → rag → synthesize → validate

Key improvements:
  B. Research intent detection — expanded signal set, lower threshold
  A. academic_web_search() used for academic queries (2-pass + domain sort)
  A. fetch_full_content() enriches snippets from academic URLs
  D. smart_reflect() checks methodology/dataset/metrics/results
  E. _build_synthesis_prompt() enforces 9-section research paper structure
  C. Sources sorted by credibility before synthesis context is built
"""

import os
import re
import json
import uuid
import logging
from typing import Optional

from langgraph.graph import StateGraph, START, END

from .state import AgentState, Source, ReflectionDecision, StreamAborted
from .tools import web_search, academic_web_search, fetch_full_content, academic_web_search_batch, web_search_batch
from .llm import get_llm
from .rag import index_sources, retrieve_relevant_chunks, format_retrieved_context
from .reranker import rerank_chunks, is_reranker_available
from .reflection import smart_reflect, post_synthesis_check
from .memory import retrieve_memories, format_memory_context, get_conversation_context
from .credibility import score_url

logger = logging.getLogger(__name__)

RETRIEVAL_CANDIDATES = int(os.environ.get("RAG_CANDIDATES", "25"))
RERANK_TOP_K = int(os.environ.get("RERANK_TOP_K", "8"))


def parse_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# B: Research intent detection — expanded signals, score >= 1 for academic
# ---------------------------------------------------------------------------

ACADEMIC_SIGNALS = {
    # Method signals
    "novel", "proposed", "hybrid", "deep learning", "neural network",
    "cnn", "lstm", "transformer", "bert", "resnet", "vgg", "attention",
    "encoder", "decoder", "autoencoder", "gan", "diffusion",
    # Task signals
    "classification", "detection", "segmentation", "recognition",
    "prediction", "diagnosis", "prognosis", "screening",
    # Research signals
    "architecture", "dataset", "benchmark", "evaluation", "accuracy",
    "f1", "auc", "roc", "precision", "recall", "sensitivity", "specificity",
    "sota", "state of the art", "baseline", "ablation", "experiment",
    "survey", "framework", "methodology", "model",
    # Domain signals
    "cancer", "tumor", "medical imaging", "ct scan", "mri", "pathology",
    "radiology", "histology", "genomics", "drug", "clinical",
    "arxiv", "ieee", "paper", "journal", "conference",
}

TECHNICAL_SIGNALS = {
    "how does", "how to", "implement", "build", "create", "deploy",
    "configure", "install", "library", "api", "code", "tutorial",
}


def _detect_research_intent(question: str) -> str:
    """
    B: Classify query intent.
    Academic threshold lowered to 1 signal — any single academic term
    (e.g. 'neural network', 'novel', 'CNN') triggers academic mode.
    This prevents generic-mode fallback for clearly research queries.
    """
    q = question.lower()
    academic_score = sum(1 for s in ACADEMIC_SIGNALS if s in q)
    technical_score = sum(1 for s in TECHNICAL_SIGNALS if s in q)

    if academic_score >= 1:
        return "academic"
    if technical_score >= 2:
        return "technical"
    return "general"


# ---------------------------------------------------------------------------
# Planner — research-mode queries target academic databases
# ---------------------------------------------------------------------------

def planner_node(state: AgentState) -> dict:
    memory_context = format_memory_context(state.get("memories", []))
    conv_context = get_conversation_context()
    intent = _detect_research_intent(state["question"])

    if intent == "academic":
        system_prompt = (
            "You are an expert academic research planner. "
            "Respond ONLY with a JSON object.\n"
            'Format: {"queries": ["q1", "q2", "q3", "q4", "q5"]}\n\n'
            "Generate exactly 5 search queries. Each query must target a DIFFERENT aspect:\n"
            "  Query 1: The EXACT method/architecture name + 'deep learning' + domain\n"
            "           Example: 'hybrid CNN LSTM lung cancer nodule detection'\n"
            "  Query 2: Datasets + benchmarks used in this area\n"
            "           Example: 'LUNA16 LIDC-IDRI lung nodule dataset benchmark'\n"
            "  Query 3: Evaluation metrics + experimental results\n"
            "           Example: 'lung cancer detection sensitivity specificity AUC results'\n"
            "  Query 4: Recent papers 2022-2025 on this exact topic\n"
            "           Example: 'lung cancer early detection deep learning 2023 arxiv'\n"
            "  Query 5: SOTA comparison methods\n"
            "           Example: 'lung cancer detection transformer ResNet comparison SOTA'\n\n"
            "RULES:\n"
            "- Use technical terminology, model names, metric names\n"
            "- Do NOT write 'overview of', 'introduction to', 'what is'\n"
            "- Include domain-specific terminology from the question\n"
            "- Prefer queries that return arxiv, IEEE, PubMed, Springer results"
        )
    elif intent == "technical":
        system_prompt = (
            "You are a technical research planner.\n"
            "Respond ONLY with a JSON object.\n"
            'Format: {"queries": ["q1", "q2", "q3", "q4"]}\n'
            "Generate 4 queries: implementation details, performance benchmarks, "
            "best practices, common pitfalls. Use technical terms and version numbers."
        )
    else:
        system_prompt = (
            "You are a research planner. Respond ONLY with a JSON object.\n"
            'Format: {"queries": ["q1", "q2", "q3"]}\n'
            "Produce 3 to 5 specific, non-overlapping search queries."
        )

    if memory_context:
        system_prompt += f"\n\n{memory_context}\nUse this context to plan more targeted queries."
    if conv_context:
        system_prompt += f"\n\n{conv_context}"

    llm = get_llm()
    response = llm.invoke([
        ("system", system_prompt),
        ("human", state["question"]),
    ])

    data = parse_json(response.content)
    queries = data.get("queries") or [state["question"]]

    logger.info(f"Planner: {len(queries)} {intent} queries")
    return {
        "plan": queries,
        "round": 0,
        "log": state["log"] + [
            f"Planned {len(queries)} {intent} search queries"
            + (" (memory-enhanced)" if memory_context else "")
            + (" (context-aware)" if conv_context else "")
        ],
    }


# ---------------------------------------------------------------------------
# A: Search — academic queries use academic_web_search + full content fetch
# ---------------------------------------------------------------------------

def search_node(state: AgentState) -> dict:
    reflection = state.get("reflection")
    queries = state["plan"] if state["round"] == 0 else (
        reflection.follow_up_queries if reflection else []
    )

    intent = _detect_research_intent(state["question"])
    sources = dict(state["sources"])
    next_id = max(sources.keys(), default=0) + 1
    seen_urls = {s["url"] for s in sources.values()}

    MAX_SOURCES = 20
    MAX_FULL_FETCHES = 6
    full_fetch_count = 0

    # #1: Run all queries concurrently instead of sequentially —
    # was the single largest latency contributor (5 queries x ~7s each = 35s+ serial)
    if intent == "academic":
        batch_results = academic_web_search_batch(queries, max_results=4)
    else:
        batch_results = web_search_batch(queries, max_results=5)

    for query in queries:
        if len(sources) >= MAX_SOURCES:
            break
        results = batch_results.get(query, [])

        for result in results:
            if len(sources) >= MAX_SOURCES:
                break
            url = result.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            snippet = (result.get("content", "") or result.get("snippet", "") or "")[:1500]

            # Only fetch full content for a small number of top sources —
            # this was the main bottleneck (blocking HTTP call per source)
            if full_fetch_count < MAX_FULL_FETCHES and score_url(url) >= 90:
                enriched = fetch_full_content(url)
                if enriched and len(enriched) > len(snippet):
                    snippet = snippet + "\n\n[Full content]:\n" + enriched[:1500]
                full_fetch_count += 1

            sources[next_id] = Source(
                id=next_id,
                url=url,
                title=result.get("title", url),
                snippet=snippet[:3000],
            )
            next_id += 1

    sorted_sources = dict(
        sorted(
            sources.items(),
            key=lambda x: score_url(x[1].get("url", "")),
            reverse=True,
        )
    )

    logger.info(f"Search round {state['round']+1}: {len(sorted_sources)} sources (intent: {intent}, full-fetched: {full_fetch_count})")
    return {
        "sources": sorted_sources,
        "round": state["round"] + 1,
        "log": state["log"] + [f"Round {state['round']+1}: gathered {len(sorted_sources)} sources"],
    }


# ---------------------------------------------------------------------------
# Reflect — uses smart_reflect from reflection.py (already has D checks)
# ---------------------------------------------------------------------------

def reflect_node(state: AgentState) -> dict:
    if state["round"] >= state["max_rounds"]:
        decision = ReflectionDecision(
            sufficient=True, follow_up_queries=[],
            reasoning="Reached maximum search rounds."
        )
        return {
            "reflection": decision,
            "log": state["log"] + ["Max rounds reached — moving to RAG + synthesis"],
        }

    decision = smart_reflect(
        question=state["question"],
        sources=state["sources"],
        current_round=state["round"],
        max_rounds=state["max_rounds"],
    )

    status = "sufficient" if decision.sufficient else f"gap: {decision.reasoning}"
    logger.info(f"Reflection round {state['round']}: {status}")

    return {
        "reflection": decision,
        "log": state["log"] + [f"Reflection: {decision.reasoning}"],
    }


def route_after_reflect(state: AgentState) -> str:
    r = state.get("reflection")
    return "search" if (r and not r.sufficient) else "rag"


# ---------------------------------------------------------------------------
# RAG — two-stage retrieval: bi-encoder + cross-encoder rerank
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    session_id = state.get("rag_session_id") or str(uuid.uuid4())[:8]
    log_entries = [f"RAG: indexing {len(state['sources'])} sources..."]

    success = index_sources(state["sources"], session_id)

    if success:
        candidates = retrieve_relevant_chunks(
            state["question"], session_id, top_k=RETRIEVAL_CANDIDATES,
        )
        log_entries.append(f"RAG: retrieved {len(candidates)} candidate chunks")

        if candidates and is_reranker_available():
            chunks = rerank_chunks(state["question"], candidates, top_k=RERANK_TOP_K)
            log_entries.append(f"Re-ranked → top {len(chunks)} chunks")
        elif candidates:
            chunks = candidates[:RERANK_TOP_K]
            log_entries.append(f"Top {len(chunks)} chunks (no cross-encoder)")
        else:
            chunks = []
            log_entries.append("RAG: no chunks — using raw sources")
    else:
        chunks = []
        log_entries.append("RAG: unavailable — using raw sources")

    return {
        "rag_session_id": session_id,
        "retrieved_chunks": chunks,
        "log": state["log"] + log_entries,
    }


# ---------------------------------------------------------------------------
# E: Synthesize — 9-section research paper structure for academic queries
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(question: str, intent: str) -> str:
    if intent == "academic":
        return (
            "You are an expert AI researcher writing a structured technical report.\n"
            "Write the report in markdown using EXACTLY these 9 sections in this order:\n\n"
            "## 1. Introduction\n"
            "State the problem, its clinical/practical importance, and motivation for "
            "using deep learning. Be specific to the query domain. (2-3 paragraphs)\n\n"
            "## 2. Related Work\n"
            "Summarize prior methods chronologically. Name specific models, papers, "
            "and their reported results. Cite with [n]. (2-3 paragraphs)\n\n"
            "## 3. Proposed Method\n"
            "Describe the overall system design, pipeline stages, and key innovations. "
            "What makes this approach different from prior work? Cite with [n].\n\n"
            "## 4. Model Architecture\n"
            "Detail the neural network architecture: layer types (Conv2D, LSTM, "
            "Transformer blocks, attention heads), input/output dimensions, activation "
            "functions, skip connections, loss functions. Use bullet points for components. "
            "If multiple architectures compared, list each. Cite with [n].\n\n"
            "## 5. Dataset\n"
            "For EACH dataset mentioned: name, size (number of samples/patients/images), "
            "modality (CT/MRI/X-ray/histology), class distribution, source/availability, "
            "preprocessing steps. If not reported in sources, state explicitly. Cite with [n].\n\n"
            "## 6. Experimental Results\n"
            "Report EXACT numbers from sources: accuracy (%), sensitivity (%), specificity (%), "
            "AUC, F1-score, Dice coefficient, etc. Use a markdown table:\n"
            "| Method | Dataset | Accuracy | AUC | Sensitivity | Specificity |\n"
            "|-|-|-|-|-|-|\n"
            "Fill with values from sources. Mark unknown cells as '-'. Cite with [n].\n\n"
            "## 7. Comparison with Existing Methods\n"
            "Compare the proposed method against baselines and SOTA. Explain WHY it "
            "performs better or worse. Include a comparison table if multiple methods cited.\n\n"
            "## 8. Limitations\n"
            "List specific limitations: dataset size, generalizability, computational cost, "
            "class imbalance, lack of external validation, etc. Be specific.\n\n"
            "## 9. References\n"
            "List all cited sources as [n] Title — URL\n\n"
            "ABSOLUTE RULES:\n"
            "1. Cite EVERY factual claim with [n] — use ONLY source ids from the provided list\n"
            "2. NEVER invent numbers — if a metric is not in sources, write 'not reported'\n"
            "3. Section 4 (Model Architecture) MUST name specific layer types\n"
            "4. Section 5 (Dataset) MUST include sample counts if available\n"
            "5. Section 6 MUST include a comparison table\n"
            "6. Do NOT write generic background paragraphs without citations"
        )
    elif intent == "technical":
        return (
            "You are a senior engineer writing precise technical documentation.\n"
            "Structure: ## Overview, ## Architecture, ## Implementation, "
            "## Performance & Benchmarks, ## Best Practices, ## Known Issues.\n"
            "Be specific: include version numbers, code concepts, configuration details.\n"
            "Cite every technical claim with [n]."
        )
    else:
        return (
            "You are a research analyst. Write a well-organized report in markdown.\n"
            "Cite claims with [n] markers. Use clear headings and concise paragraphs.\n"
            "Only cite ids from the provided sources."
        )


def synthesize_node(state: AgentState) -> dict:
    retrieved = state.get("retrieved_chunks", [])
    intent = _detect_research_intent(state["question"])

    if retrieved:
        rag_context = format_retrieved_context(retrieved)
        sorted_src = sorted(
            state["sources"].values(),
            key=lambda s: score_url(s.get("url", "")),
            reverse=True
        )
        max_raw = 10 if intent == "academic" else 5
        raw_context = "\n\n".join(
            f"[{s['id']}] {s['title']}\n{s['url']}\n{s['snippet'][:600]}"
            for s in sorted_src[:max_raw]
        )
        context_block = (
            f"{rag_context}\n\n"
            f"## All Sources (sorted by credibility — use for citation reference)\n"
            f"{raw_context}"
        )
        cite_note = "Cite with [source_id] numbers. Use semantically retrieved context as primary."
    else:
        sorted_src = sorted(
            state["sources"].values(),
            key=lambda s: score_url(s.get("url", "")),
            reverse=True
        )
        context_block = "\n\n".join(
            f"[{s['id']}] {s['title']}\n{s['url']}\n{s['snippet']}"
            for s in sorted_src
        )
        cite_note = "Cite claims using [n] markers from the source ids."

    system_prompt = _build_synthesis_prompt(state["question"], intent)
    system_prompt += f"\n\n{cite_note} Only use source ids that appear in the list above."

    temp = 0.1 if intent == "academic" else 0.2
    llm = get_llm(temperature=temp)

    # #6 / Milestone 1: Stream tokens incrementally instead of one blocking
    # call. stream_callback (if present in state) receives partial text so
    # the frontend can show live generation instead of "Working on it..."
    # for the full synthesis duration. main.py is now the caller that
    # actually supplies this — see _run_research_stream().
    stream_callback = state.get("stream_callback")
    full_text = ""
    if stream_callback:
        try:
            for chunk in llm.stream([
                ("system", system_prompt),
                ("human",
                 f"Research topic: {state['question']}\n\n"
                 f"{context_block}\n\n"
                 f"Write the full structured report now. Every section is required."),
            ]):
                token = getattr(chunk, "content", "") or ""
                if token:
                    full_text += token
                    stream_callback(token)
        except StreamAborted:
            # Client disconnected mid-stream (see StreamAborted docstring).
            # Do NOT fall back to a blocking llm.invoke() here — that would
            # spend an extra LLM call finishing a report nobody will see.
            # Propagate so the run stops cleanly one level up.
            logger.info("Synthesis streaming aborted — client disconnected")
            raise
        except Exception as exc:
            logger.warning(f"Streaming synthesis failed, falling back to blocking call: {exc}")
            full_text = ""

    if not full_text:
        response = llm.invoke([
            ("system", system_prompt),
            ("human",
             f"Research topic: {state['question']}\n\n"
             f"{context_block}\n\n"
             f"Write the full structured report now. Every section is required."),
        ])
        full_text = response.content

    # #4: Post-synthesis reflection — verify report actually satisfies query
    check = post_synthesis_check(state["question"], full_text)
    log_msg = f"Synthesized {intent} report ({'RAG chunks' if retrieved else 'raw sources'}, {len(state['sources'])} sources)"
    if not check["satisfies_query"]:
        log_msg += f" — quality concern: {check['reasoning']}"
        logger.warning(f"Post-synthesis check flagged weak sections: {check['weak_sections']}")

    return {
        "report": full_text,
        "log": state["log"] + [log_msg],
    }


# ---------------------------------------------------------------------------
# Validate — strip hallucinated citations
# ---------------------------------------------------------------------------

CITATION_RE = re.compile(r"\[(\d+)\]")


def validate_node(state: AgentState) -> dict:
    report = state["report"]
    valid_ids = set(state["sources"].keys())
    found_ids = {int(m) for m in CITATION_RE.findall(report)}
    used_ids = sorted(found_ids & valid_ids)

    for bad in found_ids - valid_ids:
        report = re.sub(rf"\[{bad}\]", "", report)

    if used_ids:
        refs = "\n".join(
            f"[{i}] {state['sources'][i]['title']} — {state['sources'][i]['url']}"
            for i in used_ids
        )
        report += f"\n\n---\n\n**Sources**\n\n{refs}"
    else:
        report += "\n\n---\n\n*No verifiable citations produced.*"

    hallucinated = found_ids - valid_ids
    msg = f"Validated {len(used_ids)} citations"
    if hallucinated:
        msg += f" — removed {len(hallucinated)} hallucinated"

    return {
        "report": report,
        "citations_used": used_ids,
        "log": state["log"] + [msg],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("search", search_node)
    g.add_node("reflect", reflect_node)
    g.add_node("rag", rag_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("validate", validate_node)

    g.add_edge(START, "planner")
    g.add_edge("planner", "search")
    g.add_edge("search", "reflect")
    g.add_conditional_edges(
        "reflect", route_after_reflect,
        {"search": "search", "rag": "rag"},
    )
    g.add_edge("rag", "synthesize")
    g.add_edge("synthesize", "validate")
    g.add_edge("validate", END)
    return g.compile()


def initial_state(question: str, max_rounds: int = 2,
                  memories: Optional[list] = None,
                  stream_callback=None) -> AgentState:
    return AgentState(
        question=question, plan=[], sources={}, round=0,
        max_rounds=max_rounds, reflection=None,
        report="", citations_used=[], log=[],
        retrieved_chunks=[], rag_session_id=None,
        memories=memories or [],
        stream_callback=stream_callback,
    )
