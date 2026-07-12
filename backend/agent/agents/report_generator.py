"""
ReportGeneratorAgent — Phase 3 Milestone 1.

Moved from graph.py's synthesize_node. Prompt-building, RAG-context
assembly, and the Milestone 1 token-streaming logic (including
StreamAborted handling on client disconnect) are byte-for-byte unchanged
from before this milestone.
"""

import logging

from ..credibility import score_url
from ..llm import get_llm
from ..rag import format_retrieved_context
from ..reflection import post_synthesis_check
from ..state import AgentState, StreamAborted
from .base import Agent
from .intent import _detect_research_intent

logger = logging.getLogger(__name__)


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
            "Provide at least 4-5 SPECIFIC limitations, each its own bullet with a "
            "brief explanation of WHY it matters (a label alone like 'Class imbalance' "
            "is not enough). Cover multiple categories where applicable: "
            "(1) dataset limitations — size, diversity, class imbalance, labeling quality; "
            "(2) methodological limitations — architecture constraints, computational cost, "
            "training time; "
            "(3) evaluation limitations — lack of external/clinical validation, "
            "single-dataset testing, missing baselines or ablations; "
            "(4) generalizability — performance on unseen populations, imaging equipment, "
            "or conditions not represented in the training data. "
            "Ground each limitation in what the sources actually say where possible; cite with [n].\n\n"
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


class ReportGeneratorAgent(Agent):
    """Writes the final report from gathered sources / RAG chunks,
    streaming tokens incrementally when a stream_callback is present
    (Milestone 1), then runs a non-blocking post-synthesis quality check."""

    name = "synthesize"

    def trace_inputs(self, state: AgentState):
        return {
            "source_count": len(state.get("sources", {})),
            "retrieved_chunks": len(state.get("retrieved_chunks", [])),
        }

    def run(self, state: AgentState) -> dict:
        retrieved = state.get("retrieved_chunks", [])
        intent = _detect_research_intent(state["question"])

        def _label(s) -> str:
            # Milestone 2: visually distinguishes uploaded-document
            # sources from web sources in the prompt itself, so the model
            # can naturally write "per the uploaded document [n]" when
            # appropriate. Only touches the raw/fallback source listing
            # below — format_retrieved_context() (rag.py) is untouched,
            # so the RAG-retrieved-chunk section of the prompt is exactly
            # as before this milestone.
            return "[PDF] " if s.get("source_type", "web") == "pdf" else ""

        if retrieved:
            rag_context = format_retrieved_context(retrieved)
            sorted_src = sorted(
                state["sources"].values(), key=lambda s: score_url(s.get("url", "")), reverse=True
            )
            max_raw = 10 if intent == "academic" else 5
            raw_context = "\n\n".join(
                f"[{s['id']}] {_label(s)}{s['title']}\n{s['url']}\n{s['snippet'][:600]}"
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
                state["sources"].values(), key=lambda s: score_url(s.get("url", "")), reverse=True
            )
            context_block = "\n\n".join(
                f"[{s['id']}] {_label(s)}{s['title']}\n{s['url']}\n{s['snippet']}" for s in sorted_src
            )
            cite_note = "Cite claims using [n] markers from the source ids."

        system_prompt = _build_synthesis_prompt(state["question"], intent)
        system_prompt += f"\n\n{cite_note} Only use source ids that appear in the list above."

        temp = 0.1 if intent == "academic" else 0.2
        llm = get_llm(temperature=temp)

        # Milestone 1: stream tokens incrementally instead of one blocking
        # call. stream_callback (if present in state) receives partial
        # text so the frontend can show live generation instead of
        # "Working on it..." for the full synthesis duration.
        stream_callback = state.get("stream_callback")
        full_text = ""
        if stream_callback:
            try:
                for chunk in llm.stream(
                    [
                        ("system", system_prompt),
                        (
                            "human",
                            f"Research topic: {state['question']}\n\n"
                            f"{context_block}\n\n"
                            f"Write the full structured report now. Every section is required.",
                        ),
                    ]
                ):
                    token = getattr(chunk, "content", "") or ""
                    if token:
                        full_text += token
                        stream_callback(token)
            except StreamAborted:
                # Client disconnected mid-stream. Do NOT fall back to a
                # blocking llm.invoke() here — that would spend an extra
                # LLM call finishing a report nobody will see. Propagate
                # so the run stops cleanly one level up.
                logger.info("Synthesis streaming aborted — client disconnected")
                raise
            except Exception as exc:
                logger.warning(f"Streaming synthesis failed, falling back to blocking call: {exc}")
                full_text = ""

        if not full_text:
            response = llm.invoke(
                [
                    ("system", system_prompt),
                    (
                        "human",
                        f"Research topic: {state['question']}\n\n"
                        f"{context_block}\n\n"
                        f"Write the full structured report now. Every section is required.",
                    ),
                ]
            )
            full_text = response.content

        # Post-synthesis reflection — verify report actually satisfies query
        check = post_synthesis_check(state["question"], full_text)
        log_msg = f"Synthesized {intent} report ({'RAG chunks' if retrieved else 'raw sources'}, {len(state['sources'])} sources)"
        if not check["satisfies_query"]:
            log_msg += f" — quality concern: {check['reasoning']}"
            logger.warning(f"Post-synthesis check flagged weak sections: {check['weak_sections']}")

        return {
            "report": full_text,
            "log": state["log"] + [log_msg],
        }
