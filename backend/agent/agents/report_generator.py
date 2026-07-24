"""
ReportGeneratorAgent — Phase 3 Milestone 1.

Moved from graph.py's synthesize_node. Prompt-building, RAG-context
assembly, and the Milestone 1 token-streaming logic (including
StreamAborted handling on client disconnect) are byte-for-byte unchanged
from before this milestone.
"""

import logging
import re

from ..credibility import score_url
from ..llm import get_llm
from ..memory import format_memory_context, get_conversation_context
from ..rag import format_retrieved_context
from ..reflection import post_synthesis_check
from ..state import AgentState, StreamAborted
from .base import Agent
from .intent import _detect_research_intent

logger = logging.getLogger(__name__)


# Deliberately separate from _detect_research_intent (intent.py) rather
# than a new top-level intent value: academic/technical/general also
# drives planner.py's query strategy and supervisor.py's search routing,
# and this only needs to change the REPORT SHAPE for a subset of general
# questions ("top 10 X", "best X", "X vs Y") — not touch query planning
# or search routing at all. Keeping it a separate, narrow check means
# this feature can't regress anything else in the pipeline.
_LIST_STYLE_PATTERN = re.compile(
    r"\b(top|best|worst)\s+\d+\b"
    r"|\blist\s+of\b"
    r"|\bmost\s+(popular|common|important|notable)\b"
    r"|\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b",
    re.IGNORECASE,
)


def _is_list_style_question(question: str) -> bool:
    return bool(_LIST_STYLE_PATTERN.search(question))


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
            "IMPORTANT: this report synthesizes findings from the cited sources — "
            "it does not describe a system you invented or trained. Do not write "
            "'our proposed method' as if this were your own novel contribution. "
            "Instead, synthesize the most common or most effective methodology "
            "pattern(s) actually used across the cited sources (e.g. CNN + transfer "
            "learning, ensemble methods, attention mechanisms), attributing each "
            "specific technique to the source(s) that used it with [n]. If sources "
            "use varied or disagreeing approaches, say so explicitly rather than "
            "inventing a single unified system. Every sentence in this section "
            "should trace back to a citation — if you can't cite it, don't write it.\n\n"
            "## 4. Model Architecture\n"
            "Detail the SPECIFIC architectures actually described in the cited "
            "sources — layer types, backbones (e.g. ResNet-50, VGG-16), key "
            "components (attention, ensembling) — each attributed to the source "
            "that used it with [n]. Use bullet points, one per architecture "
            "actually reported in the literature. Do NOT invent a novel "
            "architecture, and do NOT state implementation details (layer counts, "
            "dimensions, activation functions) that aren't explicitly present in a "
            "cited source — write 'not specified in available sources' rather than "
            "guessing plausible-sounding numbers.\n\n"
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
            "6. Do NOT write generic background paragraphs without citations\n"
            "7. Draw on AT LEAST 6-8 DIFFERENT numbered sources across the whole report, "
            "not just the first 2-3 you see — spread citations across as many of the "
            "provided distinct source ids as the evidence actually supports. Do not "
            "repeatedly cite the same 2-3 sources while ignoring the rest of the list.\n"
            "8. Every paragraph of 2+ sentences must contain at least one [n] citation — "
            "a paragraph making factual claims with zero citations is not acceptable"
        )
    elif intent == "technical":
        return (
            "You are a senior engineer writing precise technical documentation.\n"
            "Structure: ## Overview, ## Architecture, ## Implementation, "
            "## Performance & Benchmarks, ## Best Practices, ## Known Issues.\n"
            "Be specific: include version numbers, code concepts, configuration details.\n"
            "Cite every technical claim with [n]. Draw on as many of the distinct "
            "provided source ids as the evidence supports — don't lean on just 2-3 "
            "sources when more are available. Every paragraph with factual claims "
            "should have at least one citation."
        )
    elif _is_list_style_question(question):
        # New: dedicated shape for "top 10 X" / "best X" / "X vs Y"
        # general questions, instead of forcing them through the same
        # free-form prompt as every other general question. Still fully
        # cited and still degrades gracefully (no invented items) — this
        # is a report-shape change only, not a new confidence formula or
        # intent classification.
        return (
            "You are a research analyst producing a ranked list / comparison report.\n"
            "Structure the report as:\n\n"
            "## Summary\n"
            "1-2 sentences directly answering the question.\n\n"
            "## Ranked List\n"
            "A numbered list of the strongest, most notable items (match the count the "
            "question asks for, if it specifies one). Each entry:\n"
            "**N. Item name** — 1-3 sentences of concrete detail (numbers, dates, specs, "
            "outcomes) explaining why it belongs on the list. Cite every factual claim with [n].\n\n"
            "## Comparison Table\n"
            "A markdown table comparing the listed items across the 2-4 attributes most "
            "relevant to this question (e.g. spec, year, notable use, outcome). Only include "
            "this section if the sources actually support a structured comparison — omit it "
            "rather than inventing values to fill cells.\n\n"
            "## Notes & Caveats\n"
            "Any disagreement between sources on ranking or inclusion, and anything this "
            "list does not cover.\n\n"
            "ABSOLUTE RULES:\n"
            "1. Cite every factual claim with [n] — use ONLY source ids from the provided list.\n"
            "2. Do not pad the list with items the sources don't actually support — a shorter, "
            "fully-cited list is correct; an invented entry is not.\n"
            "3. If the question asks for a specific count (e.g. 'top 10') and the sources only "
            "support fewer, say so explicitly rather than filling the rest with guesses.\n"
            "4. Draw on as many of the distinct provided source ids as the evidence supports — "
            "don't lean on just 2-3 sources for the whole list when more are available."
        )
    else:
        return (
            "You are a research analyst. Write a well-organized report in markdown.\n"
            "Cite claims with [n] markers. Use clear headings and concise paragraphs.\n"
            "Only cite ids from the provided sources. Draw on as many of the distinct "
            "provided source ids as the evidence supports — don't lean on just 2-3 sources "
            "when more are available. Every paragraph with factual claims should have "
            "at least one citation."
        )


class ReportGeneratorAgent(Agent):
    """Writes the final report from gathered sources / RAG chunks,
    streaming tokens incrementally when a stream_callback is present
    (Milestone 1), then runs a non-blocking post-synthesis quality check."""

    name = "synthesize"
    uses_llm = True  # calls get_llm() below — see base.py for what this enables

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

        # Bug fix: this agent never saw conversation/memory context at
        # all — PlannerAgent already had it (format_memory_context /
        # get_conversation_context, see planner.py), which quietly made
        # *search queries* smarter for a follow-up, but the actual report
        # text was always written from scratch with zero awareness of
        # what was just told to the user. That's why clicking a
        # "recommended follow-up" felt like starting an unrelated new
        # chat instead of continuing the conversation: the plumbing to
        # track prior turns already existed (memory.py), it just never
        # reached the one place that writes the words the user reads.
        memory_context = format_memory_context(state.get("memories", []))
        conv_context = get_conversation_context()
        if memory_context or conv_context:
            system_prompt += (
                "\n\n"
                "You are continuing an ongoing research conversation, not "
                "starting fresh. Below is what was already covered — if "
                "this new question is a follow-up to it, explicitly build "
                "on those prior findings (e.g. reference what's already "
                "known and focus on what's new) instead of re-deriving "
                "everything from zero. If this question is unrelated, "
                "ignore the context below and answer it on its own terms."
            )
            if memory_context:
                system_prompt += f"\n{memory_context}"
            if conv_context:
                system_prompt += f"\n{conv_context}"

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
