"""
PlannerAgent — Phase 3 Milestone 1.
"""

import json
import logging
import re

from ..llm import get_llm
from ..memory import format_memory_context, get_conversation_context
from ..state import AgentState
from .base import Agent
from .intent import _detect_research_intent

logger = logging.getLogger(__name__)


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


class PlannerAgent(Agent):
    name = "planner"
    uses_llm = True  # calls get_llm() below — see base.py for what this enables

    def run(self, state: AgentState) -> dict:
        _q = state["question"].strip().lower()

        GREETINGS = [
            "hi", "hello", "hey", "thanks", "thank you", "bye", "goodbye",
            "ok", "okay", "yes", "no", "sup", "good morning", "good evening",
            "good night", "how are you", "how r u", "hi how are you",
            "hi how r u", "hii", "hiii", "heyyy", "hi how are u",
            "what's up", "whats up", "howdy",
        ]
        is_greeting = any(
            _q == g or _q == g + "?" or _q == g + "!"
            or _q.startswith(g + " ") or _q.startswith(g + ",")
            for g in GREETINGS
        )

        RESEARCH_WORDS = [
            "deep learning", "machine learning", "neural", "model", "algorithm",
            "predict", "detect", "classify", "cancer", "disease", "medical",
            "study", "research", "paper", "review", "analyze", "compare",
            "effect", "impact", "method", "approach", "technique", "system",
            "performance", "accuracy", "dataset", "training", "network",
            "what is", "explain", "describe", "survey", "analysis", "using",
            "based on", "for", "with", "detection", "classification",
        ]
        has_research_content = any(w in _q for w in RESEARCH_WORDS)

        if is_greeting or (not has_research_content and len(_q.split()) < 5):
            return {
                "report": "Please ask a research question. Example: 'What are the latest advances in deep learning for medical imaging?'",
                "citations_used": [],
                "log": state["log"] + ["Query rejected: not a research question"],
            }

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
                "  Query 2: Datasets + benchmarks used in this area\n"
                "  Query 3: Evaluation metrics + experimental results\n"
                "  Query 4: Recent papers 2022-2025 on this exact topic\n"
                "  Query 5: SOTA comparison methods\n\n"
                "RULES:\n"
                "- Use technical terminology, model names, metric names\n"
                "- Do NOT write 'overview of', 'introduction to', 'what is'\n"
                "- Prefer queries that return arxiv, IEEE, PubMed, Springer results"
            )
        elif intent == "technical":
            system_prompt = (
                "You are a technical research planner.\n"
                "Respond ONLY with a JSON object.\n"
                'Format: {"queries": ["q1", "q2", "q3", "q4"]}\n'
                "Generate 4 queries: implementation details, performance benchmarks, "
                "best practices, common pitfalls."
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