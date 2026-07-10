"""
PlannerAgent — Phase 3 Milestone 1.

Moved from graph.py's planner_node. The prompt-building and LLM-call
logic below is byte-for-byte unchanged from before this milestone; only
its home (a class instead of a bare function) and its instrumentation
(inherited from Agent.__call__, see base.py) are new.
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
    """Breaks the research question into targeted search queries, with a
    different prompt template per detected intent (academic/technical/
    general) and optional memory/conversation context injected."""

    name = "planner"

    def trace_inputs(self, state: AgentState):
        return {"question": state.get("question", "")[:200]}

    def run(self, state: AgentState) -> dict:
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
        response = llm.invoke(
            [
                ("system", system_prompt),
                ("human", state["question"]),
            ]
        )

        data = parse_json(response.content)
        queries = data.get("queries") or [state["question"]]

        logger.info(f"Planner: {len(queries)} {intent} queries")
        return {
            "plan": queries,
            "round": 0,
            "log": state["log"]
            + [
                f"Planned {len(queries)} {intent} search queries"
                + (" (memory-enhanced)" if memory_context else "")
                + (" (context-aware)" if conv_context else "")
            ],
        }
