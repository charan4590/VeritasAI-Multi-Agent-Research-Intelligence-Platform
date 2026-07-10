"""
Phase 3 Milestone 1: agent classes.

Every pipeline step except `reflect` (deliberately left as a plain
function this milestone — no ReflectAgent yet, see the Phase 3 plan)
now lives here as an Agent subclass. graph.py imports from this package
to build the LangGraph StateGraph; nothing in here imports graph.py, so
there's no circular import.
"""

from .base import Agent
from .intent import _detect_research_intent, ACADEMIC_SIGNALS, TECHNICAL_SIGNALS
from .planner import PlannerAgent, parse_json
from .search import WebResearchAgent, AcademicSearchAgent
from .pdf_agent import PDFAgent
from .supervisor import SupervisorAgent
from .rag_agent import RAGAgent
from .report_generator import ReportGeneratorAgent
from .citation import CitationAgent

__all__ = [
    "Agent",
    "_detect_research_intent", "ACADEMIC_SIGNALS", "TECHNICAL_SIGNALS",
    "PlannerAgent", "parse_json",
    "WebResearchAgent", "AcademicSearchAgent", "PDFAgent",
    "SupervisorAgent",
    "RAGAgent",
    "ReportGeneratorAgent",
    "CitationAgent",
]
