"""
Phase 3 Milestone 1: shared intent-detection utility.

Moved out of graph.py — logic is byte-for-byte unchanged from before this
milestone. It now lives here (rather than in graph.py or in any single
agent module) because multiple agents need it independently (PlannerAgent
to pick a prompt template, SupervisorAgent to pick web vs. academic
search, ReportGeneratorAgent to pick a synthesis structure) and putting
it in graph.py would create a circular import: graph.py needs to import
the agent classes to register them as LangGraph nodes, and the agents
need intent detection — agents/*.py importing from graph.py while
graph.py imports from agents/ would be circular.

graph.py re-exports _detect_research_intent at its original import path
(`from .agents import _detect_research_intent`) so the one external
caller, main.py's `from agent.graph import _detect_research_intent`,
keeps working completely unchanged.
"""

# ---------------------------------------------------------------------------
# B: Research intent detection — expanded signals, score >= 1 for academic
# ---------------------------------------------------------------------------

ACADEMIC_SIGNALS = {
    # Method signals
    "novel",
    "proposed",
    "hybrid",
    "deep learning",
    "neural network",
    "cnn",
    "lstm",
    "transformer",
    "bert",
    "resnet",
    "vgg",
    "attention",
    "encoder",
    "decoder",
    "autoencoder",
    "gan",
    "diffusion",
    # Task signals
    "classification",
    "detection",
    "segmentation",
    "recognition",
    "prediction",
    "diagnosis",
    "prognosis",
    "screening",
    # Research signals
    "architecture",
    "dataset",
    "benchmark",
    "evaluation",
    "accuracy",
    "f1",
    "auc",
    "roc",
    "precision",
    "recall",
    "sensitivity",
    "specificity",
    "sota",
    "state of the art",
    "baseline",
    "ablation",
    "experiment",
    "survey",
    "framework",
    "methodology",
    "model",
    # Domain signals
    "cancer",
    "tumor",
    "medical imaging",
    "ct scan",
    "mri",
    "pathology",
    "radiology",
    "histology",
    "genomics",
    "drug",
    "clinical",
    "arxiv",
    "ieee",
    "paper",
    "journal",
    "conference",
}

TECHNICAL_SIGNALS = {
    "how does",
    "how to",
    "implement",
    "build",
    "create",
    "deploy",
    "configure",
    "install",
    "library",
    "api",
    "code",
    "tutorial",
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
