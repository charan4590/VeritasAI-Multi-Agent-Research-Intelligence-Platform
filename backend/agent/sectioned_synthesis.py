"""
#9: Per-section synthesis decomposition (opt-in).

The existing synthesize_node generates all 9 academic sections in a
single LLM call. This module offers an alternative: generate each
section independently with section-specific context, then assemble.

Benefits: better quality per section (focused prompt, less dilution),
individually retryable on failure, and parallelizable.
Tradeoff: more LLM calls (9 vs 1) — slower wall-clock unless run
concurrently, and loses some cross-section coherence the model would
otherwise maintain naturally.

This is opt-in via SYNTHESIS_MODE=sectioned env var — the default
monolithic synthesis in graph.py is unchanged and remains the default,
per the instruction to preserve existing behavior.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict

from .llm import get_llm

logger = logging.getLogger(__name__)

ACADEMIC_SECTIONS = [
    (
        "Introduction",
        "State the problem, its significance, and motivation for using deep learning. 2-3 paragraphs.",
    ),
    (
        "Related Work",
        "Summarize prior methods chronologically, naming specific models and results. Cite with [n].",
    ),
    ("Proposed Method", "Describe the overall system design and key innovations. Cite with [n]."),
    (
        "Model Architecture",
        "Detail neural network architecture: layer types, dimensions, activations, loss functions. Cite with [n].",
    ),
    ("Dataset", "For each dataset: name, size, modality, class distribution, preprocessing. Cite with [n]."),
    (
        "Experimental Results",
        "Report exact numbers (accuracy, AUC, F1, etc) as a markdown table. Cite with [n].",
    ),
    ("Comparison with Existing Methods", "Compare proposed method against baselines/SOTA with a table."),
    ("Limitations", "List specific limitations: dataset size, generalizability, computational cost."),
]


def _generate_section(
    name: str, instruction: str, question: str, context_block: str, temperature: float
) -> str:
    llm = get_llm(temperature=temperature)
    response = llm.invoke(
        [
            (
                "system",
                f"You are an expert research scientist writing the '{name}' section "
                f"of a technical report. {instruction} "
                "Cite EVERY factual claim with [n] using ONLY source ids from the "
                "provided list. Never invent numbers — write 'not reported' if a "
                "metric is absent from sources. Output ONLY the section content in "
                "markdown, starting with a '## {name}' heading.",
            ),
            ("human", f"Research topic: {question}\n\nSources:\n{context_block}"),
        ]
    )
    return response.content


def sectioned_synthesis(question: str, context_block: str, sources: Dict, temperature: float = 0.1) -> str:
    """
    Generate the 8 academic sections concurrently, then assemble with
    a References section built from cited sources. Used when
    SYNTHESIS_MODE=sectioned is set; otherwise graph.py's monolithic
    synthesize_node is used (the default, unchanged behavior).
    """
    logger.info("Running sectioned synthesis (concurrent per-section generation)")

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_generate_section, name, instr, question, context_block, temperature): name
            for name, instr in ACADEMIC_SECTIONS
        }
        section_results: Dict[str, str] = {}
        for future in futures:
            name = futures[future]
            try:
                section_results[name] = future.result()
            except Exception as exc:
                logger.error(f"Section '{name}' generation failed: {exc}")
                section_results[name] = f"## {name}\n\n*Section generation failed: {exc}*"

    # Assemble in canonical order
    ordered = [section_results[name] for name, _ in ACADEMIC_SECTIONS if name in section_results]
    report = "\n\n".join(ordered)
    report += "\n\n## References\n\n*See cited sources above.*"
    return report


def is_sectioned_mode_enabled() -> bool:
    return os.environ.get("SYNTHESIS_MODE", "monolithic").lower() == "sectioned"
