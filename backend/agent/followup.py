"""
Generates follow-up question suggestions after a report is written.
"""
import re
import json
from .llm import get_llm


def generate_follow_ups(question: str, report: str) -> list:
    try:
        llm = get_llm(temperature=0.4)
        response = llm.invoke([
            (
                "system",
                "You suggest follow-up research questions. "
                "Respond ONLY with a JSON array of exactly 3 short questions. "
                'Example: ["What is X?", "How does Y work?", "Why did Z happen?"]'
            ),
            (
                "human",
                f"Original question: {question}\n\nReport summary (first 500 chars): "
                f"{report[:500]}\n\nSuggest 3 follow-up questions."
            ),
        ])
        text = response.content.strip()
        # Try to extract JSON array
        match = re.search(r"\[[\s\S]*?\]", text)
        if match:
            questions = json.loads(match.group(0))
            return [q for q in questions if isinstance(q, str)][:3]
    except Exception as e:
        print(f"[followup] error: {e}")
    return [
        f"What are the challenges related to {question}?",
        f"What is the future outlook for {question}?",
        f"How does {question} compare to alternatives?",
    ]
