"""Conservative answer extraction for supported budget-table questions."""

import re
from typing import Iterable

from modules.models import RetrievedDocument


def extract_answer(query: str, docs: Iterable[RetrievedDocument]) -> str | None:
    """Extract the 2025 Ministry of Education total from its official summary row."""
    query_terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    asks_education_allocation = (
        "education" in query_terms
        and bool(query_terms & {"budget", "allocation", "allocated", "spending", "expenditure"})
        and "2025" in query_terms
    )
    if not asks_education_allocation:
        return None

    for doc in docs:
        text = doc.text
        if not all(term in text.lower() for term in ("appendix 4a", "2025", "social sector summary")):
            continue

        row = re.search(
            r"Ministry of Education\s+((?:[\d,]+\s+){5}[\d,]+)",
            text,
        )
        if row:
            amounts = re.findall(r"[\d,]+", row.group(1))
            return (
                f"The 2025 budget's Ministry of Education allocation totals "
                f"GH₵{amounts[-1]} across all funding sources."
            )

    return None