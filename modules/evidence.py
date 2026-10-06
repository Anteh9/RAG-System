"""Evidence sufficiency checks shared by the UI and API."""

from typing import Iterable
import re

from modules.models import RetrievedDocument

_SCOPE_TERMS = {
    "ghana", "budget", "fiscal", "financial", "allocation", "allocations",
    "allocate", "allocated", "spend", "spending", "expenditure", "revenue",
    "debt", "tax", "appropriation", "education", "school", "schools", "health",
    "election", "electoral", "vote", "votes", "voting", "won", "candidate",
    "party", "parties", "region", "regional", "constituency", "president",
    "presidential", "ashanti", "accra", "npp", "ndc",
}


def is_in_scope_query(query: str) -> bool:
    """Check whether a question concerns the indexed Ghana budget/election domain."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return bool(terms & _SCOPE_TERMS)


def has_sufficient_evidence(query: str, docs: Iterable[RetrievedDocument]) -> bool:
    """Require an in-scope query and at least one relevant, validly scored result."""
    if not is_in_scope_query(query):
        return False

    return any(
        0.42 <= doc.vector_score <= 1.0 or doc.keyword_score >= 0.15
        for doc in docs
    )