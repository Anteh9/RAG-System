"""
Keyword Search
TF-IDF based keyword matching for hybrid search
"""

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordSearcher:
    """
    TF-IDF based keyword matching
    
    Purpose: Catch exact term matches that semantic search might miss
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = None
        self.corpus = []
    
    def build_index(self, texts: List[str]):
        """Build TF-IDF index"""
        self.corpus = texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[KEYWORD] Index built: {self.tfidf_matrix.shape}")
    
    def search(self, query: str, doc_indices: List[int] | None = None) -> Dict[int, float]:
        """Score selected documents, or the full corpus when no indices are given."""
        query_vec = self.vectorizer.transform([query])
        scores = (query_vec @ self.tfidf_matrix.T).toarray()[0]
        indices = range(len(self.corpus)) if doc_indices is None else doc_indices
        return {idx: float(scores[idx]) for idx in indices}
