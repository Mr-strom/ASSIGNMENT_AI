"""
FAQ Retrieval module using TF-IDF and keyword-weighted similarity matching.
Extracts relevant context from faq_data.json and flags unreliable/out-of-scope queries.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_assistant.config import DATA_PATH, SIMILARITY_THRESHOLD, TOP_K_FAQS

# Stopwords and common noise words that should never receive a domain keyword boost
NOISE_TOKENS = {
    "what", "when", "where", "which", "who", "whom", "this", "that", "there",
    "have", "from", "with", "about", "into", "through", "during", "before",
    "after", "above", "below", "under", "again", "further", "then", "once",
    "here", "hall", "room", "make", "take", "just", "like", "will", "would",
    "could", "should", "does", "done", "doing", "rule", "rules", "item", "items"
}


class FAQItem:
    def __init__(self, id: str, category: str, question: str, answer: str, keywords: List[str]):
        self.id = id
        self.category = category
        self.question = question
        self.answer = answer
        self.keywords = keywords

    def full_text(self) -> str:
        """Combine question, answer, and keywords for rich textual indexing."""
        kw_str = " ".join(self.keywords)
        return f"{self.question} {self.category} {kw_str} {self.answer}"


class FAQRetriever:
    """
    Intelligent FAQ Retriever combining TF-IDF cosine similarity with domain-keyword boosting.
    """

    def __init__(self, data_path: Path = DATA_PATH, threshold: float = SIMILARITY_THRESHOLD):
        self.data_path = Path(data_path)
        self.threshold = threshold
        self.faqs: List[FAQItem] = []
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words="english"
        )
        self.tfidf_matrix = None
        self.load_faqs()

    def load_faqs(self):
        """Load FAQs from JSON file and fit the TF-IDF index."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"FAQ data file not found at {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_faqs = data.get("faqs", [])
        self.faqs = [
            FAQItem(
                id=item["id"],
                category=item.get("category", "General"),
                question=item["question"],
                answer=item["answer"],
                keywords=item.get("keywords", [])
            )
            for item in raw_faqs
        ]

        if self.faqs:
            documents = [faq.full_text() for faq in self.faqs]
            self.tfidf_matrix = self.vectorizer.fit_transform(documents)

    def _keyword_boost(self, query: str, faq: FAQItem) -> float:
        """
        Calculate an additive boost if query explicitly mentions FAQ domain keywords.
        Filters out common/noise tokens to prevent false positives.
        """
        query_tokens = set(re.findall(r"\b\w{3,}\b", query.lower())) - NOISE_TOKENS

        # Tokenize FAQ keywords into individual non-noise tokens
        kw_tokens = set()
        for k in faq.keywords:
            tokens = re.findall(r"\b\w{3,}\b", k.lower())
            for t in tokens:
                if t not in NOISE_TOKENS:
                    kw_tokens.add(t)

        overlap = query_tokens.intersection(kw_tokens)
        # 0.08 per matched domain keyword, up to max 0.24
        return min(len(overlap) * 0.08, 0.24)

    def retrieve(self, query: str, top_k: int = TOP_K_FAQS) -> Tuple[List[Dict[str, Any]], bool, float]:
        """
        Retrieve top matching FAQs for a student query.
        Returns:
            - matches: List of dicts with faq details and composite similarity score
            - is_reliable: bool indicating whether top match meets confidence threshold
            - max_score: float representing the highest match score
        """
        if not self.faqs or self.tfidf_matrix is None:
            return [], False, 0.0

        query_vec = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        results = []
        for idx, faq in enumerate(self.faqs):
            cos_score = float(cosine_scores[idx])
            kw_boost = self._keyword_boost(query, faq)
            composite_score = cos_score + kw_boost

            results.append({
                "faq": faq,
                "score": composite_score,
                "cosine_score": cos_score,
                "keyword_boost": kw_boost
            })

        # Sort by composite score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]

        max_score = top_results[0]["score"] if top_results else 0.0
        is_reliable = max_score >= self.threshold

        formatted_matches = [
            {
                "id": r["faq"].id,
                "category": r["faq"].category,
                "question": r["faq"].question,
                "answer": r["faq"].answer,
                "score": round(r["score"], 4)
            }
            for r in top_results
        ]

        return formatted_matches, is_reliable, max_score

    def format_context(self, matches: List[Dict[str, Any]], is_reliable: bool) -> str:
        """Format retrieved FAQs into a clean prompt context block."""
        if not is_reliable or not matches:
            return "NO DIRECT OFFICIAL FAQ POLICIES FOUND FOR THIS INQUIRY."

        # Filter out noisy low-similarity secondary matches
        min_score = self.threshold * 0.6
        valid_matches = [m for m in matches if m.get("score", 0.0) >= min_score]
        if not valid_matches and matches:
            valid_matches = [matches[0]]

        lines = ["--- OFFICIAL EVERGREEN UNIVERSITY FAQ POLICY DOCUMENTS ---"]
        for idx, m in enumerate(valid_matches, 1):
            lines.append(f"[{idx}] Category: {m['category']} | Question: {m['question']}")
            lines.append(f"Policy / Details: {m['answer']}\n")
        lines.append("-----------------------------------------------------------")
        return "\n".join(lines)


# Shared retriever instance
default_retriever = FAQRetriever()
