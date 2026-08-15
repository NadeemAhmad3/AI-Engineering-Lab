import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    """
    Semantic Chunking Engine grouping sentences into coherent chunks based on
    semantic similarity threshold boundaries.
    """
    def __init__(self, similarity_threshold: float = 0.35):
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
        if not raw_sentences:
            return []

        if len(raw_sentences) == 1:
            return [{
                "chunk_id": f"{doc_id}_sem_1",
                "doc_id": doc_id,
                "text": raw_sentences[0],
                "token_count": len(raw_sentences[0].split())
            }]

        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_mat = vectorizer.fit_transform(raw_sentences)
        sims = cosine_similarity(tfidf_mat)

        chunks = []
        current_group = [raw_sentences[0]]
        chunk_idx = 1

        for i in range(1, len(raw_sentences)):
            sim_score = sims[i - 1, i]
            if sim_score >= self.similarity_threshold:
                current_group.append(raw_sentences[i])
            else:
                chunk_text = " ".join(current_group)
                chunks.append({
                    "chunk_id": f"{doc_id}_sem_{chunk_idx}",
                    "doc_id": doc_id,
                    "text": chunk_text,
                    "token_count": len(chunk_text.split())
                })
                chunk_idx += 1
                current_group = [raw_sentences[i]]

        if current_group:
            chunk_text = " ".join(current_group)
            chunks.append({
                "chunk_id": f"{doc_id}_sem_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "token_count": len(chunk_text.split())
            })

        return chunks
