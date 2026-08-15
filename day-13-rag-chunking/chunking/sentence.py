import re
from typing import List, Dict, Any

class SentenceChunker:
    """
    Sentence-Based Chunking Engine splitting on natural sentence boundaries.
    """
    def __init__(self, target_sentences: int = 3):
        self.target_sentences = target_sentences

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]
        if not raw_sentences:
            return []

        chunks = []
        chunk_idx = 0
        for i in range(0, len(raw_sentences), self.target_sentences):
            group = raw_sentences[i:i + self.target_sentences]
            chunk_text = " ".join(group)
            chunk_idx += 1
            chunks.append({
                "chunk_id": f"{doc_id}_sent_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "token_count": len(chunk_text.split())
            })

        return chunks
