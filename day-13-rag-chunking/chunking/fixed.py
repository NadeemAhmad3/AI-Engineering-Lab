from typing import List, Dict, Any

class FixedSizeChunker:
    """
    Fixed-Size Chunking Engine with configurable chunk size and token overlap.
    """
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        words = text.split()
        if not words:
            return []

        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunk_idx = 0

        for i in range(0, len(words), step):
            sub_words = words[i:i + self.chunk_size]
            if not sub_words:
                break
            chunk_text = " ".join(sub_words)
            chunk_idx += 1
            chunks.append({
                "chunk_id": f"{doc_id}_fixed_{chunk_idx}",
                "doc_id": doc_id,
                "text": chunk_text,
                "token_count": len(sub_words)
            })

            if i + self.chunk_size >= len(words):
                break

        return chunks
