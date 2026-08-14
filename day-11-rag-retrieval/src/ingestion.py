import os
from typing import List, Dict, Any

DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")

def load_and_chunk_documents(docs_dir: str = DOCS_DIR) -> List[Dict[str, Any]]:
    """
    Loads text documents from docs_dir and splits them into chunk objects.
    Each chunk contains:
    - chunk_id: Unique string identifier
    - doc_id: Source filename (e.g. hr_policy.txt)
    - text: Line/paragraph text content
    """
    chunks = []
    chunk_counter = 0

    if not os.path.exists(docs_dir):
        return chunks

    for fname in sorted(os.listdir(docs_dir)):
        if fname.endswith(".txt"):
            fpath = os.path.join(docs_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line_idx, line in enumerate(lines):
                # Ignore main header lines starting with '#'
                if line.startswith("# "):
                    continue
                chunk_counter += 1
                chunks.append({
                    "chunk_id": f"c_{chunk_counter}",
                    "doc_id": fname,
                    "line_index": line_idx,
                    "text": line
                })
    return chunks

if __name__ == "__main__":
    c = load_and_chunk_documents()
    print(f"Loaded {len(c)} chunks from documents directory.")
