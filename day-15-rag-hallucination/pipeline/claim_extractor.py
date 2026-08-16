import re
from typing import List

class ClaimExtractor:
    """
    Decomposes generated RAG response strings into verifiable atomic claim statements.
    """
    def extract_claims(self, text: str) -> List[str]:
        if not text or "couldn't find sufficient information" in text.lower():
            return []

        # Split into sentence-level atomic claims
        raw_claims = [c.strip() for c in re.split(r'(?<=[.!?])\s+|\n+|;\s*', text) if c.strip()]
        
        claims = []
        for claim in raw_claims:
            # Clean up introductory boilerplate
            cleaned = re.sub(r'^(based on the context,|according to the document,|the document states that)', '', claim, flags=re.IGNORECASE).strip()
            if len(cleaned.split()) >= 3:
                claims.append(cleaned)
        return claims if claims else [text.strip()]
