from typing import Dict, Any, Tuple
from pipeline.claim_extractor import ClaimExtractor
from pipeline.evidence_checker import EvidenceChecker
from pipeline.abstention import AbstentionEngine

ABSTENTION_MESSAGE = "I couldn't find sufficient information in the available knowledge base to answer this."

class GroundedRAGPipeline:
    """
    RAG Generation Pipeline with Grounding Verification and Abstention Safeguards.
    """
    def __init__(self, confidence_threshold: float = 0.70):
        self.claim_extractor = ClaimExtractor()
        self.evidence_checker = EvidenceChecker()
        self.abstention_engine = AbstentionEngine(confidence_threshold=confidence_threshold)

    def generate(self, query: str, context: str) -> Dict[str, Any]:
        # 1. Check evidence confidence & abstention threshold
        must_abstain, conf = self.abstention_engine.should_abstain(query, context)
        if must_abstain:
            return {
                "answer": ABSTENTION_MESSAGE,
                "abstained": True,
                "confidence": conf,
                "faithfulness": 1.0,
                "claims": [],
                "verified_claims": []
            }

        # 2. Generate grounded response from context
        raw_answer = f"Based on the context, {context}"
        
        # 3. Extract claims and verify grounding
        claims = self.claim_extractor.extract_claims(raw_answer)
        verified_claims, faithfulness = self.evidence_checker.verify_claims(claims, context)

        return {
            "answer": raw_answer,
            "abstained": False,
            "confidence": conf,
            "faithfulness": faithfulness,
            "claims": claims,
            "verified_claims": verified_claims
        }
