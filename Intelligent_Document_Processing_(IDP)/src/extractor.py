import re
from schemas import QueryResponse, QueryIntent


class Extractor:
    """
    UPGRADE: Extractor now returns a typed Pydantic QueryResponse instead of raw strings.

    The rule logic is mostly unchanged — we just:
    1. Detect intent using the QueryIntent enum (clean, extensible)
    2. Wrap the result in a QueryResponse model (typed JSON output)
    3. Set confidence="high" for rule hits, return None for LLM fallback
    """

    def _detect_intent(self, query: str) -> QueryIntent:
        """Map query text to a QueryIntent enum value."""
        query = query.lower()

        if any(w in query for w in ["email", "mail"]):
            return QueryIntent.EMAIL
        if "name" in query:
            return QueryIntent.NAME
        if any(w in query for w in ["amount", "total", "price", "cost"]):
            return QueryIntent.AMOUNT
        if "date" in query:
            return QueryIntent.DATE

        return QueryIntent.GENERAL

    def extract(self, query: str, context: str) -> QueryResponse:
        """
        Attempt rule-based extraction.

        Returns a QueryResponse with:
        - answer set + confidence="high" → rule matched, use this
        - answer=None + confidence="low"  → no rule matched, fall back to LLM

        Args:
            query: user question
            context: relevant text chunks joined together

        Returns:
            QueryResponse (Pydantic model)
        """
        query_lower = query.lower()
        intent = self._detect_intent(query_lower)
        snippet = context[:300] if context else None

        # --- EMAIL ---
        if intent == QueryIntent.EMAIL:
            emails = re.findall(r'\S+@\S+\.\S+', context)
            if emails:
                if any(w in query_lower for w in ["support", "help", "contact", "problem"]):
                    answer = emails[0]
                elif "customer" in query_lower:
                    answer = emails[-1]
                else:
                    answer = emails[-1]

                return QueryResponse(
                    query=query,
                    intent=intent,
                    answer=answer,
                    confidence="high",
                    source="rule_based",
                    context_snippet=snippet
                )

        # --- NAME ---
        if intent == QueryIntent.NAME:
            match = re.search(r'Name:\s*([A-Za-z ]+?)(?=\s*Email|$)', context)
            if match:
                return QueryResponse(
                    query=query,
                    intent=intent,
                    answer=match.group(1).strip(),
                    confidence="high",
                    source="rule_based",
                    context_snippet=snippet
                )

        # --- AMOUNT ---
        if intent == QueryIntent.AMOUNT:
            match = re.search(r'Total Amount:\s*₹?\d+', context)
            if match:
                return QueryResponse(
                    query=query,
                    intent=intent,
                    answer=match.group(),
                    confidence="high",
                    source="rule_based",
                    context_snippet=snippet
                )

        # --- DATE ---
        if intent == QueryIntent.DATE:
            match = re.search(r'\d{1,2}\s[A-Za-z]+\s\d{4}', context)
            if match:
                return QueryResponse(
                    query=query,
                    intent=intent,
                    answer=match.group(),
                    confidence="high",
                    source="rule_based",
                    context_snippet=snippet
                )

        # --- No rule matched → signal LLM fallback ---
        return QueryResponse(
            query=query,
            intent=intent,
            answer=None,
            confidence="low",
            source="not_found",
            context_snippet=snippet
        )