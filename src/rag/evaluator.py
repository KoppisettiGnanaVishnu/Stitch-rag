from typing import List, Dict, Any, Tuple
import re


class AnswerEvaluator:
    """
    Evaluates answer quality and confidence based on retrieved documents
    and generated answer content.
    """

    def evaluate_answer(
        self,
        query: str,
        context: str,
        answer: str,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the answer for confidence, groundedness, and completeness.

        Returns:
            Dict with keys:
            - 'confidence': float between 0 and 1 (higher = more confident)
            - 'groundedness': float between 0 and 1 (proportion of answer supported by context)
            - 'completeness': float between 0 and 1 (1.0 if answer fully addresses query)
            - 'verdict': str - "confident", "partial", or "unconfident"
            - 'issues': List[str] - list of detected issues
        """

        if not answer or answer.lower().startswith("the information is unavailable"):
            return self._create_evaluation(
                0.0, 0.0, 0.0, "unconfident",
                ["No answer provided or explicitly unavailable"]
            )

        issues = []

        # 1. Calculate retrieval confidence
        retrieval_confidence = self._calculate_retrieval_confidence(results)

        # 2. Calculate groundedness (proportion of answer supported by context)
        groundedness = self._calculate_groundedness(answer, context)

        # 3. Calculate completeness (address query coverage)
        completeness = self._calculate_completeness(answer, query, context)

        # 4. Determine overall confidence (weighted combination)
        overall_confidence = (
            0.5 * retrieval_confidence +
            0.3 * groundedness +
            0.2 * completeness
        )

        # 5. Determine verdict
        verdict = self._determine_verdict(overall_confidence, groundedness, issues)

        # 6. Check for potential hallucination or unsupported claims
        hallucination_issues = self._detect_potential_hallucinations(answer, context)
        if hallucination_issues:
            issues.extend(hallucination_issues)
            # Reduce confidence for hallucination risk
            overall_confidence *= 0.5
            if overall_confidence < 0.3:
                verdict = "unconfident"

        # 7. Check citation compliance
        citation_issues = self._check_citation_compliance(answer, results)
        if citation_issues:
            issues.extend(citation_issues)

        return self._create_evaluation(
            overall_confidence, groundedness, completeness, verdict, issues
        )

    def _calculate_retrieval_confidence(self, results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence based on the quality of retrieved passages.
        Higher confidence when the highest similarity score is high.
        """

        if not results:
            return 0.0

        highest_score = max(result.get("score", 0.0) for result in results)

        # Map similarity score to confidence (score range: 0-1)
        # Conservative: only confident when similarity > 0.7
        confidence = min(1.0, highest_score * 1.5)

        return max(0.0, min(1.0, confidence))

    def _calculate_groundedness(self, answer: str, context: str) -> float:
        """
        Calculate the proportion of answer content that is supported by context.
        This is a simple keyword overlap heuristic.
        """

        if not context:
            return 0.0

        # Normalize text for comparison
        answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
        context_words = set(re.findall(r"\b\w+\b", context.lower()))

        if not answer_words:
            return 0.0

        # Calculate word overlap
        overlap = answer_words.intersection(context_words)
        groundedness = len(overlap) / len(answer_words)

        return groundedness

    def _calculate_completeness(self, answer: str, query: str, context: str) -> float:
        """
        Calculate how well the answer addresses the query.
        """

        # Simple heuristics for completeness

        # 1. Check if answer contains key terms from query
        query_terms = set(re.findall(r"\b\w+\b", query.lower()))
        answer_terms = set(re.findall(r"\b\w+\b", answer.lower()))

        if not query_terms:
            return 0.5

        # Calculate term overlap
        relevant_terms = query_terms.intersection(answer_terms)
        term_overlap = len(relevant_terms) / len(query_terms)

        # 2. Check if answer is too short (potential incompleteness)
        answer_length_factor = min(1.0, len(answer.split()) / 50.0)

        # 3. Check if answer contradicts or ignores context
        contradiction_risk = self._detect_contradiction(answer, context)

        # Combine factors
        completeness = (
            0.5 * term_overlap +
            0.3 * answer_length_factor +
            0.2 * (1.0 - contradiction_risk)
        )

        return max(0.0, min(1.0, completeness))

    def _detect_potential_hallucinations(self, answer: str, context: str) -> List[str]:
        """
        Detect potential hallucinations by checking for specific red flags.
        This is a heuristic approach that catches obvious unsupported claims.
        """

        issues = []
        answer_lower = answer.lower()

        # Check for unsupported temporal references
        unsupported_temporal = self._check_unsupported_temporal(answer_lower, context)
        if unsupported_temporal:
            issues.append(unsupported_temporal)

        # Check for unsupported specific claims (numbers, dates, names)
        unsupported_specific = self._check_unsupported_specifics(answer_lower, context)
        if unsupported_specific:
            issues.append(unsupported_specific)

        # Check for generic filler that doesn't add value
        generic_fillers = [
            "additionally, ", "moreover, ", "furthermore, ",
            "in conclusion, ", "to summarize, ", "in summary, "
        ]
        filler_count = sum(1 for filler in generic_fillers if filler in answer_lower)
        if filler_count > 1:
            issues.append(f"Excessive generic filler ({filler_count} instances)")

        return issues

    def _check_unsupported_temporal(self, answer: str, context: str) -> str:
        """
        Check for unsupported temporal references in the answer.
        """

        # Look for years, dates, time periods in answer
        year_pattern = r'\b(19|20)\d{2}\b'
        date_patterns = [r'\b\d{1,2}/\d{1,2}/\d{4}\b', r'\b\w+ \d{1,2}, \d{4}\b']

        years_found = re.findall(year_pattern, answer)
        dates_found = []
        for pattern in date_patterns:
            dates_found.extend(re.findall(pattern, answer))

        if years_found or dates_found:
            # Check if these temporal references appear in context
            context_years = re.findall(year_pattern, context)
            context_dates = []
            for pattern in date_patterns:
                context_dates.extend(re.findall(pattern, context))

            if not years_found or not context_years:
                return "Unsupported temporal reference found"
            else:
                # Check if there's at least some overlap
                overlapping_years = set(years_found).intersection(set(context_years))
                if len(overlapping_years) / len(years_found) < 0.5:
                    return "Most temporal references unsupported by context"

        return ""

    def _check_unsupported_specifics(self, answer: str, context: str) -> str:
        """
        Check for unsupported specific claims like numbers, statistics, names.
        """

        # Look for percentage patterns
        percentage_pattern = r'\b\d+(?:\.\d+)?\s*%'
        percentages = re.findall(percentage_pattern, answer)

        # Look for number patterns that could be statistics
        number_pattern = r'\b\d+(?:\.\d+)?\s*(?:people|students|workers|cases|items|documents)\b'
        numbers = re.findall(number_pattern, answer.lower())

        if percentages or numbers:
            # Check context for supporting information
            if percentages:
                # Look for the same percentages in context
                context_percentages = re.findall(percentage_pattern, context)
                if not any(p in context for p in percentages):
                    return "Percentage information unsupported by context"

            if numbers:
                # Simple check for number presence in context
                context_lower = context.lower()
                for num in numbers:
                    if num not in context_lower:
                        return "Statistical information unsupported by context"

        return ""

    def _detect_contradiction(self, answer: str, context: str) -> float:
        """
        Detect potential contradiction between answer and context.
        Returns a value between 0 and 1 indicating contradiction risk.
        """

        # Simple contradiction detection based on negative keywords
        contradiction_words = {
            "not", "never", "no", "none", "absence", "lack", "without"
        }

        answer_lower = answer.lower()

        # Check if answer contains negative statements
        contradiction_count = sum(
            1 for word in contradiction_words if word in answer_lower
        )

        if contradiction_count == 0:
            return 0.0

        # Check if context has supporting positive information
        context_lower = context.lower()
        supporting_positive = any(
            word for word in contradiction_words if word in context_lower
        )

        if not supporting_positive:
            return 1.0

        return 0.5

    def _check_citation_compliance(self, answer: str, results: List[Dict[str, Any]]) -> List[str]:
        """
        Check if the answer properly cites sources as per Stitch rules.
        """

        issues = []

        # Check for citation format in answer
        citation_pattern = r'\[Source: [\w\s.-]+,?(?: Page: \d+)?\]'
        citations_in_answer = re.findall(citation_pattern, answer)

        if not citations_in_answer and results:
            issues.append("Answer contains information from documents but lacks source citations")

        # Check if citations reference actual sources
        if citations_in_answer:
            sources_in_results = {r["source"] for r in results}
            for citation in citations_in_answer:
                # Extract source name from citation
                # (stop at the comma or closing bracket so
                # punctuation is never part of the name)
                source_match = re.search(
                    r'Source:\s*([^,\]]+)', citation
                )
                if source_match:
                    source_name = source_match.group(1).strip()
                    if source_name not in sources_in_results:
                        issues.append(f"Citation references unsupported source: {source_name}")

        return issues

    def _determine_verdict(self, confidence: float, groundedness: float, issues: List[str]) -> str:
        """
        Determine the overall verdict based on confidence and groundedness.
        """

        if confidence < 0.3 or groundedness < 0.2:
            return "unconfident"

        if confidence >= 0.7 and groundedness >= 0.7:
            return "confident"

        return "partial"

    def _create_evaluation(
        self,
        confidence: float,
        groundedness: float,
        completeness: float,
        verdict: str,
        issues: List[str]
    ) -> Dict[str, Any]:
        """
        Create a standardized evaluation dictionary.
        """

        return {
            "confidence": round(confidence, 4),
            "groundedness": round(groundedness, 4),
            "completeness": round(completeness, 4),
            "verdict": verdict,
            "issues": issues
        }