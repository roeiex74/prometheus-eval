"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Metric
Formula: ROUGE-N = Σ match(n-gram)/Σ ref(n-gram); ROUGE-L via LCS; F=(1+β²)RP/(R+β²P)
Reference: Lin (2004). "ROUGE: Package for Automatic Evaluation of Summaries"
"""
from typing import List, Dict, Union, Tuple
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ROUGEMetric:
    """ROUGE metric with ROUGE-1, ROUGE-2, and ROUGE-L support."""

    def __init__(self, variants: List[str] = None, beta: float = 1.0):
        """Initialize ROUGE calculator (variants: rouge1/rouge2/rougeL, beta for F-measure)."""
        if variants is None:
            variants = ['rouge1', 'rouge2', 'rougeL']
        valid = {'rouge1', 'rouge2', 'rougeL'}
        for v in variants:
            if v not in valid:
                raise ValueError(f"Invalid variant '{v}'. Must be in {valid}")
        self.variants = variants
        self.beta_squared = beta ** 2

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize and lowercase text."""
        return word_tokenize(text.lower()) if text.strip() else []

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _compute_f1(self, recall: float, precision: float) -> float:
        """Compute F-measure with beta parameter."""
        if recall + precision == 0:
            return 0.0
        return ((1 + self.beta_squared) * recall * precision) / \
               (recall + self.beta_squared * precision)

    def _compute_rouge_n(
        self, cand_tokens: List[str], ref_tokens: List[str], n: int
    ) -> float:
        """Compute ROUGE-N F1 score."""
        cand_ngrams = self._get_ngrams(cand_tokens, n)
        ref_ngrams = self._get_ngrams(ref_tokens, n)

        if not ref_ngrams or not cand_ngrams:
            return 0.0

        overlap = set(cand_ngrams) & set(ref_ngrams)
        recall = len(overlap) / len(set(ref_ngrams))
        precision = len(overlap) / len(set(cand_ngrams))

        return self._compute_f1(recall, precision)

    def _lcs_length(self, seq1: List[str], seq2: List[str]) -> int:
        """Compute LCS length using space-optimized DP."""
        if not seq1 or not seq2:
            return 0

        m, n = len(seq1), len(seq2)
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, prev

        return prev[n]

    def _compute_rouge_l(self, cand_tokens: List[str], ref_tokens: List[str]) -> float:
        """Compute ROUGE-L F1 score."""
        if not ref_tokens or not cand_tokens:
            return 0.0

        lcs_len = self._lcs_length(cand_tokens, ref_tokens)
        recall = lcs_len / len(ref_tokens)
        precision = lcs_len / len(cand_tokens)

        return self._compute_f1(recall, precision)

    def compute(self, candidate: str, references: Union[str, List[str]], **kwargs) -> Dict[str, float]:
        """Compute ROUGE scores. Returns dict with rouge1/rouge2/rougeL/overall scores."""
        ref_list = [references] if isinstance(references, str) else references
        if not ref_list:
            raise ValueError("References cannot be empty")

        cand_tokens = self._tokenize(candidate)
        scores = {}

        for variant in self.variants:
            max_score = 0.0
            for ref in ref_list:
                ref_tokens = self._tokenize(ref)
                if variant == 'rouge1':
                    score = self._compute_rouge_n(cand_tokens, ref_tokens, 1)
                elif variant == 'rouge2':
                    score = self._compute_rouge_n(cand_tokens, ref_tokens, 2)
                elif variant == 'rougeL':
                    score = self._compute_rouge_l(cand_tokens, ref_tokens)
                else:
                    continue
                max_score = max(max_score, score)
            scores[variant] = max_score

        scores['overall'] = sum(scores.values()) / len(scores) if scores else 0.0
        return scores
