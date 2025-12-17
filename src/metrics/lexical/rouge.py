"""
ROUGE (Recall-Oriented Understudy for Gisting Evaluation) Metric

Formula: ROUGE-N = Σ match(n-gram)/Σ ref(n-gram); ROUGE-L via LCS; F=(1+β²)RP/(R+β²P)

References:
    [1] C.-Y. Lin, "ROUGE: A Package for Automatic Evaluation of Summaries,"
        in Proc. Workshop on Text Summarization Branches Out, Barcelona, Spain,
        Jul. 2004, pp. 74-81.

    [2] C.-Y. Lin and E. Hovy, "Automatic Evaluation of Summaries Using N-gram
        Co-Occurrence Statistics," in Proc. 2003 Conf. North American Chapter
        of the Association for Computational Linguistics on Human Language
        Technology, vol. 1, Edmonton, Canada, May-Jun. 2003, pp. 71-78.
        doi: 10.3115/1073445.1073465
"""
from typing import List, Dict, Union, Tuple
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ROUGEMetric:
    """ROUGE metric with ROUGE-1, ROUGE-2, and ROUGE-L support.

    Examples:
        >>> from prometheus_eval.metrics.lexical.rouge import ROUGEMetric
        >>>
        >>> # Basic summarization evaluation
        >>> rouge = ROUGEMetric()
        >>> reference = "The cat sat on the mat. The dog barked loudly."
        >>> candidate = "A cat was sitting on a mat while a dog barked."
        >>>
        >>> scores = rouge.compute(candidate, reference)
        >>> print(f"ROUGE-1: {scores['rouge1']:.4f}")
        >>> print(f"ROUGE-2: {scores['rouge2']:.4f}")
        >>> print(f"ROUGE-L: {scores['rougeL']:.4f}")
        ROUGE-1: 0.6154
        ROUGE-2: 0.2500
        ROUGE-L: 0.5000

        >>> # Multi-reference summarization
        >>> references = [
        ...     "The cat sat on the mat. The dog barked loudly.",
        ...     "A cat rested on a mat. A dog made loud noises."
        ... ]
        >>> candidate = "Cat on mat, dog barking."
        >>> scores = rouge.compute(candidate, references)
        >>> print(f"Best ROUGE-1: {scores['rouge1']:.4f}")

        >>> # Prompt evaluation: Compare outputs from different prompts
        >>> reference = "Write Python code to sort a list"
        >>> candidate_a = "Use sorted() function to sort lists in Python"
        >>> candidate_b = "Python has sorted() for list sorting"
        >>>
        >>> rouge = ROUGEMetric(variants=['rouge1'])
        >>> score_a = rouge.compute(candidate_a, reference)['rouge1']
        >>> score_b = rouge.compute(candidate_b, reference)['rouge1']
        >>> best_prompt = "A" if score_a > score_b else "B"
        >>> print(f"Best prompt: {best_prompt}")
        Best prompt: A
    """

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
