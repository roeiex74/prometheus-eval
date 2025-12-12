"""
BLEU (Bilingual Evaluation Understudy) Metric Implementation

This module implements the BLEU metric for evaluating n-gram overlap between
candidate and reference texts, as specified in PRD Section 3.1.1.

Mathematical Foundation:
    BLEU = BP × exp(Σ(w_n × log p_n))

    Where:
    - p_n: modified n-gram precision
    - w_n: uniform weights (typically 1/4 for 4-grams)
    - BP: brevity penalty = min(1, exp(1 - r/c))
    - c: candidate length
    - r: reference length

References:
    Papineni et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation"
"""

import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Union
from loguru import logger
import nltk
from nltk.tokenize import word_tokenize

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer")
    nltk.download('punkt', quiet=True)


class BLEUMetric:
    """
    BLEU metric implementation with configurable n-gram order and smoothing.

    Attributes:
        max_n (int): Maximum n-gram order to consider (default: 4)
        smoothing (str): Smoothing method for zero n-gram matches
            Options: "none", "epsilon", "add-k"
        epsilon (float): Small constant for epsilon smoothing (default: 0.1)
        k (float): Constant for add-k smoothing (default: 1.0)

    Example:
        >>> metric = BLEUMetric(max_n=4, smoothing="epsilon")
        >>> result = metric.compute(
        ...     hypothesis="The cat is on the mat",
        ...     reference="The cat is sitting on the mat"
        ... )
        >>> print(result['bleu'])
        0.6687...
    """

    def __init__(
        self,
        max_n: int = 4,
        smoothing: str = "epsilon",
        epsilon: float = 0.1,
        k: float = 1.0
    ):
        """
        Initialize BLEU metric calculator.

        Args:
            max_n: Maximum n-gram order (default: 4 for BLEU-4)
            smoothing: Smoothing method ("none", "epsilon", "add-k")
            epsilon: Epsilon value for epsilon smoothing
            k: K value for add-k smoothing

        Raises:
            ValueError: If max_n < 1 or smoothing method is invalid
        """
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")

        valid_smoothing = {"none", "epsilon", "add-k"}
        if smoothing not in valid_smoothing:
            raise ValueError(f"smoothing must be one of {valid_smoothing}, got {smoothing}")

        self.max_n = max_n
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.k = k

        logger.debug(
            f"Initialized BLEUMetric: max_n={max_n}, smoothing={smoothing}, "
            f"epsilon={epsilon}, k={k}"
        )

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text using NLTK word tokenizer.

        Args:
            text: Input text string

        Returns:
            List of token strings
        """
        if not text.strip():
            return []
        return word_tokenize(text.lower())

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from token list.

        Args:
            tokens: List of tokens
            n: N-gram order

        Returns:
            List of n-gram tuples
        """
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _count_ngrams(self, ngrams: List[Tuple[str, ...]]) -> Counter:
        """
        Count occurrences of each n-gram.

        Args:
            ngrams: List of n-gram tuples

        Returns:
            Counter object with n-gram frequencies
        """
        return Counter(ngrams)

    def _compute_clipped_counts(
        self,
        candidate_ngrams: Counter,
        reference_ngrams: Counter
    ) -> int:
        """
        Compute clipped n-gram counts (for modified precision).

        The clipped count for each n-gram is the minimum of its count in the
        candidate and reference texts, preventing over-counting.

        Args:
            candidate_ngrams: N-gram counts from candidate text
            reference_ngrams: N-gram counts from reference text

        Returns:
            Total clipped count
        """
        clipped_count = 0
        for ngram, count in candidate_ngrams.items():
            clipped_count += min(count, reference_ngrams.get(ngram, 0))
        return clipped_count

    def _compute_precision(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str],
        n: int
    ) -> Tuple[float, int, int]:
        """
        Compute modified n-gram precision.

        Formula:
            p_n = Σ Count_clip(n-gram) / Σ Count(n-gram)

        Args:
            candidate_tokens: Tokenized candidate text
            reference_tokens: Tokenized reference text
            n: N-gram order

        Returns:
            Tuple of (precision, clipped_count, total_count)
        """
        candidate_ngrams = self._get_ngrams(candidate_tokens, n)
        reference_ngrams = self._get_ngrams(reference_tokens, n)

        if not candidate_ngrams:
            return 0.0, 0, 0

        candidate_counts = self._count_ngrams(candidate_ngrams)
        reference_counts = self._count_ngrams(reference_ngrams)

        clipped_count = self._compute_clipped_counts(candidate_counts, reference_counts)
        total_count = len(candidate_ngrams)

        # Apply smoothing for zero matches
        if clipped_count == 0:
            if self.smoothing == "epsilon":
                clipped_count = self.epsilon
            elif self.smoothing == "add-k":
                clipped_count = self.k

        precision = clipped_count / total_count if total_count > 0 else 0.0

        return precision, clipped_count, total_count

    def _compute_brevity_penalty(
        self,
        candidate_length: int,
        reference_length: int
    ) -> float:
        """
        Compute brevity penalty to penalize short candidates.

        Formula:
            BP = 1                      if c > r
            BP = exp(1 - r/c)          if c <= r

        Args:
            candidate_length: Length of candidate text
            reference_length: Length of reference text

        Returns:
            Brevity penalty value in (0, 1]
        """
        if candidate_length == 0:
            return 0.0

        if candidate_length > reference_length:
            return 1.0

        return math.exp(1 - reference_length / candidate_length)

    def compute(
        self,
        hypothesis: str,
        reference: Union[str, List[str]]
    ) -> Dict[str, float]:
        """
        Compute BLEU score between hypothesis and reference(s).

        Args:
            hypothesis: Candidate text to evaluate
            reference: Reference text or list of reference texts

        Returns:
            Dictionary containing:
                - 'bleu': Overall BLEU score
                - 'precisions': List of n-gram precisions [p1, p2, ..., p_N]
                - 'bp': Brevity penalty
                - 'length_ratio': Ratio of candidate to reference length
                - 'candidate_length': Number of tokens in candidate
                - 'reference_length': Number of tokens in reference

        Example:
            >>> metric = BLEUMetric()
            >>> result = metric.compute(
            ...     hypothesis="The cat sat on the mat",
            ...     reference="The cat is on the mat"
            ... )
            >>> result.keys()
            dict_keys(['bleu', 'precisions', 'bp', 'length_ratio',
                      'candidate_length', 'reference_length'])
        """
        # Handle multiple references
        if isinstance(reference, list):
            if not reference:
                raise ValueError("Reference list cannot be empty")
            # For multiple references, use the one closest in length
            # and track max clipped counts across all references
            hypothesis_tokens = self._tokenize(hypothesis)
            reference_tokens_list = [self._tokenize(ref) for ref in reference]

            # Choose reference closest in length for brevity penalty
            closest_ref_tokens = min(
                reference_tokens_list,
                key=lambda ref: abs(len(ref) - len(hypothesis_tokens))
            )

            return self._compute_bleu_multi_ref(
                hypothesis_tokens,
                reference_tokens_list,
                closest_ref_tokens
            )
        else:
            hypothesis_tokens = self._tokenize(hypothesis)
            reference_tokens = self._tokenize(reference)

            return self._compute_bleu_single_ref(
                hypothesis_tokens,
                reference_tokens
            )

    def _compute_bleu_single_ref(
        self,
        candidate_tokens: List[str],
        reference_tokens: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU for single reference.

        Args:
            candidate_tokens: Tokenized candidate
            reference_tokens: Tokenized reference

        Returns:
            Dictionary with BLEU scores and metadata
        """
        candidate_length = len(candidate_tokens)
        reference_length = len(reference_tokens)

        if candidate_length == 0:
            logger.warning("Empty candidate text, returning BLEU = 0.0")
            return {
                'bleu': 0.0,
                'precisions': [0.0] * self.max_n,
                'bp': 0.0,
                'length_ratio': 0.0,
                'candidate_length': 0,
                'reference_length': reference_length
            }

        # Compute n-gram precisions
        precisions = []
        for n in range(1, self.max_n + 1):
            precision, _, _ = self._compute_precision(
                candidate_tokens,
                reference_tokens,
                n
            )
            precisions.append(precision)

        # Compute geometric mean of precisions
        # BLEU = BP × exp(Σ w_n log p_n)
        # With uniform weights w_n = 1/N

        # Filter out zero precisions to avoid log(0)
        non_zero_precisions = [p for p in precisions if p > 0]

        if not non_zero_precisions:
            bleu_score = 0.0
        else:
            log_precisions = [math.log(p) for p in non_zero_precisions]
            geo_mean = math.exp(sum(log_precisions) / len(log_precisions))

            # Compute brevity penalty
            bp = self._compute_brevity_penalty(candidate_length, reference_length)

            bleu_score = bp * geo_mean

        # Recompute BP for return value
        bp = self._compute_brevity_penalty(candidate_length, reference_length)
        length_ratio = candidate_length / reference_length if reference_length > 0 else 0.0

        return {
            'bleu': bleu_score,
            'precisions': precisions,
            'bp': bp,
            'length_ratio': length_ratio,
            'candidate_length': candidate_length,
            'reference_length': reference_length
        }

    def _compute_bleu_multi_ref(
        self,
        candidate_tokens: List[str],
        reference_tokens_list: List[List[str]],
        closest_ref_tokens: List[str]
    ) -> Dict[str, float]:
        """
        Compute BLEU for multiple references.

        Uses maximum clipped counts across all references.

        Args:
            candidate_tokens: Tokenized candidate
            reference_tokens_list: List of tokenized references
            closest_ref_tokens: Reference closest in length to candidate

        Returns:
            Dictionary with BLEU scores and metadata
        """
        candidate_length = len(candidate_tokens)
        reference_length = len(closest_ref_tokens)

        if candidate_length == 0:
            return {
                'bleu': 0.0,
                'precisions': [0.0] * self.max_n,
                'bp': 0.0,
                'length_ratio': 0.0,
                'candidate_length': 0,
                'reference_length': reference_length
            }

        # Compute max clipped counts across all references
        precisions = []
        for n in range(1, self.max_n + 1):
            max_clipped = 0
            total_count = 0

            for ref_tokens in reference_tokens_list:
                _, clipped, total = self._compute_precision(
                    candidate_tokens,
                    ref_tokens,
                    n
                )
                max_clipped = max(max_clipped, clipped)
                total_count = total  # Same for all references

            precision = max_clipped / total_count if total_count > 0 else 0.0
            precisions.append(precision)

        # Compute BLEU score
        non_zero_precisions = [p for p in precisions if p > 0]

        if not non_zero_precisions:
            bleu_score = 0.0
        else:
            log_precisions = [math.log(p) for p in non_zero_precisions]
            geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
            bp = self._compute_brevity_penalty(candidate_length, reference_length)
            bleu_score = bp * geo_mean

        bp = self._compute_brevity_penalty(candidate_length, reference_length)
        length_ratio = candidate_length / reference_length if reference_length > 0 else 0.0

        return {
            'bleu': bleu_score,
            'precisions': precisions,
            'bp': bp,
            'length_ratio': length_ratio,
            'candidate_length': candidate_length,
            'reference_length': reference_length
        }

    def compute_corpus(
        self,
        hypotheses: List[str],
        references: Union[List[str], List[List[str]]]
    ) -> Dict[str, float]:
        """
        Compute corpus-level BLEU score.

        Aggregates statistics across all sentence pairs before computing BLEU,
        which is the standard corpus-level BLEU calculation.

        Args:
            hypotheses: List of candidate texts
            references: List of reference texts or list of reference lists

        Returns:
            Dictionary with corpus-level BLEU score and statistics

        Raises:
            ValueError: If hypotheses and references have different lengths

        Example:
            >>> metric = BLEUMetric()
            >>> result = metric.compute_corpus(
            ...     hypotheses=["The cat sat", "A dog barked"],
            ...     references=["The cat is sitting", "A dog is barking"]
            ... )
            >>> 0 <= result['bleu'] <= 1
            True
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Length mismatch: {len(hypotheses)} hypotheses, "
                f"{len(references)} references"
            )

        if not hypotheses:
            raise ValueError("Cannot compute BLEU on empty corpus")

        # Aggregate corpus-level statistics
        total_candidate_length = 0
        total_reference_length = 0

        # Track clipped counts and totals for each n-gram order
        corpus_clipped = [0] * self.max_n
        corpus_total = [0] * self.max_n

        for hyp, ref in zip(hypotheses, references):
            hyp_tokens = self._tokenize(hyp)

            # Handle multiple references per hypothesis
            if isinstance(ref, list):
                ref_tokens_list = [self._tokenize(r) for r in ref]
                # Use closest reference for length calculation
                ref_tokens = min(
                    ref_tokens_list,
                    key=lambda r: abs(len(r) - len(hyp_tokens))
                )
            else:
                ref_tokens = self._tokenize(ref)
                ref_tokens_list = [ref_tokens]

            total_candidate_length += len(hyp_tokens)
            total_reference_length += len(ref_tokens)

            # Accumulate n-gram statistics
            for n in range(1, self.max_n + 1):
                max_clipped = 0
                total_count = 0

                for ref_tok in ref_tokens_list:
                    _, clipped, total = self._compute_precision(
                        hyp_tokens,
                        ref_tok,
                        n
                    )
                    max_clipped = max(max_clipped, clipped)
                    total_count = total

                corpus_clipped[n-1] += max_clipped
                corpus_total[n-1] += total_count

        # Compute corpus-level precisions
        precisions = []
        for n in range(self.max_n):
            if corpus_total[n] > 0:
                precision = corpus_clipped[n] / corpus_total[n]
            else:
                precision = 0.0
            precisions.append(precision)

        # Compute corpus-level BLEU
        non_zero_precisions = [p for p in precisions if p > 0]

        if not non_zero_precisions:
            bleu_score = 0.0
        else:
            log_precisions = [math.log(p) for p in non_zero_precisions]
            geo_mean = math.exp(sum(log_precisions) / len(log_precisions))
            bp = self._compute_brevity_penalty(
                total_candidate_length,
                total_reference_length
            )
            bleu_score = bp * geo_mean

        bp = self._compute_brevity_penalty(
            total_candidate_length,
            total_reference_length
        )
        length_ratio = (
            total_candidate_length / total_reference_length
            if total_reference_length > 0 else 0.0
        )

        logger.info(
            f"Corpus BLEU: {bleu_score:.4f} "
            f"(BP={bp:.4f}, length_ratio={length_ratio:.4f})"
        )

        return {
            'bleu': bleu_score,
            'precisions': precisions,
            'bp': bp,
            'length_ratio': length_ratio,
            'candidate_length': total_candidate_length,
            'reference_length': total_reference_length,
            'num_sentences': len(hypotheses)
        }


if __name__ == "__main__":
    # Example usage
    metric = BLEUMetric(max_n=4, smoothing="epsilon")

    # Single reference
    result = metric.compute(
        hypothesis="The cat is on the mat",
        reference="The cat is sitting on the mat"
    )
    print(f"BLEU Score: {result['bleu']:.4f}")
    print(f"Precisions: {result['precisions']}")
    print(f"Brevity Penalty: {result['bp']:.4f}")

    # Multiple references
    result_multi = metric.compute(
        hypothesis="The cat is on the mat",
        reference=[
            "The cat is sitting on the mat",
            "A cat is on the mat"
        ]
    )
    print(f"\nMulti-reference BLEU: {result_multi['bleu']:.4f}")

    # Corpus-level
    result_corpus = metric.compute_corpus(
        hypotheses=["The cat sat", "A dog barked"],
        references=["The cat is sitting", "A dog is barking"]
    )
    print(f"\nCorpus BLEU: {result_corpus['bleu']:.4f}")
