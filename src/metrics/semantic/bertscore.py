"""
BERTScore Metric Implementation

This module implements BERTScore for semantic similarity evaluation using
contextual embeddings, as specified in PRD Section 3.2.1.

Mathematical Foundation:
    BERTScore uses token-level embeddings with greedy matching:

    R_BERT = (1/|x|) Σ max(x_i^T x̂_j)  [Recall]
    P_BERT = (1/|x̂|) Σ max(x_i^T x̂_j)  [Precision]
    F1_BERT = 2 × (R_BERT × P_BERT) / (R_BERT + P_BERT)

    Where:
    - x_i, x̂_j: Normalized embedding vectors for tokens
    - Greedy matching: Each token matched to most similar token in other text
    - Similarity computed via cosine similarity (dot product of normalized vectors)

References:
    Zhang et al. (2020). "BERTScore: Evaluating Text Generation with BERT"
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from loguru import logger
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import warnings

# Suppress tokenizer warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class BERTScoreMetric:
    """
    BERTScore metric using contextual embeddings for semantic similarity.

    This implementation uses sentence-transformers for efficient embedding
    generation and supports both single-pair and batch computation.

    Attributes:
        model_name (str): Name of the embedding model
        device (str): Device to run model on ("cpu", "cuda", "mps")
        rescale_with_baseline (bool): Whether to apply baseline rescaling
        model: Loaded sentence-transformer model
        max_length (int): Maximum sequence length for tokenization

    Example:
        >>> metric = BERTScoreMetric()
        >>> result = metric.compute(
        ...     hypothesis="The cat is on the mat",
        ...     reference="A cat sits on a mat"
        ... )
        >>> 0.8 < result['f1'] < 1.0  # High semantic similarity
        True
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
        rescale_with_baseline: bool = False,
        max_length: int = 512
    ):
        """
        Initialize BERTScore metric calculator.

        Args:
            model_name: HuggingFace model identifier for embeddings
                Default: "sentence-transformers/all-mpnet-base-v2"
                Alternatives: "bert-base-uncased", "roberta-base"
            device: Device to use ("cpu", "cuda", "mps"). Auto-detected if None.
            rescale_with_baseline: Apply baseline rescaling (optional feature)
            max_length: Maximum sequence length for tokenization

        Note:
            The default model (all-mpnet-base-v2) provides strong performance
            across diverse tasks. For domain-specific applications, consider
            fine-tuned models.
        """
        self.model_name = model_name
        self.rescale_with_baseline = rescale_with_baseline
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Initializing BERTScore with model: {model_name} on {self.device}")

        # Load model and tokenizer
        try:
            # Use sentence-transformers for efficiency
            self.model = SentenceTransformer(model_name, device=self.device)
            self.tokenizer = self.model.tokenizer
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

        # Baseline statistics (for rescaling, if enabled)
        self.baseline_mean = None
        self.baseline_std = None

    def _get_token_embeddings(
        self,
        text: str,
        return_tokens: bool = True
    ) -> Tuple[torch.Tensor, Optional[List[str]]]:
        """
        Generate token-level embeddings for input text.

        Args:
            text: Input text string
            return_tokens: Whether to return token strings

        Returns:
            Tuple of (embeddings_tensor, token_list)
            - embeddings_tensor: Shape [num_tokens, embedding_dim]
            - token_list: List of token strings (if return_tokens=True)
        """
        if not text.strip():
            logger.warning("Empty text provided, returning zero embeddings")
            empty_embedding = torch.zeros(1, self.model.get_sentence_embedding_dimension())
            return empty_embedding, [""] if return_tokens else None

        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # Get model outputs
        with torch.no_grad():
            # For sentence-transformers, we need to access the underlying transformer
            if hasattr(self.model, '_first_module'):
                # SentenceTransformer wraps the model
                transformer = self.model._first_module().auto_model
            else:
                # Fallback to direct model access
                transformer = AutoModel.from_pretrained(self.model_name).to(self.device)

            outputs = transformer(**inputs)
            # Use last hidden state for token embeddings
            token_embeddings = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden_dim]

        # Get actual tokens (excluding special tokens for alignment)
        tokens = None
        if return_tokens:
            token_ids = inputs['input_ids'][0].cpu().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)

            # Remove special tokens [CLS], [SEP], <s>, </s>, etc.
            special_tokens = set(self.tokenizer.all_special_tokens)
            valid_indices = [
                i for i, tok in enumerate(tokens)
                if tok not in special_tokens
            ]

            if valid_indices:
                token_embeddings = token_embeddings[valid_indices]
                tokens = [tokens[i] for i in valid_indices]
            else:
                # If all tokens are special, keep at least one
                token_embeddings = token_embeddings[:1]
                tokens = tokens[:1]

        # Normalize embeddings for cosine similarity (dot product = cosine sim)
        token_embeddings = torch.nn.functional.normalize(token_embeddings, p=2, dim=1)

        return token_embeddings, tokens

    def _compute_greedy_matching(
        self,
        embeddings_ref: torch.Tensor,
        embeddings_hyp: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Compute greedy matching between reference and hypothesis embeddings.

        For each token in reference, find the most similar token in hypothesis
        (and vice versa) using cosine similarity.

        Args:
            embeddings_ref: Reference token embeddings [num_ref_tokens, embed_dim]
            embeddings_hyp: Hypothesis token embeddings [num_hyp_tokens, embed_dim]

        Returns:
            Tuple of (recall, precision)
            - recall: Average max similarity from ref to hyp
            - precision: Average max similarity from hyp to ref
        """
        # Compute pairwise cosine similarities
        # Shape: [num_ref_tokens, num_hyp_tokens]
        sim_matrix = torch.matmul(embeddings_ref, embeddings_hyp.t())

        # Recall: For each ref token, find max similarity with any hyp token
        # Shape: [num_ref_tokens]
        ref_max_sim = sim_matrix.max(dim=1)[0]
        recall = ref_max_sim.mean().item()

        # Precision: For each hyp token, find max similarity with any ref token
        # Shape: [num_hyp_tokens]
        hyp_max_sim = sim_matrix.max(dim=0)[0]
        precision = hyp_max_sim.mean().item()

        return recall, precision

    def compute(
        self,
        hypothesis: str,
        reference: str
    ) -> Dict[str, float]:
        """
        Compute BERTScore between hypothesis and reference.

        Args:
            hypothesis: Candidate text to evaluate
            reference: Reference (ground truth) text

        Returns:
            Dictionary containing:
                - 'precision': BERTScore precision (P_BERT)
                - 'recall': BERTScore recall (R_BERT)
                - 'f1': BERTScore F1 score
                - 'num_hyp_tokens': Number of tokens in hypothesis
                - 'num_ref_tokens': Number of tokens in reference

        Example:
            >>> metric = BERTScoreMetric()
            >>> result = metric.compute(
            ...     hypothesis="The quick brown fox",
            ...     reference="A fast brown fox"
            ... )
            >>> 0 <= result['f1'] <= 1
            True
            >>> result['f1'] > 0.7  # High semantic overlap
            True
        """
        # Get token embeddings
        ref_embeddings, ref_tokens = self._get_token_embeddings(reference)
        hyp_embeddings, hyp_tokens = self._get_token_embeddings(hypothesis)

        logger.debug(
            f"Computing BERTScore: {len(ref_tokens)} ref tokens, "
            f"{len(hyp_tokens)} hyp tokens"
        )

        # Compute greedy matching
        recall, precision = self._compute_greedy_matching(
            ref_embeddings,
            hyp_embeddings
        )

        # Compute F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        # Apply baseline rescaling if enabled
        if self.rescale_with_baseline and self.baseline_mean is not None:
            precision = (precision - self.baseline_mean) / self.baseline_std
            recall = (recall - self.baseline_mean) / self.baseline_std
            f1 = (f1 - self.baseline_mean) / self.baseline_std

            # Clip to [0, 1] range after rescaling
            precision = max(0.0, min(1.0, precision))
            recall = max(0.0, min(1.0, recall))
            f1 = max(0.0, min(1.0, f1))

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_hyp_tokens': len(hyp_tokens),
            'num_ref_tokens': len(ref_tokens)
        }

    def compute_batch(
        self,
        hypotheses: List[str],
        references: List[str]
    ) -> Dict[str, List[float]]:
        """
        Compute BERTScore for multiple hypothesis-reference pairs.

        This method processes pairs independently (not corpus-level aggregation).
        For efficiency, it could be optimized with batched embedding generation,
        but current implementation prioritizes clarity.

        Args:
            hypotheses: List of candidate texts
            references: List of reference texts

        Returns:
            Dictionary containing lists of scores:
                - 'precision': List of precision scores
                - 'recall': List of recall scores
                - 'f1': List of F1 scores
                - 'mean_precision': Average precision
                - 'mean_recall': Average recall
                - 'mean_f1': Average F1

        Raises:
            ValueError: If hypotheses and references have different lengths

        Example:
            >>> metric = BERTScoreMetric()
            >>> results = metric.compute_batch(
            ...     hypotheses=["The cat sat", "A dog ran"],
            ...     references=["The cat is sitting", "A dog is running"]
            ... )
            >>> len(results['f1']) == 2
            True
        """
        if len(hypotheses) != len(references):
            raise ValueError(
                f"Length mismatch: {len(hypotheses)} hypotheses, "
                f"{len(references)} references"
            )

        if not hypotheses:
            raise ValueError("Cannot compute BERTScore on empty batch")

        logger.info(f"Computing BERTScore for batch of {len(hypotheses)} pairs")

        precisions = []
        recalls = []
        f1s = []

        for hyp, ref in zip(hypotheses, references):
            result = self.compute(hyp, ref)
            precisions.append(result['precision'])
            recalls.append(result['recall'])
            f1s.append(result['f1'])

        return {
            'precision': precisions,
            'recall': recalls,
            'f1': f1s,
            'mean_precision': np.mean(precisions),
            'mean_recall': np.mean(recalls),
            'mean_f1': np.mean(f1s)
        }

    def set_baseline(
        self,
        baseline_texts: List[str],
        num_samples: int = 1000
    ):
        """
        Compute baseline statistics for rescaling.

        Baseline rescaling normalizes scores based on expected similarity
        between random text pairs, improving score interpretability.

        Args:
            baseline_texts: Corpus of texts for baseline computation
            num_samples: Number of random pairs to sample

        Note:
            This is an optional feature. Most use cases do not require
            baseline rescaling.
        """
        if len(baseline_texts) < 2:
            raise ValueError("Need at least 2 texts for baseline computation")

        logger.info(f"Computing baseline from {num_samples} random pairs")

        scores = []
        for _ in range(num_samples):
            # Sample random pair
            idx1, idx2 = np.random.choice(len(baseline_texts), size=2, replace=False)
            result = self.compute(baseline_texts[idx1], baseline_texts[idx2])
            scores.append(result['f1'])

        self.baseline_mean = np.mean(scores)
        self.baseline_std = np.std(scores)

        logger.info(
            f"Baseline statistics: mean={self.baseline_mean:.4f}, "
            f"std={self.baseline_std:.4f}"
        )

    def __call__(self, hypothesis: str, reference: str) -> Dict[str, float]:
        """
        Convenience method to compute score.

        Args:
            hypothesis: Candidate text
            reference: Reference text

        Returns:
            Dictionary with BERTScore results
        """
        return self.compute(hypothesis, reference)


def compare_with_reference_implementation(
    hypothesis: str,
    reference: str
) -> Dict[str, Dict[str, float]]:
    """
    Compare custom implementation with bert_score library (if available).

    This function is useful for validation during development.

    Args:
        hypothesis: Candidate text
        reference: Reference text

    Returns:
        Dictionary with results from both implementations
    """
    results = {}

    # Custom implementation
    metric = BERTScoreMetric()
    results['custom'] = metric.compute(hypothesis, reference)

    # Reference implementation (if available)
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            [hypothesis],
            [reference],
            lang="en",
            verbose=False
        )

        results['reference'] = {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }

        logger.info("Comparison with reference bert_score library:")
        logger.info(f"  Custom F1:    {results['custom']['f1']:.4f}")
        logger.info(f"  Reference F1: {results['reference']['f1']:.4f}")
        logger.info(f"  Difference:   {abs(results['custom']['f1'] - results['reference']['f1']):.4f}")

    except ImportError:
        logger.warning("bert_score library not available for comparison")
        results['reference'] = None

    return results


if __name__ == "__main__":
    # Example usage
    logger.info("BERTScore Metric - Example Usage")

    metric = BERTScoreMetric()

    # Example 1: Identical sentences
    print("\n=== Example 1: Identical Sentences ===")
    result = metric.compute(
        hypothesis="The cat is on the mat",
        reference="The cat is on the mat"
    )
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")

    # Example 2: Paraphrases
    print("\n=== Example 2: Paraphrases ===")
    result = metric.compute(
        hypothesis="A cat sits on a mat",
        reference="The cat is on the mat"
    )
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")

    # Example 3: Unrelated sentences
    print("\n=== Example 3: Unrelated Sentences ===")
    result = metric.compute(
        hypothesis="The weather is sunny today",
        reference="The cat is on the mat"
    )
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall:    {result['recall']:.4f}")
    print(f"F1 Score:  {result['f1']:.4f}")

    # Example 4: Batch computation
    print("\n=== Example 4: Batch Computation ===")
    results = metric.compute_batch(
        hypotheses=[
            "The cat sat on the mat",
            "A dog is barking loudly"
        ],
        references=[
            "The cat is sitting on the mat",
            "A dog barks very loudly"
        ]
    )
    print(f"Mean F1 Score: {results['mean_f1']:.4f}")
    print(f"Individual F1 Scores: {[f'{f:.4f}' for f in results['f1']]}")

    # Example 5: Comparison with reference (if available)
    print("\n=== Example 5: Validation Against Reference ===")
    comparison = compare_with_reference_implementation(
        hypothesis="The quick brown fox jumps over the lazy dog",
        reference="A fast brown fox leaps over a lazy dog"
    )
