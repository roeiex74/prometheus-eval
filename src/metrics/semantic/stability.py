"""
Semantic Stability metric using sentence embeddings and pairwise cosine similarity.
Measures semantic consistency across multiple text outputs via mean pairwise similarity.
"""
from typing import List, Dict, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticStabilityMetric:
    """Measures semantic stability across multiple outputs using sentence embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', device: Optional[str] = None):
        """Initialize with sentence transformer model.

        Args:
            model_name: HuggingFace model identifier (default: all-MiniLM-L6-v2)
            device: Device to run on ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def compute(
        self,
        outputs: List[str],
        return_matrix: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Compute semantic stability across multiple text outputs.

        Stability Score = Mean pairwise cosine similarity
        Given N outputs [o₁, o₂, ..., oₙ]:
        1. Encode each output to sentence embedding: v₁, v₂, ..., vₙ
        2. Compute all pairwise cosine similarities: cos_sim(vᵢ, vⱼ) for i < j
        3. Stability = (2 / (N(N-1))) × Σ_{i<j} cos_sim(vᵢ, vⱼ)

        Args:
            outputs: List of N text outputs to compare (N ≥ 2)
            return_matrix: If True, include NxN similarity matrix in results
            **kwargs: Additional arguments (ignored for compatibility)

        Returns:
            Dictionary containing:
                - stability: Mean pairwise cosine similarity [0, 1]
                - min_similarity: Minimum pairwise similarity
                - max_similarity: Maximum pairwise similarity
                - std_similarity: Standard deviation of pairwise similarities
                - n_outputs: Number of outputs analyzed
                - model_name: Model used for embeddings
                - similarity_matrix: NxN similarity matrix (if return_matrix=True)

        Raises:
            TypeError: If outputs is not a list of strings
            ValueError: If fewer than 2 outputs provided
        """
        self._validate_inputs(outputs)

        # Encode all outputs to embeddings (batch processing for efficiency)
        embeddings = self.model.encode(outputs, convert_to_numpy=True)

        # Compute pairwise cosine similarity matrix
        similarity_matrix = self._compute_similarity_matrix(embeddings)

        # Extract upper triangle (pairwise comparisons without duplicates)
        pairwise_similarities = self._extract_pairwise_similarities(similarity_matrix)

        # Compute statistics
        result = {
            'stability': float(np.mean(pairwise_similarities)),
            'min_similarity': float(np.min(pairwise_similarities)),
            'max_similarity': float(np.max(pairwise_similarities)),
            'std_similarity': float(np.std(pairwise_similarities)),
            'n_outputs': len(outputs),
            'model_name': self.model_name
        }

        if return_matrix:
            result['similarity_matrix'] = similarity_matrix

        return result

    def _validate_inputs(self, outputs: List[str]) -> None:
        """Validate that outputs is a list of at least 2 strings.

        Args:
            outputs: Input to validate

        Raises:
            TypeError: If outputs is not a list or contains non-strings
            ValueError: If fewer than 2 outputs provided
        """
        if not isinstance(outputs, list):
            raise TypeError(f"outputs must be a list, got {type(outputs).__name__}")

        if len(outputs) < 2:
            raise ValueError(
                f"Need at least 2 outputs for stability computation, got {len(outputs)}"
            )

        if not all(isinstance(o, str) for o in outputs):
            non_string_types = [type(o).__name__ for o in outputs
                                if not isinstance(o, str)]
            raise TypeError(
                f"All outputs must be strings, found: {', '.join(set(non_string_types))}"
            )

    def _compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute NxN cosine similarity matrix from embeddings.

        Cosine similarity = dot product of normalized vectors.

        Args:
            embeddings: (N, D) array of sentence embeddings

        Returns:
            (N, N) symmetric similarity matrix where M[i,j] = cosine_sim(emb[i], emb[j])
        """
        # Normalize embeddings to unit vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / norms

        # Cosine similarity = dot product of normalized vectors
        return np.dot(normalized, normalized.T)

    def _extract_pairwise_similarities(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Extract unique pairwise similarities from upper triangle of matrix.

        Args:
            similarity_matrix: (N, N) symmetric similarity matrix

        Returns:
            1D array of N(N-1)/2 pairwise similarities
        """
        n = similarity_matrix.shape[0]
        pairwise_sims = []
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sims.append(similarity_matrix[i, j])
        return np.array(pairwise_sims)
