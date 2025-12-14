"""
Comprehensive test suite for Semantic Stability metric.
Tests sentence embedding-based stability measurement across multiple outputs.
"""
import pytest
import numpy as np
from src.metrics.semantic.stability import SemanticStabilityMetric


EPSILON = 1e-4


class TestSemanticStabilityBasic:
    """Test basic semantic stability functionality."""

    def test_identical_outputs(self):
        """Identical outputs should have stability ≈ 1.0."""
        metric = SemanticStabilityMetric()
        outputs = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog"
        ]
        result = metric.compute(outputs)

        assert abs(result['stability'] - 1.0) < EPSILON, \
            "Identical texts should have stability ~1.0"
        assert abs(result['min_similarity'] - 1.0) < EPSILON
        assert abs(result['max_similarity'] - 1.0) < EPSILON
        assert result['std_similarity'] < EPSILON, "Std should be ~0 for identical texts"
        assert result['n_outputs'] == 3
        assert result['model_name'] == 'all-MiniLM-L6-v2'

    def test_different_outputs(self):
        """Unrelated topics should have low stability."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Python is a programming language used for web development and data science.",
            "The Eiffel Tower is a famous landmark located in Paris, France.",
            "Quantum mechanics describes the behavior of matter at atomic scales."
        ]
        result = metric.compute(outputs)

        assert result['stability'] < 0.5, "Unrelated topics should have low stability"
        assert 0.0 <= result['stability'] <= 1.0, "Stability must be in [0, 1]"
        assert result['min_similarity'] >= -1.0
        assert result['max_similarity'] <= 1.0
        assert result['n_outputs'] == 3

    def test_paraphrases(self):
        """Paraphrases should have high stability."""
        metric = SemanticStabilityMetric()
        outputs = [
            "The cat sat on the mat.",
            "A feline was seated on the rug.",
            "The small cat rested upon the floor covering."
        ]
        result = metric.compute(outputs)

        assert result['stability'] > 0.5, "Paraphrases should have moderate-high stability"
        assert result['stability'] <= 1.0
        assert result['n_outputs'] == 3

    def test_two_outputs_minimum(self):
        """Exactly 2 outputs should compute valid stability."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Machine learning is a subset of artificial intelligence.",
            "Artificial intelligence includes machine learning as a subfield."
        ]
        result = metric.compute(outputs)

        assert 0.0 <= result['stability'] <= 1.0
        assert result['n_outputs'] == 2
        assert result['stability'] > 0.7, "Similar sentences should have high similarity"

    def test_return_similarity_matrix(self):
        """Verify similarity matrix is returned when requested."""
        metric = SemanticStabilityMetric()
        outputs = ["Hello world", "Hi there", "Greetings"]
        result = metric.compute(outputs, return_matrix=True)

        assert 'similarity_matrix' in result
        matrix = result['similarity_matrix']
        assert matrix.shape == (3, 3), "Matrix should be NxN"
        assert np.allclose(matrix, matrix.T), "Matrix should be symmetric"
        assert np.allclose(np.diag(matrix), 1.0, atol=EPSILON), "Diagonal should be 1.0"

    def test_no_similarity_matrix_by_default(self):
        """Similarity matrix should not be included by default."""
        metric = SemanticStabilityMetric()
        outputs = ["Test one", "Test two"]
        result = metric.compute(outputs)

        assert 'similarity_matrix' not in result


class TestSemanticStabilityEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_strings(self):
        """Empty strings should be handled gracefully."""
        metric = SemanticStabilityMetric()
        outputs = ["", ""]
        result = metric.compute(outputs)

        assert 0.0 <= result['stability'] <= 1.0
        assert result['n_outputs'] == 2

    def test_mixed_empty_and_content(self):
        """Mix of empty and non-empty strings."""
        metric = SemanticStabilityMetric()
        outputs = ["Hello world", "", "Goodbye"]
        result = metric.compute(outputs)

        assert 0.0 <= result['stability'] <= 1.0
        assert result['n_outputs'] == 3

    def test_very_long_text(self):
        """Handle long texts (1000+ words)."""
        metric = SemanticStabilityMetric()
        long_text = " ".join(["word"] * 1000)
        outputs = [long_text, long_text + " extra", long_text]
        result = metric.compute(outputs)

        assert result['stability'] > 0.95, "Near-identical long texts should have high stability"
        assert result['n_outputs'] == 3

    def test_mixed_lengths(self):
        """Mix of short and long texts."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Hi",
            "This is a moderately long sentence with several words in it.",
            "Short"
        ]
        result = metric.compute(outputs)

        assert 0.0 <= result['stability'] <= 1.0
        assert result['n_outputs'] == 3

    def test_non_english_text(self):
        """Model supports 100+ languages."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Bonjour le monde",  # French
            "Hola mundo",  # Spanish
            "Hello world"  # English
        ]
        result = metric.compute(outputs)

        assert result['stability'] > 0.2, \
            "Similar greetings in different languages should show some similarity"
        assert result['n_outputs'] == 3

    def test_special_characters(self):
        """Handle special characters and punctuation."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Hello! How are you?",
            "Hello!!! How are you???",
            "Hello. How are you."
        ]
        result = metric.compute(outputs)

        assert result['stability'] > 0.85, "Same content with different punctuation"
        assert result['n_outputs'] == 3


class TestSemanticStabilityValidation:
    """Test input validation and error handling."""

    def test_single_output_raises_error(self):
        """Single output should raise ValueError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(ValueError, match="at least 2 outputs"):
            metric.compute(["Only one output"])

    def test_empty_list_raises_error(self):
        """Empty list should raise ValueError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(ValueError, match="at least 2 outputs"):
            metric.compute([])

    def test_non_list_input_raises_error(self):
        """String input should raise TypeError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(TypeError, match="must be a list"):
            metric.compute("Not a list")

    def test_tuple_input_raises_error(self):
        """Tuple input should raise TypeError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(TypeError, match="must be a list"):
            metric.compute(("Tuple", "of", "strings"))

    def test_non_string_elements_raises_error(self):
        """List with non-string elements should raise TypeError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(TypeError, match="must be strings"):
            metric.compute([123, 456])

    def test_mixed_types_in_list_raises_error(self):
        """List with mixed types should raise TypeError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(TypeError, match="must be strings"):
            metric.compute(["Valid string", 123, None])

    def test_none_in_list_raises_error(self):
        """List containing None should raise TypeError."""
        metric = SemanticStabilityMetric()
        with pytest.raises(TypeError, match="must be strings"):
            metric.compute(["Valid", None])


class TestSemanticStabilityKnownExamples:
    """Test with known examples and expected behaviors."""

    def test_known_paraphrase_similarity(self):
        """Known paraphrase pair should have high similarity."""
        metric = SemanticStabilityMetric()
        outputs = [
            "The cat sat on the mat",
            "A feline was seated on the rug"
        ]
        result = metric.compute(outputs)

        assert result['stability'] > 0.55, "Known paraphrases should be similar"
        assert result['n_outputs'] == 2

    def test_known_different_topics(self):
        """Known different topics should have low similarity."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Python programming tutorial",
            "Chocolate chip cookie recipe",
            "Lunar eclipse astronomy"
        ]
        result = metric.compute(outputs)

        assert result['stability'] < 0.4, "Different topics should have low stability"

    def test_score_structure_complete(self):
        """Verify all expected keys are present in result."""
        metric = SemanticStabilityMetric()
        outputs = ["Test A", "Test B"]
        result = metric.compute(outputs)

        expected_keys = {
            'stability', 'min_similarity', 'max_similarity',
            'std_similarity', 'n_outputs', 'model_name'
        }
        assert set(result.keys()) == expected_keys, "Result should have all expected keys"

    def test_score_structure_with_matrix(self):
        """Verify structure when matrix is requested."""
        metric = SemanticStabilityMetric()
        outputs = ["Test A", "Test B"]
        result = metric.compute(outputs, return_matrix=True)

        expected_keys = {
            'stability', 'min_similarity', 'max_similarity',
            'std_similarity', 'n_outputs', 'model_name', 'similarity_matrix'
        }
        assert set(result.keys()) == expected_keys

    def test_all_values_are_floats_or_valid_types(self):
        """Ensure all numeric values are proper Python floats."""
        metric = SemanticStabilityMetric()
        outputs = ["Alpha", "Beta", "Gamma"]
        result = metric.compute(outputs)

        assert isinstance(result['stability'], float)
        assert isinstance(result['min_similarity'], float)
        assert isinstance(result['max_similarity'], float)
        assert isinstance(result['std_similarity'], float)
        assert isinstance(result['n_outputs'], int)
        assert isinstance(result['model_name'], str)


class TestSemanticStabilityStatistics:
    """Test statistical properties of results."""

    def test_min_max_bounds(self):
        """Min similarity should be <= stability <= max similarity."""
        metric = SemanticStabilityMetric()
        outputs = [
            "Dogs are loyal companions",
            "Cats are independent animals",
            "Fish live in water"
        ]
        result = metric.compute(outputs)

        assert result['min_similarity'] <= result['stability']
        assert result['stability'] <= result['max_similarity']

    def test_stability_is_mean_of_pairwise(self):
        """For 2 outputs, stability should equal the single pairwise similarity."""
        metric = SemanticStabilityMetric()
        outputs = ["First text", "Second text"]
        result = metric.compute(outputs, return_matrix=True)

        # For 2 outputs, there's only one pairwise comparison
        matrix = result['similarity_matrix']
        pairwise_sim = matrix[0, 1]

        assert abs(result['stability'] - pairwise_sim) < EPSILON
        assert abs(result['min_similarity'] - pairwise_sim) < EPSILON
        assert abs(result['max_similarity'] - pairwise_sim) < EPSILON
        assert result['std_similarity'] < EPSILON  # No variance with 1 sample

    def test_three_outputs_statistics(self):
        """Verify statistics are computed correctly for 3 outputs."""
        metric = SemanticStabilityMetric()
        outputs = ["A", "B", "C"]
        result = metric.compute(outputs, return_matrix=True)

        # 3 outputs -> 3 pairwise comparisons
        matrix = result['similarity_matrix']
        pairwise = [matrix[0, 1], matrix[0, 2], matrix[1, 2]]

        assert abs(result['stability'] - np.mean(pairwise)) < EPSILON
        assert abs(result['min_similarity'] - np.min(pairwise)) < EPSILON
        assert abs(result['max_similarity'] - np.max(pairwise)) < EPSILON
        assert abs(result['std_similarity'] - np.std(pairwise)) < EPSILON


class TestSemanticStabilityCustomModel:
    """Test custom model initialization."""

    def test_custom_device_cpu(self):
        """Initialize with explicit CPU device."""
        metric = SemanticStabilityMetric(device='cpu')
        outputs = ["Test A", "Test B"]
        result = metric.compute(outputs)

        assert result['stability'] >= 0.0
        assert result['model_name'] == 'all-MiniLM-L6-v2'

    def test_default_model_name(self):
        """Verify default model name is set correctly."""
        metric = SemanticStabilityMetric()
        assert metric.model_name == 'all-MiniLM-L6-v2'

    def test_kwargs_ignored_gracefully(self):
        """Extra kwargs should be ignored without error."""
        metric = SemanticStabilityMetric()
        outputs = ["Test", "Data"]
        result = metric.compute(outputs, extra_param="ignored", another=123)

        assert 'stability' in result
        assert result['n_outputs'] == 2
