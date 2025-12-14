"""
Comprehensive test suite for Perplexity metric.
Tests OpenAI API integration and perplexity calculation.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.metrics.logic.perplexity import PerplexityMetric

EPSILON = 1e-4


class TestPerplexityBasic:
    """Test basic perplexity functionality."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_simple_text(self, mock_openai_class):
        """Test perplexity computation with simple text."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_choice = Mock()
        mock_logprobs = Mock()

        mock_token1 = Mock()
        mock_token1.token = 'Hello'
        mock_token1.logprob = -0.5

        mock_token2 = Mock()
        mock_token2.token = ' world'
        mock_token2.logprob = -0.3

        mock_logprobs.content = [mock_token1, mock_token2]
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("Hello world")

        assert 'perplexity' in result
        assert 'log_perplexity' in result
        assert result['num_tokens'] == 2
        assert result['perplexity'] > 0

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_perplexity_range(self, mock_openai_class):
        """Verify perplexity is always positive."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = self._create_mock_response([
            ('test', -1.0),
            ('tokens', -2.0),
            ('here', -0.5)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test tokens here")

        assert result['perplexity'] > 0
        assert result['log_perplexity'] >= 0

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_return_structure(self, mock_openai_class):
        """Check all required keys are present in results."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = self._create_mock_response([('test', -0.5)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test")

        required_keys = {
            'perplexity', 'log_perplexity', 'num_tokens',
            'mean_logprob', 'token_perplexities', 'tokens'
        }
        assert required_keys.issubset(result.keys())

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_num_tokens_count(self, mock_openai_class):
        """Verify token count accuracy."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        tokens = [('a', -0.5), ('b', -0.6), ('c', -0.4), ('d', -0.7)]
        mock_response = self._create_mock_response(tokens)
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("a b c d")

        assert result['num_tokens'] == 4
        assert len(result['tokens']) == 4
        assert len(result['token_perplexities']) == 4

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_tokens_list(self, mock_openai_class):
        """Verify tokens are correctly extracted."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        tokens = [('Hello', -0.5), (' world', -0.3), ('!', -0.2)]
        mock_response = self._create_mock_response(tokens)
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("Hello world!")

        assert result['tokens'] == ['Hello', ' world', '!']

    @staticmethod
    def _create_mock_response(token_logprob_pairs):
        """Helper to create mock OpenAI response."""
        mock_response = Mock()
        mock_choice = Mock()
        mock_logprobs = Mock()

        mock_tokens = []
        for token_text, logprob_value in token_logprob_pairs:
            mock_token = Mock()
            mock_token.token = token_text
            mock_token.logprob = logprob_value
            mock_tokens.append(mock_token)

        mock_logprobs.content = mock_tokens
        mock_choice.logprobs = mock_logprobs
        mock_response.choices = [mock_choice]

        return mock_response


class TestPerplexityMath:
    """Test mathematical correctness of perplexity calculations."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_log_perplexity_relationship(self, mock_openai_class):
        """Verify: perplexity = exp(log_perplexity)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([
            ('test', -0.5),
            ('text', -0.3)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test text")

        expected_ppl = np.exp(result['log_perplexity'])
        assert abs(result['perplexity'] - expected_ppl) < EPSILON

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_mean_logprob_calculation(self, mock_openai_class):
        """Verify: log_perplexity = -mean_logprob."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        logprobs = [-0.5, -0.3, -0.7]
        tokens_data = [(f'tok{i}', lp) for i, lp in enumerate(logprobs)]
        mock_response = TestPerplexityBasic._create_mock_response(tokens_data)
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test")

        expected_mean = np.mean(logprobs)
        assert abs(result['mean_logprob'] - expected_mean) < EPSILON
        assert abs(result['log_perplexity'] - (-expected_mean)) < EPSILON

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_token_perplexities_length(self, mock_openai_class):
        """Verify per-token list matches num_tokens."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        tokens_data = [(f't{i}', -0.5 * i) for i in range(1, 6)]
        mock_response = TestPerplexityBasic._create_mock_response(tokens_data)
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test")

        assert len(result['token_perplexities']) == result['num_tokens']
        assert len(result['token_perplexities']) == 5

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_token_perplexity_calculation(self, mock_openai_class):
        """Verify per-token perplexity = exp(-logprob)."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        logprob = -0.5
        mock_response = TestPerplexityBasic._create_mock_response([('test', logprob)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test")

        expected_token_ppl = np.exp(-logprob)
        assert abs(result['token_perplexities'][0] - expected_token_ppl) < EPSILON


class TestPerplexityEdgeCases:
    """Test edge cases and error handling."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_empty_text_raises_error(self, mock_openai_class):
        """Empty string should raise ValueError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        metric = PerplexityMetric(api_key='test-key')

        with pytest.raises(ValueError, match="Text cannot be empty"):
            metric.compute("")

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_whitespace_only_raises_error(self, mock_openai_class):
        """Whitespace-only string should raise ValueError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        metric = PerplexityMetric(api_key='test-key')

        with pytest.raises(ValueError, match="Text cannot be empty"):
            metric.compute("   ")

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_none_text_raises_error(self, mock_openai_class):
        """None text should raise ValueError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        metric = PerplexityMetric(api_key='test-key')

        with pytest.raises(ValueError, match="Text cannot be empty"):
            metric.compute(None)

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_no_logprobs_raises_error(self, mock_openai_class):
        """No log probabilities should raise ValueError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.logprobs = None
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')

        with pytest.raises(ValueError, match="No log probabilities available"):
            metric.compute("test")


class TestPerplexityAPI:
    """Test API integration and error handling."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_api_call_parameters(self, mock_openai_class):
        """Verify correct API parameters are passed."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([('test', -0.5)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(model_name='gpt-4', api_key='test-key')
        metric.compute("test text")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-4'
        assert call_args[1]['logprobs'] is True
        assert call_args[1]['max_tokens'] == 1

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_api_error_handling(self, mock_openai_class):
        """API failure should raise RuntimeError."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_client.chat.completions.create.side_effect = Exception("API Error")

        metric = PerplexityMetric(api_key='test-key')

        with pytest.raises(RuntimeError, match="OpenAI API call failed"):
            metric.compute("test")

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_custom_model_name(self, mock_openai_class):
        """Test with different model selection."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        metric = PerplexityMetric(model_name='gpt-4-turbo', api_key='test-key')
        assert metric.model_name == 'gpt-4-turbo'

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_temperature_parameter(self, mock_openai_class):
        """Test custom temperature parameter."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([('test', -0.5)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        metric.compute("test", temperature=0.7)

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.7

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_default_temperature(self, mock_openai_class):
        """Test default temperature is 0.0."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([('test', -0.5)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        metric.compute("test")

        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.0

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_api_key_from_env(self, mock_openai_class):
        """Test API key is loaded from environment."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            metric = PerplexityMetric()
            # Verify OpenAI was called with env key
            mock_openai_class.assert_called_with(api_key='env-key')


class TestPerplexityInterpretation:
    """Test perplexity value interpretation and ranges."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_high_confidence_low_perplexity(self, mock_openai_class):
        """High probability (low negative logprob) should yield low perplexity."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Very confident predictions (high probability)
        mock_response = TestPerplexityBasic._create_mock_response([
            ('The', -0.01),
            ('cat', -0.02),
            ('sat', -0.01)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("The cat sat")

        # Low perplexity indicates high confidence
        assert result['perplexity'] < 2.0

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_low_confidence_high_perplexity(self, mock_openai_class):
        """Low probability (high negative logprob) should yield high perplexity."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Uncertain predictions (low probability)
        mock_response = TestPerplexityBasic._create_mock_response([
            ('xzq', -5.0),
            ('qwp', -4.5),
            ('zyx', -5.2)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("xzq qwp zyx")

        # High perplexity indicates low confidence
        assert result['perplexity'] > 50.0

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_perplexity_variance(self, mock_openai_class):
        """Test that varied token probabilities affect perplexity."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        # Mixed probabilities
        mock_response = TestPerplexityBasic._create_mock_response([
            ('common', -0.1),
            ('rare', -3.0),
            ('word', -0.2)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("common rare word")

        # Perplexity should be moderate
        assert 1.0 < result['perplexity'] < 100.0
        # Token perplexities should vary
        assert max(result['token_perplexities']) > min(result['token_perplexities'])


class TestPerplexityNumericalStability:
    """Test numerical stability and edge cases in calculations."""

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_single_token(self, mock_openai_class):
        """Test with single token."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([('test', -1.0)])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("test")

        assert result['num_tokens'] == 1
        assert result['perplexity'] == pytest.approx(np.exp(1.0), abs=EPSILON)

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_very_negative_logprobs(self, mock_openai_class):
        """Test with very negative log probabilities."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([
            ('a', -10.0),
            ('b', -15.0)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("a b")

        # Should handle large exponents
        assert result['perplexity'] > 1e5
        assert np.isfinite(result['perplexity'])

    @patch('src.metrics.logic.perplexity.OpenAI')
    def test_near_zero_logprobs(self, mock_openai_class):
        """Test with near-zero log probabilities."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        mock_response = TestPerplexityBasic._create_mock_response([
            ('a', -0.0001),
            ('b', -0.0002)
        ])
        mock_client.chat.completions.create.return_value = mock_response

        metric = PerplexityMetric(api_key='test-key')
        result = metric.compute("a b")

        # Very confident predictions
        assert 1.0 <= result['perplexity'] < 1.001
        assert np.isfinite(result['perplexity'])
