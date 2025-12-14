"""
Perplexity metric using OpenAI API for token-level log probabilities.
Measures model confidence in generated text via exp(-mean(log_probs)).
"""
from typing import Dict, List, Optional, Any
import numpy as np
from openai import OpenAI
import os


class PerplexityMetric:
    """Perplexity metric: exp(-mean(log_probs)) for model confidence measurement."""

    def __init__(self, model_name: str = 'gpt-3.5-turbo', api_key: Optional[str] = None):
        """Initialize with OpenAI model.

        Args:
            model_name: OpenAI model (default: gpt-3.5-turbo)
            api_key: OpenAI API key (default: from env OPENAI_API_KEY)
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))

    def compute(self, text: str, **kwargs) -> Dict[str, Any]:
        """Compute perplexity for given text.

        Formula: PPL = exp(-1/N × Σ log P(token_i))

        Args:
            text: Text to analyze
            **kwargs: Additional parameters (temperature, etc.)

        Returns:
            Dictionary containing:
                - perplexity: exp(-mean(log_probs))
                - log_perplexity: -mean(log_probs)
                - num_tokens: Number of tokens
                - mean_logprob: Mean log probability
                - token_perplexities: Per-token perplexity values
                - tokens: Token strings

        Raises:
            ValueError: If text is empty
            RuntimeError: If API call fails
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        logprobs_data = self._get_logprobs(text, **kwargs)
        result = self._calculate_perplexity(logprobs_data)

        return result

    def _get_logprobs(self, text: str, **kwargs) -> List[Dict[str, Any]]:
        """Get token-level log probabilities from OpenAI API.

        Args:
            text: Input text to analyze
            **kwargs: Additional API parameters (temperature, etc.)

        Returns:
            List of dicts with 'token' and 'logprob' keys

        Raises:
            RuntimeError: If API call fails
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are analyzing text."},
                    {"role": "user", "content": text}
                ],
                logprobs=True,
                max_tokens=1,
                temperature=kwargs.get('temperature', 0.0)
            )

            logprobs_list = []
            if response.choices and response.choices[0].logprobs:
                for token_info in response.choices[0].logprobs.content:
                    logprobs_list.append({
                        'token': token_info.token,
                        'logprob': token_info.logprob
                    })

            return logprobs_list

        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {str(e)}")

    def _calculate_perplexity(self, logprobs_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate perplexity from log probabilities.

        Formula: PPL = exp(-1/N × Σ log P(token_i))

        Args:
            logprobs_data: List of dicts with 'token' and 'logprob' keys

        Returns:
            Dictionary with perplexity metrics

        Raises:
            ValueError: If no log probabilities available
        """
        if not logprobs_data:
            raise ValueError("No log probabilities available")

        logprobs = [item['logprob'] for item in logprobs_data]
        tokens = [item['token'] for item in logprobs_data]

        # Mean log probability
        mean_logprob = np.mean(logprobs)

        # Log perplexity = -mean(log_probs)
        log_perplexity = -mean_logprob

        # Perplexity = exp(log_perplexity)
        perplexity = np.exp(log_perplexity)

        # Per-token perplexity
        token_perplexities = [float(np.exp(-lp)) for lp in logprobs]

        return {
            'perplexity': float(perplexity),
            'log_perplexity': float(log_perplexity),
            'num_tokens': len(tokens),
            'mean_logprob': float(mean_logprob),
            'token_perplexities': token_perplexities,
            'tokens': tokens
        }
