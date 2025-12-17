"""
Chain of Thought Plus (CoT++) Variator

Combines Chain-of-Thought reasoning with majority voting (self-consistency).
Generates multiple reasoning paths and selects the most common answer.

This technique further improves CoT by sampling multiple reasoning chains
and selecting the answer that appears most frequently.

Reference: Wang et al. (2022) "Self-Consistency Improves Chain of Thought Reasoning in Language Models"

Building Block Documentation:

Input Data:
    - base_prompt: str - The question/task requiring reasoning
    - num_samples: int - Number of reasoning paths to generate (default: 5)

Output Data:
    - prompt: str - CoT prompt for generating multiple samples
    - metadata: dict - Contains sampling configuration

Setup Data:
    - num_samples: int - How many times to sample (default: 5, max: 10)
    - temperature: float - Sampling temperature for diversity (default: 0.7)
    - cot_trigger: str - CoT trigger phrase
"""

from typing import Dict, Any, Optional
from src.variator.cot import ChainOfThoughtVariator


class CoTPlusVariator(ChainOfThoughtVariator):
    """
    CoT++ variator with majority voting (self-consistency).

    Extends ChainOfThoughtVariator to support multiple sampling
    for improved accuracy through majority voting.
    """

    def __init__(
        self,
        num_samples: int = 5,
        temperature: float = 0.7,
        cot_trigger: str = "Let's think step by step and show all reasoning.",
        add_reasoning_prefix: bool = True,
        **kwargs
    ):
        """
        Initialize CoT++ variator.

        Args:
            num_samples: Number of reasoning paths to generate
            temperature: Sampling temperature for diversity
            cot_trigger: Phrase to trigger step-by-step reasoning
            add_reasoning_prefix: Whether to add reasoning instruction
            **kwargs: Additional configuration options
        """
        super().__init__(
            cot_trigger=cot_trigger,
            add_reasoning_prefix=add_reasoning_prefix,
            **kwargs
        )

        # Update config with CoT++ specific parameters
        self.config.update({
            "num_samples": num_samples,
            "temperature": temperature,
        })

        self.num_samples = num_samples
        self.temperature = temperature

    def generate_prompt(
        self,
        base_prompt: str,
        custom_trigger: Optional[str] = None,
        system_message: Optional[str] = None,
        example_reasoning: Optional[str] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate CoT++ prompt with majority voting instructions.

        Args:
            base_prompt: The question/task to answer
            custom_trigger: Optional custom CoT trigger phrase
            system_message: Optional system instruction
            example_reasoning: Optional example reasoning
            num_samples: Override default number of samples
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - prompt: CoT-formatted prompt
                - metadata: Information about CoT++ settings including num_samples
        """
        # Use parent's CoT prompt generation
        result = super().generate_prompt(
            base_prompt=base_prompt,
            custom_trigger=custom_trigger,
            system_message=system_message,
            example_reasoning=example_reasoning,
            **kwargs
        )

        # Update metadata for CoT++
        samples = num_samples if num_samples is not None else self.num_samples

        result["metadata"].update({
            "variator_type": "cot_plus",
            "num_samples": samples,
            "temperature": self.temperature,
            "uses_majority_voting": True,
        })

        return result

    def aggregate_responses(
        self,
        responses: list,
        extract_answer_fn=None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple CoT responses using majority voting.

        This method should be called AFTER generating multiple responses
        from the LLM using the same prompt.

        Args:
            responses: List of response strings from multiple samples
            extract_answer_fn: Optional function to extract final answer
                             from reasoning chain. If None, uses last line.

        Returns:
            Dictionary with:
                - final_answer: The most common answer
                - vote_counts: Dictionary of answer frequencies
                - all_responses: All sampled responses

        Raises:
            ValueError: If responses list is empty
        """
        if not responses:
            raise ValueError("Cannot aggregate empty responses list")

        if len(responses) < self.num_samples:
            import warnings
            warnings.warn(
                f"Only {len(responses)} responses provided, "
                f"expected {self.num_samples}"
            )

        # Extract answers from responses
        if extract_answer_fn is None:
            # Default: use last non-empty line as answer
            extract_answer_fn = self._default_answer_extractor

        answers = []
        for response in responses:
            try:
                answer = extract_answer_fn(response)
                if answer:
                    answers.append(answer)
            except Exception:
                # Skip invalid responses
                continue

        if not answers:
            raise ValueError("Could not extract any valid answers from responses")

        # Count votes
        vote_counts = {}
        for answer in answers:
            vote_counts[answer] = vote_counts.get(answer, 0) + 1

        # Get majority answer
        final_answer = max(vote_counts.items(), key=lambda x: x[1])[0]

        return {
            "final_answer": final_answer,
            "vote_counts": vote_counts,
            "all_responses": responses,
            "total_samples": len(responses),
            "valid_answers": len(answers),
        }

    def _default_answer_extractor(self, response: str) -> str:
        """
        Default answer extraction from CoT response.

        Extracts the last non-empty line as the final answer.

        Args:
            response: The full CoT response text

        Returns:
            Extracted answer string
        """
        lines = [line.strip() for line in response.strip().split('\n')]
        lines = [line for line in lines if line]

        if not lines:
            return ""

        # Return last non-empty line
        return lines[-1]

    def _validate_config(self):
        """Validate CoT++ variator configuration."""
        super()._validate_config()

        num_samples = self.config.get("num_samples", 5)
        if not isinstance(num_samples, int) or num_samples < 2:
            raise ValueError("num_samples must be an integer >= 2")

        if num_samples > 10:
            raise ValueError(
                "num_samples should not exceed 10 (high API cost and diminishing returns)"
            )

        temperature = self.config.get("temperature", 0.7)
        if not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a number")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0")
