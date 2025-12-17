"""
Few-Shot Learning Variator

Adds 1-3 examples before the main prompt to demonstrate the desired behavior.
This technique improves LLM performance by showing examples of correct responses.

Building Block Documentation:

Input Data:
    - base_prompt: str - The question/task to be answered
    - examples: List[Dict[str, str]] - List of example dictionaries with 'input' and 'output' keys
    - num_examples: Optional[int] - Number of examples to include (defaults to all provided)

Output Data:
    - prompt: str - Prompt with few-shot examples formatted before the question
    - metadata: dict - Contains example count, variator settings

Setup Data:
    - example_separator: str - Separator between examples (default: "\n\n")
    - example_format: str - Format template for examples
    - max_examples: int - Maximum number of examples to use (default: 3)
"""

from typing import Dict, Any, List, Optional
from src.variator.base import BaseVariator


class FewShotVariator(BaseVariator):
    """
    Few-shot learning variator that adds examples to prompts.

    Formats 1-3 demonstration examples before the main prompt to guide
    the model's behavior through in-context learning.
    """

    def __init__(
        self,
        example_separator: str = "\n\n",
        example_format: str = "Input: {input}\nOutput: {output}",
        max_examples: int = 3,
        **kwargs
    ):
        """
        Initialize few-shot variator.

        Args:
            example_separator: String to separate examples
            example_format: Format template for each example
            max_examples: Maximum number of examples to include
            **kwargs: Additional configuration options
        """
        config = {
            "example_separator": example_separator,
            "example_format": example_format,
            "max_examples": max_examples,
            **kwargs
        }
        super().__init__(config)
        self.example_separator = example_separator
        self.example_format = example_format
        self.max_examples = max_examples

    def generate_prompt(
        self,
        base_prompt: str,
        examples: List[Dict[str, str]],
        num_examples: Optional[int] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate few-shot prompt with examples.

        Args:
            base_prompt: The question/task to answer
            examples: List of example dicts with 'input' and 'output'
            num_examples: Number of examples to use (None = use all, up to max)
            system_message: Optional system instruction
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - prompt: Few-shot formatted prompt
                - metadata: Information about examples used

        Raises:
            ValueError: If examples are invalid or empty
        """
        # Validate inputs
        self._validate_base_prompt(base_prompt)
        self._validate_examples(examples)

        # Determine number of examples to use
        if num_examples is None:
            num_examples = min(len(examples), self.max_examples)
        else:
            num_examples = min(num_examples, len(examples), self.max_examples)

        if num_examples < 1:
            raise ValueError("Must include at least 1 example for few-shot learning")

        # Format examples
        selected_examples = examples[:num_examples]
        formatted_examples = []

        for i, example in enumerate(selected_examples, 1):
            formatted_example = self.example_format.format(
                input=example["input"],
                output=example["output"]
            )
            formatted_examples.append(formatted_example)

        # Build prompt
        prompt_parts = []

        if system_message:
            prompt_parts.append(system_message.strip())

        # Add examples section
        examples_text = self.example_separator.join(formatted_examples)
        prompt_parts.append(f"Here are some examples:\n\n{examples_text}")

        # Add the actual query
        prompt_parts.append(f"Now, please answer this:\nInput: {base_prompt}\nOutput:")

        final_prompt = "\n\n".join(prompt_parts)

        # Return with metadata
        return {
            "prompt": final_prompt,
            "metadata": {
                "variator_type": "few_shot",
                "num_examples": num_examples,
                "total_available_examples": len(examples),
                "has_system_message": system_message is not None,
                "config": self.config
            }
        }

    def _validate_examples(self, examples: List[Dict[str, str]]):
        """
        Validate example format.

        Args:
            examples: List of example dictionaries

        Raises:
            ValueError: If examples are invalid
            TypeError: If examples have wrong type
        """
        if not examples:
            raise ValueError("Examples list cannot be empty")

        if not isinstance(examples, list):
            raise TypeError(f"Examples must be a list, got {type(examples)}")

        for i, example in enumerate(examples):
            if not isinstance(example, dict):
                raise TypeError(f"Example {i} must be a dictionary, got {type(example)}")

            if "input" not in example:
                raise ValueError(f"Example {i} missing required 'input' key")

            if "output" not in example:
                raise ValueError(f"Example {i} missing required 'output' key")

            if not isinstance(example["input"], str):
                raise TypeError(f"Example {i} 'input' must be a string")

            if not isinstance(example["output"], str):
                raise TypeError(f"Example {i} 'output' must be a string")

    def _validate_config(self):
        """Validate few-shot variator configuration."""
        max_examples = self.config.get("max_examples", 3)
        if not isinstance(max_examples, int) or max_examples < 1:
            raise ValueError("max_examples must be a positive integer")

        if max_examples > 5:
            raise ValueError("max_examples should not exceed 5 (too many examples can degrade performance)")
