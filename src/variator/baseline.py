"""
Baseline Variator

Simple prompt wrapper that returns the prompt with minimal formatting.
This serves as the control group for comparing prompt improvements.

Building Block Documentation:

Input Data:
    - base_prompt: str - The original question/task prompt
    - system_message: Optional[str] - Optional system instruction

Output Data:
    - prompt: str - The formatted prompt (may include system message)
    - metadata: dict - Contains variator type and applied formatting

Setup Data:
    - add_prefix: bool - Whether to add a task prefix (default: False)
    - prefix_text: str - Custom prefix text if add_prefix is True
"""

from typing import Dict, Any, Optional
from src.variator.base import BaseVariator


class BaselineVariator(BaseVariator):
    """
    Baseline prompt variator that applies minimal formatting.

    This variator serves as the control condition for experiments,
    returning prompts with basic formatting but no advanced techniques
    like few-shot learning or chain-of-thought.
    """

    def __init__(
        self,
        add_prefix: bool = False,
        prefix_text: str = "Task: ",
        **kwargs
    ):
        """
        Initialize baseline variator.

        Args:
            add_prefix: Whether to add a task prefix
            prefix_text: Text to use as prefix
            **kwargs: Additional configuration options
        """
        config = {
            "add_prefix": add_prefix,
            "prefix_text": prefix_text,
            **kwargs
        }
        super().__init__(config)
        self.add_prefix = add_prefix
        self.prefix_text = prefix_text

    def generate_prompt(
        self,
        base_prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate baseline prompt with minimal formatting.

        Args:
            base_prompt: The original prompt text
            system_message: Optional system instruction
            **kwargs: Additional parameters (unused in baseline)

        Returns:
            Dictionary with:
                - prompt: The formatted prompt string
                - metadata: Information about applied formatting
        """
        # Validate input
        self._validate_base_prompt(base_prompt)

        # Build prompt
        prompt_parts = []

        if system_message:
            prompt_parts.append(system_message.strip())

        if self.add_prefix:
            prompt_parts.append(f"{self.prefix_text}{base_prompt}")
        else:
            prompt_parts.append(base_prompt)

        final_prompt = "\n\n".join(prompt_parts)

        # Return with metadata
        return {
            "prompt": final_prompt,
            "metadata": {
                "variator_type": "baseline",
                "has_system_message": system_message is not None,
                "has_prefix": self.add_prefix,
                "config": self.config
            }
        }

    def _validate_config(self):
        """Validate baseline variator configuration."""
        if not isinstance(self.config.get("add_prefix", False), bool):
            raise TypeError("add_prefix must be a boolean")

        if not isinstance(self.config.get("prefix_text", ""), str):
            raise TypeError("prefix_text must be a string")
