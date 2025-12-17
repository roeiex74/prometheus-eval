"""
Chain of Thought (CoT) Variator

Adds step-by-step reasoning prompts to encourage the model to break down
problems and show its work. Research shows CoT improves accuracy from 18% to 58%
on complex reasoning tasks (GSM8K benchmark).

Reference: Wei et al. (2022) "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"

Building Block Documentation:

Input Data:
    - base_prompt: str - The question/task requiring reasoning
    - custom_trigger: Optional[str] - Custom CoT trigger phrase

Output Data:
    - prompt: str - Prompt with CoT trigger encouraging step-by-step reasoning
    - metadata: dict - Contains CoT settings and trigger used

Setup Data:
    - cot_trigger: str - Phrase that triggers step-by-step reasoning
    - add_reasoning_prefix: bool - Whether to add explicit reasoning instruction
    - example_reasoning: Optional[str] - Example of step-by-step reasoning
"""

from typing import Dict, Any, Optional
from src.variator.base import BaseVariator


class ChainOfThoughtVariator(BaseVariator):
    """
    Chain-of-Thought prompt variator.

    Adds triggers and instructions that encourage models to show
    step-by-step reasoning before providing final answers.
    """

    def __init__(
        self,
        cot_trigger: str = "Let's think step by step.",
        add_reasoning_prefix: bool = True,
        reasoning_prefix: str = "To solve this, let's break it down:",
        **kwargs
    ):
        """
        Initialize CoT variator.

        Args:
            cot_trigger: Phrase to trigger step-by-step reasoning
            add_reasoning_prefix: Whether to add explicit reasoning instruction
            reasoning_prefix: Text for reasoning instruction
            **kwargs: Additional configuration options
        """
        config = {
            "cot_trigger": cot_trigger,
            "add_reasoning_prefix": add_reasoning_prefix,
            "reasoning_prefix": reasoning_prefix,
            **kwargs
        }
        super().__init__(config)
        self.cot_trigger = cot_trigger
        self.add_reasoning_prefix = add_reasoning_prefix
        self.reasoning_prefix = reasoning_prefix

    def generate_prompt(
        self,
        base_prompt: str,
        custom_trigger: Optional[str] = None,
        system_message: Optional[str] = None,
        example_reasoning: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate Chain-of-Thought prompt.

        Args:
            base_prompt: The question/task to answer
            custom_trigger: Optional custom CoT trigger phrase
            system_message: Optional system instruction
            example_reasoning: Optional example of step-by-step reasoning
            **kwargs: Additional parameters

        Returns:
            Dictionary with:
                - prompt: CoT-formatted prompt
                - metadata: Information about CoT settings
        """
        # Validate input
        self._validate_base_prompt(base_prompt)

        # Use custom trigger if provided
        trigger = custom_trigger if custom_trigger else self.cot_trigger

        # Build prompt
        prompt_parts = []

        if system_message:
            prompt_parts.append(system_message.strip())

        # Add reasoning instruction if enabled
        if self.add_reasoning_prefix:
            prompt_parts.append(self.reasoning_prefix)

        # Add example reasoning if provided
        if example_reasoning:
            prompt_parts.append(f"Example reasoning:\n{example_reasoning}")

        # Add the question
        prompt_parts.append(base_prompt)

        # Add CoT trigger
        prompt_parts.append(trigger)

        final_prompt = "\n\n".join(prompt_parts)

        # Return with metadata
        return {
            "prompt": final_prompt,
            "metadata": {
                "variator_type": "chain_of_thought",
                "cot_trigger": trigger,
                "has_reasoning_prefix": self.add_reasoning_prefix,
                "has_example_reasoning": example_reasoning is not None,
                "has_system_message": system_message is not None,
                "config": self.config
            }
        }

    def _validate_config(self):
        """Validate CoT variator configuration."""
        if not isinstance(self.config.get("cot_trigger", ""), str):
            raise TypeError("cot_trigger must be a string")

        trigger = self.config.get("cot_trigger", "")
        if not trigger.strip():
            raise ValueError("cot_trigger cannot be empty")

        if not isinstance(self.config.get("add_reasoning_prefix", True), bool):
            raise TypeError("add_reasoning_prefix must be a boolean")
