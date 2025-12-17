"""
Base Variator Interface

Building Block for Prompt Variation Generation

Input Data:
    - base_prompt: str - The original prompt text
    - **kwargs: Additional parameters specific to variator type

Output Data:
    - prompt: str - The generated prompt variation
    - metadata: dict - Information about the variation applied

Setup Data:
    - variator_type: str - Type of variation (baseline, few-shot, cot, etc.)
    - config: dict - Variator-specific configuration parameters
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseVariator(ABC):
    """
    Abstract base class for prompt variators.

    All variators must implement the generate_prompt method to create
    prompt variations according to specific strategies (few-shot, CoT, etc.).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the variator.

        Args:
            config: Optional configuration dictionary for the variator
        """
        self.config = config or {}
        self.variator_type = self.__class__.__name__
        self._validate_config()

    @abstractmethod
    def generate_prompt(self, base_prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate a prompt variation.

        Args:
            base_prompt: The original prompt text
            **kwargs: Additional parameters for prompt generation

        Returns:
            Dictionary containing:
                - prompt: str - The generated prompt
                - metadata: dict - Information about the variation

        Raises:
            ValueError: If base_prompt is invalid
        """
        pass

    def _validate_config(self):
        """Validate configuration parameters. Override in subclasses."""
        pass

    def _validate_base_prompt(self, base_prompt: str):
        """
        Validate base prompt input.

        Args:
            base_prompt: The prompt to validate

        Raises:
            ValueError: If prompt is invalid
        """
        if not base_prompt:
            raise ValueError("Base prompt cannot be empty")

        if not isinstance(base_prompt, str):
            raise TypeError(f"Base prompt must be a string, got {type(base_prompt)}")

        if len(base_prompt.strip()) == 0:
            raise ValueError("Base prompt cannot be only whitespace")

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about this variator.

        Returns:
            Dictionary with variator information
        """
        return {
            "variator_type": self.variator_type,
            "config": self.config,
        }
