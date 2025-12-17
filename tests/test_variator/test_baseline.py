"""
Unit tests for BaselineVariator

Tests cover:
- Basic functionality
- Edge cases (empty input, whitespace)
- Configuration validation
- Metadata generation
"""

import pytest
from src.variator.baseline import BaselineVariator


class TestBaselineVariator:
    """Test suite for BaselineVariator"""

    def test_basic_prompt_generation(self):
        """Test basic prompt generation without prefix"""
        variator = BaselineVariator(add_prefix=False)
        result = variator.generate_prompt("What is 2+2?")

        assert "prompt" in result
        assert "metadata" in result
        assert result["prompt"] == "What is 2+2?"
        assert result["metadata"]["variator_type"] == "baseline"

    def test_prompt_with_prefix(self):
        """Test prompt generation with prefix enabled"""
        variator = BaselineVariator(add_prefix=True, prefix_text="Task: ")
        result = variator.generate_prompt("Calculate the sum")

        assert "Task: Calculate the sum" in result["prompt"]
        assert result["metadata"]["has_prefix"] is True

    def test_prompt_with_system_message(self):
        """Test prompt with system message"""
        variator = BaselineVariator()
        result = variator.generate_prompt(
            "What is the capital of France?",
            system_message="You are a helpful assistant."
        )

        assert "You are a helpful assistant." in result["prompt"]
        assert "What is the capital of France?" in result["prompt"]
        assert result["metadata"]["has_system_message"] is True

    def test_custom_prefix_text(self):
        """Test custom prefix text"""
        variator = BaselineVariator(add_prefix=True, prefix_text="Question: ")
        result = variator.generate_prompt("What is AI?")

        assert "Question: What is AI?" in result["prompt"]

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError"""
        variator = BaselineVariator()

        with pytest.raises(ValueError, match="Base prompt cannot be empty"):
            variator.generate_prompt("")

    def test_whitespace_only_prompt_raises_error(self):
        """Test that whitespace-only prompt raises ValueError"""
        variator = BaselineVariator()

        with pytest.raises(ValueError, match="cannot be only whitespace"):
            variator.generate_prompt("   \n\t  ")

    def test_non_string_prompt_raises_error(self):
        """Test that non-string prompt raises TypeError"""
        variator = BaselineVariator()

        with pytest.raises(TypeError, match="must be a string"):
            variator.generate_prompt(123)

    def test_metadata_structure(self):
        """Test metadata contains expected fields"""
        variator = BaselineVariator(add_prefix=True)
        result = variator.generate_prompt("Test prompt")

        metadata = result["metadata"]
        assert "variator_type" in metadata
        assert "has_system_message" in metadata
        assert "has_prefix" in metadata
        assert "config" in metadata

    def test_get_metadata(self):
        """Test get_metadata method"""
        variator = BaselineVariator(add_prefix=True, prefix_text="Task: ")
        metadata = variator.get_metadata()

        assert metadata["variator_type"] == "BaselineVariator"
        assert "config" in metadata
        assert metadata["config"]["add_prefix"] is True

    def test_invalid_config_add_prefix(self):
        """Test invalid add_prefix configuration"""
        with pytest.raises(TypeError, match="add_prefix must be a boolean"):
            BaselineVariator(add_prefix="yes")

    def test_invalid_config_prefix_text(self):
        """Test invalid prefix_text configuration"""
        with pytest.raises(TypeError, match="prefix_text must be a string"):
            BaselineVariator(prefix_text=123)

    def test_long_prompt(self):
        """Test handling of very long prompts"""
        variator = BaselineVariator()
        long_prompt = "A" * 10000
        result = variator.generate_prompt(long_prompt)

        assert len(result["prompt"]) >= 10000

    def test_special_characters_in_prompt(self):
        """Test prompts with special characters"""
        variator = BaselineVariator()
        special_prompt = "Test with\nnewlines\tand\ttabs and 'quotes' and \"double quotes\""
        result = variator.generate_prompt(special_prompt)

        assert result["prompt"] == special_prompt

    def test_unicode_in_prompt(self):
        """Test prompts with unicode characters"""
        variator = BaselineVariator()
        unicode_prompt = "Hello 世界 🌍 café"
        result = variator.generate_prompt(unicode_prompt)

        assert unicode_prompt in result["prompt"]

    def test_multiple_calls_independence(self):
        """Test that multiple calls don't interfere with each other"""
        variator = BaselineVariator()

        result1 = variator.generate_prompt("First prompt")
        result2 = variator.generate_prompt("Second prompt")

        assert result1["prompt"] != result2["prompt"]
        assert "First prompt" in result1["prompt"]
        assert "Second prompt" in result2["prompt"]

    def test_system_message_formatting(self):
        """Test system message is properly separated from prompt"""
        variator = BaselineVariator()
        result = variator.generate_prompt(
            "Main question",
            system_message="System instruction"
        )

        # Should have proper separation
        assert "System instruction" in result["prompt"]
        assert "Main question" in result["prompt"]
        # Check they're separated
        parts = result["prompt"].split("\n\n")
        assert len(parts) >= 2
