"""
Unit tests for ChainOfThoughtVariator

Tests cover:
- CoT trigger phrase functionality
- Reasoning prefix options
- Example reasoning inclusion
- Configuration validation
"""

import pytest
from src.variator.cot import ChainOfThoughtVariator


class TestChainOfThoughtVariator:
    """Test suite for ChainOfThoughtVariator"""

    def test_basic_cot_generation(self):
        """Test basic CoT prompt generation"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt("What is 15 + 27?")

        assert "prompt" in result
        assert "metadata" in result
        assert "Let's think step by step" in result["prompt"]
        assert "What is 15 + 27?" in result["prompt"]
        assert result["metadata"]["variator_type"] == "chain_of_thought"

    def test_custom_cot_trigger(self):
        """Test with custom CoT trigger phrase"""
        variator = ChainOfThoughtVariator(
            cot_trigger="Let's solve this carefully:"
        )
        result = variator.generate_prompt("Solve this problem")

        assert "Let's solve this carefully:" in result["prompt"]
        assert result["metadata"]["cot_trigger"] == "Let's solve this carefully:"

    def test_with_reasoning_prefix(self):
        """Test with reasoning prefix enabled"""
        variator = ChainOfThoughtVariator(add_reasoning_prefix=True)
        result = variator.generate_prompt("Calculate the area")

        assert "To solve this, let's break it down:" in result["prompt"]
        assert result["metadata"]["has_reasoning_prefix"] is True

    def test_without_reasoning_prefix(self):
        """Test with reasoning prefix disabled"""
        variator = ChainOfThoughtVariator(add_reasoning_prefix=False)
        result = variator.generate_prompt("Calculate the area")

        assert "To solve this, let's break it down:" not in result["prompt"]
        assert result["metadata"]["has_reasoning_prefix"] is False

    def test_custom_reasoning_prefix(self):
        """Test with custom reasoning prefix text"""
        variator = ChainOfThoughtVariator(
            add_reasoning_prefix=True,
            reasoning_prefix="Here's my approach:"
        )
        result = variator.generate_prompt("Solve this")

        assert "Here's my approach:" in result["prompt"]

    def test_with_example_reasoning(self):
        """Test with example reasoning provided"""
        variator = ChainOfThoughtVariator()
        example = "Step 1: Identify the numbers. Step 2: Add them. Step 3: State the answer."

        result = variator.generate_prompt(
            "What is 10 + 20?",
            example_reasoning=example
        )

        assert "Example reasoning:" in result["prompt"]
        assert example in result["prompt"]
        assert result["metadata"]["has_example_reasoning"] is True

    def test_without_example_reasoning(self):
        """Test without example reasoning"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt("Test question")

        assert result["metadata"]["has_example_reasoning"] is False

    def test_with_system_message(self):
        """Test CoT with system message"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt(
            "What is the capital of France?",
            system_message="You are a geography expert."
        )

        assert "You are a geography expert." in result["prompt"]
        assert result["metadata"]["has_system_message"] is True

    def test_custom_trigger_overrides_default(self):
        """Test that custom_trigger parameter overrides default"""
        variator = ChainOfThoughtVariator(
            cot_trigger="Default trigger"
        )
        result = variator.generate_prompt(
            "Test",
            custom_trigger="Override trigger"
        )

        assert "Override trigger" in result["prompt"]
        assert "Default trigger" not in result["prompt"]
        assert result["metadata"]["cot_trigger"] == "Override trigger"

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError"""
        variator = ChainOfThoughtVariator()

        with pytest.raises(ValueError, match="Base prompt cannot be empty"):
            variator.generate_prompt("")

    def test_whitespace_only_prompt_raises_error(self):
        """Test that whitespace-only prompt raises ValueError"""
        variator = ChainOfThoughtVariator()

        with pytest.raises(ValueError, match="cannot be only whitespace"):
            variator.generate_prompt("   \n\t  ")

    def test_non_string_prompt_raises_error(self):
        """Test that non-string prompt raises TypeError"""
        variator = ChainOfThoughtVariator()

        with pytest.raises(TypeError, match="must be a string"):
            variator.generate_prompt(12345)

    def test_empty_cot_trigger_raises_error(self):
        """Test that empty CoT trigger raises ValueError"""
        with pytest.raises(ValueError, match="cot_trigger cannot be empty"):
            ChainOfThoughtVariator(cot_trigger="")

    def test_empty_cot_trigger_whitespace_raises_error(self):
        """Test that whitespace-only CoT trigger raises ValueError"""
        with pytest.raises(ValueError, match="cot_trigger cannot be empty"):
            ChainOfThoughtVariator(cot_trigger="   ")

    def test_invalid_cot_trigger_type(self):
        """Test invalid CoT trigger type"""
        with pytest.raises(TypeError, match="cot_trigger must be a string"):
            ChainOfThoughtVariator(cot_trigger=123)

    def test_invalid_add_reasoning_prefix_type(self):
        """Test invalid add_reasoning_prefix type"""
        with pytest.raises(TypeError, match="add_reasoning_prefix must be a boolean"):
            ChainOfThoughtVariator(add_reasoning_prefix="yes")

    def test_metadata_structure(self):
        """Test metadata contains expected fields"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt("Test")

        metadata = result["metadata"]
        assert "variator_type" in metadata
        assert "cot_trigger" in metadata
        assert "has_reasoning_prefix" in metadata
        assert "has_example_reasoning" in metadata
        assert "has_system_message" in metadata
        assert "config" in metadata

    def test_get_metadata(self):
        """Test get_metadata method"""
        variator = ChainOfThoughtVariator(
            cot_trigger="Custom trigger",
            add_reasoning_prefix=False
        )
        metadata = variator.get_metadata()

        assert metadata["variator_type"] == "ChainOfThoughtVariator"
        assert "config" in metadata
        assert metadata["config"]["cot_trigger"] == "Custom trigger"
        assert metadata["config"]["add_reasoning_prefix"] is False

    def test_prompt_structure_all_components(self):
        """Test prompt structure with all components"""
        variator = ChainOfThoughtVariator(add_reasoning_prefix=True)
        example = "Step 1: Do this. Step 2: Do that."

        result = variator.generate_prompt(
            "Solve the problem",
            system_message="You are helpful.",
            example_reasoning=example
        )

        prompt = result["prompt"]
        parts = prompt.split("\n\n")

        # Should have multiple parts properly separated
        assert len(parts) >= 4  # system, reasoning prefix, example, question, trigger

    def test_long_prompt(self):
        """Test handling of very long prompts"""
        variator = ChainOfThoughtVariator()
        long_prompt = "Calculate " + "A" * 10000

        result = variator.generate_prompt(long_prompt)
        assert len(result["prompt"]) > 10000

    def test_unicode_in_prompt(self):
        """Test prompts with unicode characters"""
        variator = ChainOfThoughtVariator()
        unicode_prompt = "Translate: 你好世界"

        result = variator.generate_prompt(unicode_prompt)
        assert "你好世界" in result["prompt"]

    def test_special_characters_in_prompt(self):
        """Test prompts with special characters"""
        variator = ChainOfThoughtVariator()
        special_prompt = "Code: print('hello\nworld')"

        result = variator.generate_prompt(special_prompt)
        assert special_prompt in result["prompt"]

    def test_multiple_calls_independence(self):
        """Test that multiple calls don't interfere"""
        variator = ChainOfThoughtVariator()

        result1 = variator.generate_prompt("First question")
        result2 = variator.generate_prompt("Second question")

        assert "First question" in result1["prompt"]
        assert "Second question" in result2["prompt"]
        assert result1["prompt"] != result2["prompt"]

    def test_cot_trigger_position(self):
        """Test that CoT trigger appears at the end"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt("Test question")

        prompt = result["prompt"]
        trigger_pos = prompt.find("Let's think step by step")
        question_pos = prompt.find("Test question")

        # Trigger should come after question
        assert trigger_pos > question_pos

    def test_reasoning_prefix_before_question(self):
        """Test that reasoning prefix appears before question"""
        variator = ChainOfThoughtVariator(add_reasoning_prefix=True)
        result = variator.generate_prompt("Test question")

        prompt = result["prompt"]
        prefix_pos = prompt.find("To solve this, let's break it down:")
        question_pos = prompt.find("Test question")

        # Prefix should come before question
        assert prefix_pos < question_pos

    def test_example_reasoning_before_question(self):
        """Test that example reasoning appears before question"""
        variator = ChainOfThoughtVariator()
        example = "Step 1, Step 2"
        result = variator.generate_prompt(
            "Test question",
            example_reasoning=example
        )

        prompt = result["prompt"]
        example_pos = prompt.find(example)
        question_pos = prompt.find("Test question")

        # Example should come before question
        assert example_pos < question_pos

    def test_system_message_at_start(self):
        """Test that system message appears at the start"""
        variator = ChainOfThoughtVariator()
        result = variator.generate_prompt(
            "Test question",
            system_message="System instruction"
        )

        prompt = result["prompt"]
        system_pos = prompt.find("System instruction")
        question_pos = prompt.find("Test question")

        # System message should come before question
        assert system_pos < question_pos
