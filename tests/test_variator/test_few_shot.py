"""
Unit tests for FewShotVariator

Tests cover:
- Example formatting
- Variable number of examples (1-3)
- Edge cases and validation
- Configuration handling
"""

import pytest
from src.variator.few_shot import FewShotVariator


class TestFewShotVariator:
    """Test suite for FewShotVariator"""

    @pytest.fixture
    def sample_examples(self):
        """Fixture providing sample examples"""
        return [
            {"input": "What is 2+2?", "output": "4"},
            {"input": "What is 3+3?", "output": "6"},
            {"input": "What is 5+5?", "output": "10"},
        ]

    def test_basic_few_shot_generation(self, sample_examples):
        """Test basic few-shot prompt generation"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "What is 7+7?",
            examples=sample_examples
        )

        assert "prompt" in result
        assert "metadata" in result
        assert "What is 2+2?" in result["prompt"]
        assert "What is 7+7?" in result["prompt"]
        assert result["metadata"]["variator_type"] == "few_shot"

    def test_single_example(self, sample_examples):
        """Test with single example"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "What is 7+7?",
            examples=sample_examples,
            num_examples=1
        )

        assert result["metadata"]["num_examples"] == 1
        assert "What is 2+2?" in result["prompt"]
        # Should not contain second example
        assert "What is 3+3?" not in result["prompt"]

    def test_max_examples_limit(self, sample_examples):
        """Test that max_examples is respected"""
        variator = FewShotVariator(max_examples=2)

        # Provide 3 examples but max is 2
        result = variator.generate_prompt(
            "What is 7+7?",
            examples=sample_examples
        )

        assert result["metadata"]["num_examples"] == 2
        # Should contain first two examples only
        assert "What is 2+2?" in result["prompt"]
        assert "What is 3+3?" in result["prompt"]
        assert "What is 5+5?" not in result["prompt"]

    def test_empty_examples_raises_error(self):
        """Test that empty examples list raises ValueError"""
        variator = FewShotVariator()

        with pytest.raises(ValueError, match="Examples list cannot be empty"):
            variator.generate_prompt("Test", examples=[])

    def test_examples_not_list_raises_error(self):
        """Test that non-list examples raises TypeError"""
        variator = FewShotVariator()

        with pytest.raises(TypeError, match="Examples must be a list"):
            variator.generate_prompt("Test", examples="not a list")

    def test_missing_input_key_raises_error(self, sample_examples):
        """Test that example missing 'input' key raises ValueError"""
        variator = FewShotVariator()
        bad_examples = [{"output": "4"}]  # Missing 'input'

        with pytest.raises(ValueError, match="missing required 'input' key"):
            variator.generate_prompt("Test", examples=bad_examples)

    def test_missing_output_key_raises_error(self):
        """Test that example missing 'output' key raises ValueError"""
        variator = FewShotVariator()
        bad_examples = [{"input": "Question"}]  # Missing 'output'

        with pytest.raises(ValueError, match="missing required 'output' key"):
            variator.generate_prompt("Test", examples=bad_examples)

    def test_non_dict_example_raises_error(self):
        """Test that non-dict example raises TypeError"""
        variator = FewShotVariator()
        bad_examples = ["not a dict"]

        with pytest.raises(TypeError, match="must be a dictionary"):
            variator.generate_prompt("Test", examples=bad_examples)

    def test_custom_separator(self, sample_examples):
        """Test custom example separator"""
        variator = FewShotVariator(example_separator="\n---\n")
        result = variator.generate_prompt("Test", examples=sample_examples[:2])

        # Check separator is used
        assert "\n---\n" in result["prompt"]

    def test_custom_format(self, sample_examples):
        """Test custom example format"""
        variator = FewShotVariator(
            example_format="Q: {input}\nA: {output}"
        )
        result = variator.generate_prompt("Test", examples=sample_examples[:1])

        assert "Q: What is 2+2?" in result["prompt"]
        assert "A: 4" in result["prompt"]

    def test_with_system_message(self, sample_examples):
        """Test few-shot with system message"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "What is 7+7?",
            examples=sample_examples[:2],
            system_message="You are a math tutor."
        )

        assert "You are a math tutor." in result["prompt"]
        assert result["metadata"]["has_system_message"] is True

    def test_metadata_contains_example_count(self, sample_examples):
        """Test metadata contains correct example counts"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "Test",
            examples=sample_examples,
            num_examples=2
        )

        metadata = result["metadata"]
        assert metadata["num_examples"] == 2
        assert metadata["total_available_examples"] == 3

    def test_zero_num_examples_raises_error(self, sample_examples):
        """Test that num_examples=0 raises ValueError"""
        variator = FewShotVariator()

        with pytest.raises(ValueError, match="Must include at least 1 example"):
            variator.generate_prompt(
                "Test",
                examples=sample_examples,
                num_examples=0
            )

    def test_max_examples_validation(self):
        """Test max_examples validation in config"""
        with pytest.raises(ValueError, match="max_examples must be a positive integer"):
            FewShotVariator(max_examples=0)

        with pytest.raises(ValueError, match="max_examples must be a positive integer"):
            FewShotVariator(max_examples=-1)

    def test_max_examples_upper_limit(self):
        """Test max_examples upper limit warning"""
        with pytest.raises(ValueError, match="should not exceed 5"):
            FewShotVariator(max_examples=10)

    def test_example_order_preserved(self, sample_examples):
        """Test that example order is preserved"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "Test",
            examples=sample_examples
        )

        # Check first example appears before second
        first_pos = result["prompt"].find("What is 2+2?")
        second_pos = result["prompt"].find("What is 3+3?")
        assert first_pos < second_pos

    def test_unicode_in_examples(self):
        """Test examples with unicode characters"""
        variator = FewShotVariator()
        unicode_examples = [
            {"input": "Translate: Hello", "output": "Hola"},
            {"input": "Translate: World", "output": "世界"},
        ]

        result = variator.generate_prompt(
            "Translate: Friend",
            examples=unicode_examples
        )

        assert "世界" in result["prompt"]

    def test_long_examples(self):
        """Test with very long examples"""
        variator = FewShotVariator()
        long_examples = [
            {
                "input": "A" * 1000,
                "output": "B" * 1000
            }
        ]

        result = variator.generate_prompt("Test", examples=long_examples)
        assert len(result["prompt"]) > 2000

    def test_examples_with_special_characters(self):
        """Test examples with special characters"""
        variator = FewShotVariator()
        special_examples = [
            {"input": "Code: print('hello')", "output": "Prints hello"},
            {"input": "Code: x = {1, 2}", "output": "Creates set"},
        ]

        result = variator.generate_prompt("Test", examples=special_examples)
        assert "print('hello')" in result["prompt"]
        assert "{1, 2}" in result["prompt"]

    def test_prompt_structure(self, sample_examples):
        """Test that prompt has correct structure"""
        variator = FewShotVariator()
        result = variator.generate_prompt(
            "What is 7+7?",
            examples=sample_examples[:2]
        )

        prompt = result["prompt"]
        # Should contain examples section header
        assert "Here are some examples:" in prompt
        # Should contain the actual query
        assert "Now, please answer this:" in prompt
        assert "Input: What is 7+7?" in prompt
        assert "Output:" in prompt

    def test_empty_base_prompt_raises_error(self, sample_examples):
        """Test that empty base prompt raises error"""
        variator = FewShotVariator()

        with pytest.raises(ValueError, match="Base prompt cannot be empty"):
            variator.generate_prompt("", examples=sample_examples)

    def test_non_string_input_raises_error(self, sample_examples):
        """Test that non-string input in example raises error"""
        variator = FewShotVariator()
        bad_examples = [{"input": 123, "output": "text"}]

        with pytest.raises(TypeError, match="'input' must be a string"):
            variator.generate_prompt("Test", examples=bad_examples)

    def test_non_string_output_raises_error(self, sample_examples):
        """Test that non-string output in example raises error"""
        variator = FewShotVariator()
        bad_examples = [{"input": "text", "output": 456}]

        with pytest.raises(TypeError, match="'output' must be a string"):
            variator.generate_prompt("Test", examples=bad_examples)
