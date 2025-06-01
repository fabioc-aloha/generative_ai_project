"""
Few-Shot Prompting Module

This module provides utilities for creating and managing few-shot prompts,
which help improve LLM performance by providing examples of the desired
input-output behavior.

Author: Brij Kishore Pandey
"""

from typing import Dict, List, Optional, Any, Union
import json
import random
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class Example:
    """
    Represents a single example for few-shot prompting.
    
    Attributes:
        input (str): The input text for the example
        output (str): The expected output for the example
        explanation (Optional[str]): Optional explanation of the reasoning
        metadata (Dict[str, Any]): Additional metadata about the example
    """
    input: str
    output: str
    explanation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert example to dictionary format.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the example
        """
        result = {
            "input": self.input,
            "output": self.output
        }
        
        if self.explanation:
            result["explanation"] = self.explanation
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Example':
        """
        Create an Example from dictionary data.
        
        Args:
            data (Dict[str, Any]): Dictionary containing example data
            
        Returns:
            Example: Created example instance
        """
        return cls(
            input=data["input"],
            output=data["output"],
            explanation=data.get("explanation"),
            metadata=data.get("metadata", {})
        )
    
    def format(self, input_label: str = "Input", output_label: str = "Output") -> str:
        """
        Format the example as a string.
        
        Args:
            input_label (str): Label for the input part
            output_label (str): Label for the output part
            
        Returns:
            str: Formatted example string
        """
        formatted = f"{input_label}: {self.input}\n{output_label}: {self.output}"
        
        if self.explanation:
            formatted += f"\nExplanation: {self.explanation}"
        
        return formatted


class FewShotPrompt:
    """
    A few-shot prompt that combines examples with a template to guide LLM behavior.
    
    This class manages a collection of examples and provides methods to format
    them into effective few-shot prompts for various tasks.
    
    Example:
        >>> examples = [
        ...     Example("What is 2+2?", "4"),
        ...     Example("What is 5*3?", "15")
        ... ]
        >>> prompt = FewShotPrompt(
        ...     examples=examples,
        ...     template="Solve the math problem:\n\n{examples}\n\nProblem: {input}\nAnswer:"
        ... )
        >>> result = prompt.format(input="What is 7+8?")
    """
    
    def __init__(
        self,
        examples: List[Example],
        template: str = "{examples}\n\n{input}",
        example_separator: str = "\n\n",
        input_label: str = "Input",
        output_label: str = "Output",
        max_examples: Optional[int] = None,
        selection_strategy: str = "first"
    ):
        """
        Initialize a few-shot prompt.
        
        Args:
            examples (List[Example]): List of examples to use
            template (str): Template string with {examples} and {input} placeholders
            example_separator (str): Separator between examples
            input_label (str): Label for input in examples
            output_label (str): Label for output in examples
            max_examples (Optional[int]): Maximum number of examples to use
            selection_strategy (str): Strategy for selecting examples ("first", "random", "similar")
        """
        self.examples = examples
        self.template = template
        self.example_separator = example_separator
        self.input_label = input_label
        self.output_label = output_label
        self.max_examples = max_examples
        self.selection_strategy = selection_strategy
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def add_example(self, example: Example) -> None:
        """
        Add an example to the prompt.
        
        Args:
            example (Example): Example to add
        """
        self.examples.append(example)
        self.logger.debug(f"Added example: {example.input[:50]}...")
    
    def remove_example(self, index: int) -> None:
        """
        Remove an example by index.
        
        Args:
            index (int): Index of the example to remove
            
        Raises:
            IndexError: If index is out of range
        """
        if 0 <= index < len(self.examples):
            removed = self.examples.pop(index)
            self.logger.debug(f"Removed example at index {index}: {removed.input[:50]}...")
        else:
            raise IndexError(f"Example index {index} out of range")
    
    def select_examples(
        self,
        input_text: Optional[str] = None,
        count: Optional[int] = None
    ) -> List[Example]:
        """
        Select examples based on the configured strategy.
        
        Args:
            input_text (Optional[str]): Input text for similarity-based selection
            count (Optional[int]): Number of examples to select
            
        Returns:
            List[Example]: Selected examples
        """
        if not self.examples:
            return []
        
        count = count or self.max_examples or len(self.examples)
        count = min(count, len(self.examples))
        
        if self.selection_strategy == "first":
            return self.examples[:count]
        
        elif self.selection_strategy == "random":
            return random.sample(self.examples, count)
        
        elif self.selection_strategy == "similar":
            if input_text is None:
                self.logger.warning("Similar selection requires input_text, falling back to first")
                return self.examples[:count]
            
            # Simple similarity based on word overlap (could be enhanced with embeddings)
            scored_examples = []
            input_words = set(input_text.lower().split())
            
            for example in self.examples:
                example_words = set(example.input.lower().split())
                similarity = len(input_words & example_words) / len(input_words | example_words)
                scored_examples.append((similarity, example))
            
            # Sort by similarity (descending) and take top examples
            scored_examples.sort(key=lambda x: x[0], reverse=True)
            return [example for _, example in scored_examples[:count]]
        
        else:
            self.logger.warning(f"Unknown selection strategy: {self.selection_strategy}")
            return self.examples[:count]
    
    def format_examples(
        self,
        examples: Optional[List[Example]] = None,
        input_text: Optional[str] = None
    ) -> str:
        """
        Format examples into a string.
        
        Args:
            examples (Optional[List[Example]]): Examples to format (default: all)
            input_text (Optional[str]): Input text for example selection
            
        Returns:
            str: Formatted examples string
        """
        if examples is None:
            examples = self.select_examples(input_text)
        
        if not examples:
            return ""
        
        formatted_examples = []
        for example in examples:
            formatted = example.format(self.input_label, self.output_label)
            formatted_examples.append(formatted)
        
        return self.example_separator.join(formatted_examples)
    
    def format(self, input_text: str, **kwargs) -> str:
        """
        Format the complete few-shot prompt.
        
        Args:
            input_text (str): The input for which to generate a prompt
            **kwargs: Additional variables for template formatting
            
        Returns:
            str: Complete formatted prompt
        """
        examples = self.select_examples(input_text)
        formatted_examples = self.format_examples(examples, input_text)
        
        # Prepare template variables
        template_vars = {
            "examples": formatted_examples,
            "input": input_text,
            **kwargs
        }
        
        try:
            return self.template.format(**template_vars)
        except KeyError as e:
            self.logger.error(f"Missing template variable: {e}")
            raise ValueError(f"Missing template variable: {e}")
    
    def get_example_count(self) -> int:
        """
        Get the number of examples.
        
        Returns:
            int: Number of examples
        """
        return len(self.examples)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the few-shot prompt.
        
        Returns:
            Dict[str, Any]: Statistics including example count and lengths
        """
        if not self.examples:
            return {
                "example_count": 0,
                "avg_input_length": 0,
                "avg_output_length": 0,
                "total_length": 0
            }
        
        input_lengths = [len(ex.input) for ex in self.examples]
        output_lengths = [len(ex.output) for ex in self.examples]
        
        return {
            "example_count": len(self.examples),
            "avg_input_length": sum(input_lengths) / len(input_lengths),
            "avg_output_length": sum(output_lengths) / len(output_lengths),
            "total_length": sum(input_lengths) + sum(output_lengths),
            "has_explanations": sum(1 for ex in self.examples if ex.explanation) > 0
        }
    
    def save_examples(self, file_path: str) -> None:
        """
        Save examples to a JSON file.
        
        Args:
            file_path (str): Path to save the examples
        """
        data = {
            "examples": [ex.to_dict() for ex in self.examples],
            "config": {
                "template": self.template,
                "example_separator": self.example_separator,
                "input_label": self.input_label,
                "output_label": self.output_label,
                "max_examples": self.max_examples,
                "selection_strategy": self.selection_strategy
            }
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(self.examples)} examples to {file_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save examples: {str(e)}")
    
    @classmethod
    def load_examples(cls, file_path: str) -> 'FewShotPrompt':
        """
        Load examples from a JSON file.
        
        Args:
            file_path (str): Path to load the examples from
            
        Returns:
            FewShotPrompt: Loaded few-shot prompt
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            examples = [Example.from_dict(ex_data) for ex_data in data["examples"]]
            config = data.get("config", {})
            
            return cls(examples=examples, **config)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Examples file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Failed to load examples: {str(e)}")


class FewShotBuilder:
    """
    Builder class for creating few-shot prompts with different configurations.
    
    This class provides a fluent interface for building few-shot prompts
    with various options and configurations.
    
    Example:
        >>> builder = FewShotBuilder()
        >>> prompt = (builder
        ...     .add_example("2+2", "4")
        ...     .add_example("3+5", "8")
        ...     .set_template("Solve: {examples}\n\nProblem: {input}\nAnswer:")
        ...     .build())
    """
    
    def __init__(self):
        """Initialize the builder."""
        self.examples: List[Example] = []
        self.template: str = "{examples}\n\n{input}"
        self.example_separator: str = "\n\n"
        self.input_label: str = "Input"
        self.output_label: str = "Output"
        self.max_examples: Optional[int] = None
        self.selection_strategy: str = "first"
    
    def add_example(
        self,
        input_text: str,
        output_text: str,
        explanation: Optional[str] = None,
        **metadata
    ) -> 'FewShotBuilder':
        """
        Add an example to the builder.
        
        Args:
            input_text (str): Input for the example
            output_text (str): Output for the example
            explanation (Optional[str]): Optional explanation
            **metadata: Additional metadata
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        example = Example(
            input=input_text,
            output=output_text,
            explanation=explanation,
            metadata=metadata
        )
        self.examples.append(example)
        return self
    
    def add_examples_from_data(self, data: List[Dict[str, Any]]) -> 'FewShotBuilder':
        """
        Add multiple examples from data.
        
        Args:
            data (List[Dict[str, Any]]): List of example dictionaries
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        for item in data:
            example = Example.from_dict(item)
            self.examples.append(example)
        return self
    
    def set_template(self, template: str) -> 'FewShotBuilder':
        """
        Set the prompt template.
        
        Args:
            template (str): Template string
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        self.template = template
        return self
    
    def set_separator(self, separator: str) -> 'FewShotBuilder':
        """
        Set the example separator.
        
        Args:
            separator (str): Separator string
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        self.example_separator = separator
        return self
    
    def set_labels(self, input_label: str, output_label: str) -> 'FewShotBuilder':
        """
        Set input and output labels.
        
        Args:
            input_label (str): Label for inputs
            output_label (str): Label for outputs
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        self.input_label = input_label
        self.output_label = output_label
        return self
    
    def set_max_examples(self, max_examples: int) -> 'FewShotBuilder':
        """
        Set maximum number of examples to use.
        
        Args:
            max_examples (int): Maximum number of examples
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        self.max_examples = max_examples
        return self
    
    def set_selection_strategy(self, strategy: str) -> 'FewShotBuilder':
        """
        Set the example selection strategy.
        
        Args:
            strategy (str): Selection strategy ("first", "random", "similar")
            
        Returns:
            FewShotBuilder: Self for method chaining
        """
        self.selection_strategy = strategy
        return self
    
    def build(self) -> FewShotPrompt:
        """
        Build the few-shot prompt.
        
        Returns:
            FewShotPrompt: Constructed few-shot prompt
        """
        return FewShotPrompt(
            examples=self.examples,
            template=self.template,
            example_separator=self.example_separator,
            input_label=self.input_label,
            output_label=self.output_label,
            max_examples=self.max_examples,
            selection_strategy=self.selection_strategy
        )


# Predefined few-shot prompt templates for common tasks
COMMON_FEW_SHOT_TASKS = {
    "classification": {
        "template": "Classify the following text:\n\n{examples}\n\nText: {input}\nClass:",
        "examples": [
            Example("I love this movie!", "positive"),
            Example("This film is terrible.", "negative"),
            Example("The movie was okay.", "neutral")
        ]
    },
    
    "translation": {
        "template": "Translate from English to French:\n\n{examples}\n\nEnglish: {input}\nFrench:",
        "examples": [
            Example("Hello", "Bonjour"),
            Example("How are you?", "Comment allez-vous?"),
            Example("Thank you", "Merci")
        ]
    },
    
    "math_word_problems": {
        "template": "Solve the math word problem:\n\n{examples}\n\nProblem: {input}\nSolution:",
        "examples": [
            Example(
                "John has 5 apples. He gives 2 to Mary. How many does he have left?",
                "5 - 2 = 3 apples",
                "Subtract the given amount from the total"
            ),
            Example(
                "A box contains 12 chocolates. If 4 people share them equally, how many does each get?",
                "12 ÷ 4 = 3 chocolates each",
                "Divide the total by the number of people"
            )
        ]
    },
    
    "code_generation": {
        "template": "Generate Python code for the given task:\n\n{examples}\n\nTask: {input}\nCode:",
        "examples": [
            Example(
                "Create a function that adds two numbers",
                "def add_numbers(a, b):\n    return a + b"
            ),
            Example(
                "Create a function that finds the maximum in a list",
                "def find_max(numbers):\n    return max(numbers)"
            )
        ]
    }
}


def create_few_shot_for_task(task: str) -> FewShotPrompt:
    """
    Create a few-shot prompt for a common task.
    
    Args:
        task (str): Name of the task
        
    Returns:
        FewShotPrompt: Pre-configured few-shot prompt
        
    Raises:
        KeyError: If task is not found
    """
    if task not in COMMON_FEW_SHOT_TASKS:
        available = list(COMMON_FEW_SHOT_TASKS.keys())
        raise KeyError(f"Task '{task}' not found. Available: {available}")
    
    config = COMMON_FEW_SHOT_TASKS[task]
    return FewShotPrompt(
        examples=config["examples"],
        template=config["template"]
    )


def analyze_examples(examples: List[Example]) -> Dict[str, Any]:
    """
    Analyze a list of examples to provide insights.
    
    Args:
        examples (List[Example]): Examples to analyze
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    if not examples:
        return {"error": "No examples provided"}
    
    input_lengths = [len(ex.input) for ex in examples]
    output_lengths = [len(ex.output) for ex in examples]
    
    # Check for patterns in outputs
    output_set = set(ex.output for ex in examples)
    
    # Basic vocabulary analysis
    all_input_words = []
    all_output_words = []
    
    for ex in examples:
        all_input_words.extend(ex.input.lower().split())
        all_output_words.extend(ex.output.lower().split())
    
    return {
        "count": len(examples),
        "input_stats": {
            "min_length": min(input_lengths),
            "max_length": max(input_lengths),
            "avg_length": sum(input_lengths) / len(input_lengths),
            "unique_words": len(set(all_input_words))
        },
        "output_stats": {
            "min_length": min(output_lengths),
            "max_length": max(output_lengths),
            "avg_length": sum(output_lengths) / len(output_lengths),
            "unique_outputs": len(output_set),
            "unique_words": len(set(all_output_words))
        },
        "has_explanations": sum(1 for ex in examples if ex.explanation) > 0,
        "explanation_ratio": sum(1 for ex in examples if ex.explanation) / len(examples)
    }