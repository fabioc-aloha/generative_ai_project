"""
Prompt Chaining Module

This module provides tools for chaining multiple prompts together to create
complex workflows and multi-step reasoning processes. It supports both
sequential and conditional chaining patterns.

Author: Brij Kishore Pandey
"""

from typing import Dict, List, Optional, Any, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import time
from datetime import datetime


logger = logging.getLogger(__name__)


class ChainStepStatus(Enum):
    """Status of a chain step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ChainStepResult:
    """
    Result of executing a chain step.
    
    Attributes:
        step_name (str): Name of the step
        input_data (Any): Input data for the step
        output_data (Any): Output data from the step
        status (ChainStepStatus): Execution status
        execution_time (float): Time taken to execute the step
        error (Optional[str]): Error message if step failed
        metadata (Dict[str, Any]): Additional metadata about the execution
    """
    step_name: str
    input_data: Any
    output_data: Any = None
    status: ChainStepStatus = ChainStepStatus.PENDING
    execution_time: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "step_name": self.step_name,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status.value,
            "execution_time": self.execution_time,
            "error": self.error,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }


class ChainStep(ABC):
    """
    Abstract base class for chain steps.
    
    Each step in a prompt chain should inherit from this class and implement
    the execute method to define its behavior.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        required_inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None
    ):
        """
        Initialize a chain step.
        
        Args:
            name (str): Unique name for the step
            description (str): Description of what the step does
            required_inputs (Optional[List[str]]): List of required input keys
            outputs (Optional[List[str]]): List of output keys this step produces
        """
        self.name = name
        self.description = description
        self.required_inputs = required_inputs or []
        self.outputs = outputs or []
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    @abstractmethod
    def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the step with given input data and context.
        
        Args:
            input_data (Dict[str, Any]): Input data for this step
            context (Dict[str, Any]): Shared context across all steps
            
        Returns:
            Dict[str, Any]: Output data from this step
            
        Raises:
            Exception: If step execution fails
        """
        pass
    
    def validate_inputs(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate that all required inputs are present.
        
        Args:
            input_data (Dict[str, Any]): Input data to validate
            
        Returns:
            bool: True if all required inputs are present
        """
        missing_inputs = [key for key in self.required_inputs if key not in input_data]
        
        if missing_inputs:
            self.logger.error(f"Missing required inputs: {missing_inputs}")
            return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this step.
        
        Returns:
            Dict[str, Any]: Step information
        """
        return {
            "name": self.name,
            "description": self.description,
            "required_inputs": self.required_inputs,
            "outputs": self.outputs,
            "type": self.__class__.__name__
        }


class LLMStep(ChainStep):
    """
    A chain step that uses an LLM for processing.
    
    This step takes a prompt template and uses an LLM client to generate
    a response, which can then be used by subsequent steps.
    """
    
    def __init__(
        self,
        name: str,
        llm_client: Any,  # BaseLLMClient
        prompt_template: str,
        description: str = "",
        required_inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        **llm_kwargs
    ):
        """
        Initialize an LLM step.
        
        Args:
            name (str): Step name
            llm_client: LLM client instance
            prompt_template (str): Template string with {variable} placeholders
            description (str): Step description
            required_inputs (Optional[List[str]]): Required input keys
            outputs (Optional[List[str]]): Output keys
            **llm_kwargs: Additional arguments for LLM calls
        """
        super().__init__(name, description, required_inputs, outputs)
        self.llm_client = llm_client
        self.prompt_template = prompt_template
        self.llm_kwargs = llm_kwargs
    
    def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the LLM step.
        
        Args:
            input_data (Dict[str, Any]): Input data for this step
            context (Dict[str, Any]): Shared context
            
        Returns:
            Dict[str, Any]: Generated response and metadata
        """
        if not self.validate_inputs(input_data):
            raise ValueError(f"Invalid inputs for step '{self.name}'")
        
        try:
            # Format the prompt with input data
            prompt = self.prompt_template.format(**input_data, **context)
            
            # Generate response using LLM
            response = self.llm_client.generate_text(prompt, **self.llm_kwargs)
            
            return {
                "response": response,
                "prompt_used": prompt,
                "model_info": self.llm_client.get_model_info()
            }
            
        except Exception as e:
            self.logger.error(f"LLM step execution failed: {str(e)}")
            raise


class ProcessingStep(ChainStep):
    """
    A chain step that performs data processing using a custom function.
    
    This step allows you to insert custom processing logic between
    LLM calls or to transform data in specific ways.
    """
    
    def __init__(
        self,
        name: str,
        processing_function: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
        description: str = "",
        required_inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None
    ):
        """
        Initialize a processing step.
        
        Args:
            name (str): Step name
            processing_function: Function that takes (input_data, context) and returns output
            description (str): Step description
            required_inputs (Optional[List[str]]): Required input keys
            outputs (Optional[List[str]]): Output keys
        """
        super().__init__(name, description, required_inputs, outputs)
        self.processing_function = processing_function
    
    def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the processing step.
        
        Args:
            input_data (Dict[str, Any]): Input data for this step
            context (Dict[str, Any]): Shared context
            
        Returns:
            Dict[str, Any]: Processed data
        """
        if not self.validate_inputs(input_data):
            raise ValueError(f"Invalid inputs for step '{self.name}'")
        
        try:
            return self.processing_function(input_data, context)
        except Exception as e:
            self.logger.error(f"Processing step execution failed: {str(e)}")
            raise


class ConditionalStep(ChainStep):
    """
    A chain step that executes different sub-steps based on conditions.
    
    This allows for branching logic in prompt chains, where different
    paths can be taken based on previous results or input conditions.
    """
    
    def __init__(
        self,
        name: str,
        condition_function: Callable[[Dict[str, Any], Dict[str, Any]], str],
        step_branches: Dict[str, ChainStep],
        default_branch: Optional[str] = None,
        description: str = "",
        required_inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None
    ):
        """
        Initialize a conditional step.
        
        Args:
            name (str): Step name
            condition_function: Function that returns branch key based on input/context
            step_branches (Dict[str, ChainStep]): Mapping of branch keys to steps
            default_branch (Optional[str]): Default branch if condition returns unknown key
            description (str): Step description
            required_inputs (Optional[List[str]]): Required input keys
            outputs (Optional[List[str]]): Output keys
        """
        super().__init__(name, description, required_inputs, outputs)
        self.condition_function = condition_function
        self.step_branches = step_branches
        self.default_branch = default_branch
    
    def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the conditional step.
        
        Args:
            input_data (Dict[str, Any]): Input data for this step
            context (Dict[str, Any]): Shared context
            
        Returns:
            Dict[str, Any]: Output from the selected branch
        """
        if not self.validate_inputs(input_data):
            raise ValueError(f"Invalid inputs for step '{self.name}'")
        
        try:
            # Determine which branch to take
            branch_key = self.condition_function(input_data, context)
            
            if branch_key not in self.step_branches:
                if self.default_branch and self.default_branch in self.step_branches:
                    branch_key = self.default_branch
                    self.logger.warning(f"Unknown branch '{branch_key}', using default")
                else:
                    raise ValueError(f"Unknown branch '{branch_key}' and no default specified")
            
            # Execute the selected branch
            selected_step = self.step_branches[branch_key]
            result = selected_step.execute(input_data, context)
            
            # Add metadata about which branch was taken
            result["branch_taken"] = branch_key
            result["step_executed"] = selected_step.name
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conditional step execution failed: {str(e)}")
            raise


class PromptChain:
    """
    A chain of prompt steps that can be executed sequentially or with branching logic.
    
    This class manages the execution of multiple steps, handles data flow between
    steps, and provides monitoring and debugging capabilities.
    
    Example:
        >>> chain = PromptChain("analysis_chain")
        >>> chain.add_step(LLMStep("extract", llm_client, "Extract key points: {text}"))
        >>> chain.add_step(LLMStep("summarize", llm_client, "Summarize: {response}"))
        >>> result = chain.execute({"text": "Long document content..."})
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        fail_fast: bool = True,
        save_intermediate: bool = True
    ):
        """
        Initialize a prompt chain.
        
        Args:
            name (str): Chain name
            description (str): Chain description
            fail_fast (bool): Whether to stop on first error
            save_intermediate (bool): Whether to save intermediate results
        """
        self.name = name
        self.description = description
        self.fail_fast = fail_fast
        self.save_intermediate = save_intermediate
        
        self.steps: List[ChainStep] = []
        self.step_results: List[ChainStepResult] = []
        self.context: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{name}")
    
    def add_step(self, step: ChainStep) -> 'PromptChain':
        """
        Add a step to the chain.
        
        Args:
            step (ChainStep): Step to add
            
        Returns:
            PromptChain: Self for method chaining
        """
        self.steps.append(step)
        self.logger.debug(f"Added step '{step.name}' to chain")
        return self
    
    def remove_step(self, step_name: str) -> 'PromptChain':
        """
        Remove a step from the chain.
        
        Args:
            step_name (str): Name of the step to remove
            
        Returns:
            PromptChain: Self for method chaining
        """
        self.steps = [step for step in self.steps if step.name != step_name]
        self.logger.debug(f"Removed step '{step_name}' from chain")
        return self
    
    def get_step(self, step_name: str) -> Optional[ChainStep]:
        """
        Get a step by name.
        
        Args:
            step_name (str): Name of the step
            
        Returns:
            Optional[ChainStep]: The step if found, None otherwise
        """
        for step in self.steps:
            if step.name == step_name:
                return step
        return None
    
    def execute(
        self,
        initial_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the prompt chain.
        
        Args:
            initial_data (Dict[str, Any]): Initial input data
            context (Optional[Dict[str, Any]]): Additional context
            
        Returns:
            Dict[str, Any]: Final execution results and metadata
            
        Raises:
            Exception: If execution fails and fail_fast is True
        """
        if not self.steps:
            raise ValueError("No steps defined in the chain")
        
        # Initialize context
        self.context = context or {}
        self.context.update(initial_data)
        self.step_results = []
        
        current_data = initial_data.copy()
        
        self.logger.info(f"Starting execution of chain '{self.name}' with {len(self.steps)} steps")
        
        # Execute each step
        for i, step in enumerate(self.steps):
            step_result = ChainStepResult(
                step_name=step.name,
                input_data=current_data.copy()
            )
            
            try:
                step_result.status = ChainStepStatus.RUNNING
                start_time = time.time()
                
                self.logger.info(f"Executing step {i+1}/{len(self.steps)}: '{step.name}'")
                
                # Execute the step
                output_data = step.execute(current_data, self.context)
                
                step_result.execution_time = time.time() - start_time
                step_result.output_data = output_data
                step_result.status = ChainStepStatus.COMPLETED
                
                # Update current data with step output
                if isinstance(output_data, dict):
                    current_data.update(output_data)
                else:
                    current_data[f"step_{i}_output"] = output_data
                
                # Update context if saving intermediate results
                if self.save_intermediate:
                    self.context[f"step_{i}_result"] = output_data
                
                self.logger.info(f"Step '{step.name}' completed in {step_result.execution_time:.2f}s")
                
            except Exception as e:
                step_result.execution_time = time.time() - start_time
                step_result.error = str(e)
                step_result.status = ChainStepStatus.FAILED
                
                self.logger.error(f"Step '{step.name}' failed: {str(e)}")
                
                if self.fail_fast:
                    self.step_results.append(step_result)
                    raise Exception(f"Chain execution failed at step '{step.name}': {str(e)}")
                
                # Continue with next step if not fail_fast
                current_data[f"step_{i}_error"] = str(e)
            
            self.step_results.append(step_result)
        
        # Prepare final results
        total_time = sum(result.execution_time for result in self.step_results)
        successful_steps = sum(1 for result in self.step_results if result.status == ChainStepStatus.COMPLETED)
        
        final_result = {
            "chain_name": self.name,
            "final_data": current_data,
            "execution_summary": {
                "total_steps": len(self.steps),
                "successful_steps": successful_steps,
                "failed_steps": len(self.steps) - successful_steps,
                "total_time": total_time,
                "success_rate": successful_steps / len(self.steps) if self.steps else 0
            },
            "step_results": [result.to_dict() for result in self.step_results] if self.save_intermediate else [],
            "context": self.context if self.save_intermediate else {}
        }
        
        self.logger.info(f"Chain execution completed: {successful_steps}/{len(self.steps)} steps successful")
        
        return final_result
    
    def get_chain_info(self) -> Dict[str, Any]:
        """
        Get information about the chain.
        
        Returns:
            Dict[str, Any]: Chain information
        """
        return {
            "name": self.name,
            "description": self.description,
            "steps": [step.get_info() for step in self.steps],
            "step_count": len(self.steps),
            "fail_fast": self.fail_fast,
            "save_intermediate": self.save_intermediate
        }
    
    def visualize_chain(self) -> str:
        """
        Create a text visualization of the chain.
        
        Returns:
            str: Text representation of the chain flow
        """
        if not self.steps:
            return f"Chain '{self.name}': No steps defined"
        
        lines = [f"Chain '{self.name}':", "=" * (len(self.name) + 8)]
        
        for i, step in enumerate(self.steps):
            step_info = f"{i+1}. {step.name} ({step.__class__.__name__})"
            if step.description:
                step_info += f" - {step.description}"
            lines.append(step_info)
            
            if step.required_inputs:
                lines.append(f"   Inputs: {', '.join(step.required_inputs)}")
            if step.outputs:
                lines.append(f"   Outputs: {', '.join(step.outputs)}")
            
            if i < len(self.steps) - 1:
                lines.append("   ↓")
        
        return "\n".join(lines)
    
    def save_chain_config(self, file_path: str) -> None:
        """
        Save chain configuration to a JSON file.
        
        Args:
            file_path (str): Path to save the configuration
        """
        config = self.get_chain_info()
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved chain configuration to {file_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save chain configuration: {str(e)}")


class ChainBuilder:
    """
    Builder class for creating prompt chains with a fluent interface.
    
    This class provides a convenient way to build complex prompt chains
    with various types of steps and configurations.
    
    Example:
        >>> builder = ChainBuilder("my_chain")
        >>> chain = (builder
        ...     .add_llm_step("analyze", llm_client, "Analyze: {text}")
        ...     .add_processing_step("format", lambda data, ctx: {"formatted": data["response"].upper()})
        ...     .build())
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize the builder.
        
        Args:
            name (str): Chain name
            description (str): Chain description
        """
        self.name = name
        self.description = description
        self.steps: List[ChainStep] = []
        self.fail_fast = True
        self.save_intermediate = True
    
    def add_llm_step(
        self,
        name: str,
        llm_client: Any,
        prompt_template: str,
        description: str = "",
        required_inputs: Optional[List[str]] = None,
        **llm_kwargs
    ) -> 'ChainBuilder':
        """
        Add an LLM step to the chain.
        
        Args:
            name (str): Step name
            llm_client: LLM client instance
            prompt_template (str): Prompt template
            description (str): Step description
            required_inputs (Optional[List[str]]): Required inputs
            **llm_kwargs: Additional LLM arguments
            
        Returns:
            ChainBuilder: Self for method chaining
        """
        step = LLMStep(
            name=name,
            llm_client=llm_client,
            prompt_template=prompt_template,
            description=description,
            required_inputs=required_inputs,
            **llm_kwargs
        )
        self.steps.append(step)
        return self
    
    def add_processing_step(
        self,
        name: str,
        processing_function: Callable,
        description: str = "",
        required_inputs: Optional[List[str]] = None
    ) -> 'ChainBuilder':
        """
        Add a processing step to the chain.
        
        Args:
            name (str): Step name
            processing_function: Processing function
            description (str): Step description
            required_inputs (Optional[List[str]]): Required inputs
            
        Returns:
            ChainBuilder: Self for method chaining
        """
        step = ProcessingStep(
            name=name,
            processing_function=processing_function,
            description=description,
            required_inputs=required_inputs
        )
        self.steps.append(step)
        return self
    
    def add_conditional_step(
        self,
        name: str,
        condition_function: Callable,
        step_branches: Dict[str, ChainStep],
        default_branch: Optional[str] = None,
        description: str = ""
    ) -> 'ChainBuilder':
        """
        Add a conditional step to the chain.
        
        Args:
            name (str): Step name
            condition_function: Condition function
            step_branches (Dict[str, ChainStep]): Branch mapping
            default_branch (Optional[str]): Default branch
            description (str): Step description
            
        Returns:
            ChainBuilder: Self for method chaining
        """
        step = ConditionalStep(
            name=name,
            condition_function=condition_function,
            step_branches=step_branches,
            default_branch=default_branch,
            description=description
        )
        self.steps.append(step)
        return self
    
    def set_fail_fast(self, fail_fast: bool) -> 'ChainBuilder':
        """
        Set fail fast behavior.
        
        Args:
            fail_fast (bool): Whether to fail fast
            
        Returns:
            ChainBuilder: Self for method chaining
        """
        self.fail_fast = fail_fast
        return self
    
    def set_save_intermediate(self, save_intermediate: bool) -> 'ChainBuilder':
        """
        Set whether to save intermediate results.
        
        Args:
            save_intermediate (bool): Whether to save intermediate results
            
        Returns:
            ChainBuilder: Self for method chaining
        """
        self.save_intermediate = save_intermediate
        return self
    
    def build(self) -> PromptChain:
        """
        Build the prompt chain.
        
        Returns:
            PromptChain: Constructed prompt chain
        """
        chain = PromptChain(
            name=self.name,
            description=self.description,
            fail_fast=self.fail_fast,
            save_intermediate=self.save_intermediate
        )
        
        for step in self.steps:
            chain.add_step(step)
        
        return chain


# Common chain patterns
def create_analysis_chain(
    llm_client: Any,
    chain_name: str = "analysis_chain"
) -> PromptChain:
    """
    Create a common analysis chain pattern.
    
    Args:
        llm_client: LLM client to use
        chain_name (str): Name for the chain
        
    Returns:
        PromptChain: Pre-configured analysis chain
    """
    builder = ChainBuilder(chain_name, "Common analysis workflow")
    
    return (builder
        .add_llm_step(
            "extract_key_points",
            llm_client,
            "Extract the key points from this text:\n\n{text}\n\nKey points:",
            "Extract main points from input text"
        )
        .add_llm_step(
            "analyze_sentiment",
            llm_client,
            "Analyze the sentiment of these key points:\n\n{response}\n\nSentiment analysis:",
            "Analyze sentiment of extracted points"
        )
        .add_llm_step(
            "generate_summary",
            llm_client,
            "Create a brief summary based on this analysis:\n\nKey Points: {step_0_result}\n\nSentiment: {response}\n\nSummary:",
            "Generate final summary"
        )
        .build())


def create_qa_chain(
    llm_client: Any,
    chain_name: str = "qa_chain"
) -> PromptChain:
    """
    Create a question-answering chain pattern.
    
    Args:
        llm_client: LLM client to use
        chain_name (str): Name for the chain
        
    Returns:
        PromptChain: Pre-configured QA chain
    """
    builder = ChainBuilder(chain_name, "Question answering workflow")
    
    return (builder
        .add_llm_step(
            "understand_question",
            llm_client,
            "Analyze this question and identify what information is needed:\n\nQuestion: {question}\n\nAnalysis:",
            "Understand the question requirements"
        )
        .add_llm_step(
            "extract_relevant_info",
            llm_client,
            "From this context, extract information relevant to the question:\n\nContext: {context}\n\nQuestion Analysis: {response}\n\nRelevant Information:",
            "Extract relevant information from context"
        )
        .add_llm_step(
            "formulate_answer",
            llm_client,
            "Based on this information, answer the original question:\n\nOriginal Question: {question}\n\nRelevant Information: {response}\n\nAnswer:",
            "Formulate the final answer"
        )
        .build())