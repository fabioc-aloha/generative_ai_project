"""
Prompt Engineering Module

This module provides comprehensive tools for prompt engineering, including
template management, few-shot learning, and prompt chaining capabilities.

Author: Brij Kishore Pandey
"""

from .templates import (
    PromptTemplate,
    TemplateManager,
    COMMON_TEMPLATES,
    get_common_template,
    create_template_manager_with_common
)

from .few_shot import (
    Example,
    FewShotPrompt,
    FewShotBuilder,
    COMMON_FEW_SHOT_TASKS,
    create_few_shot_for_task,
    analyze_examples
)

from .chain import (
    ChainStep,
    ChainStepResult,
    ChainStepStatus,
    LLMStep,
    ProcessingStep,
    ConditionalStep,
    PromptChain,
    ChainBuilder,
    create_analysis_chain,
    create_qa_chain
)

__all__ = [
    # Template management
    "PromptTemplate",
    "TemplateManager",
    "COMMON_TEMPLATES",
    "get_common_template",
    "create_template_manager_with_common",
    
    # Few-shot prompting
    "Example",
    "FewShotPrompt",
    "FewShotBuilder",
    "COMMON_FEW_SHOT_TASKS",
    "create_few_shot_for_task",
    "analyze_examples",
    
    # Prompt chaining
    "ChainStep",
    "ChainStepResult",
    "ChainStepStatus",
    "LLMStep",
    "ProcessingStep",
    "ConditionalStep",
    "PromptChain",
    "ChainBuilder",
    "create_analysis_chain",
    "create_qa_chain",
]