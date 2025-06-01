"""
Prompt Template Management Module

This module provides tools for managing, loading, and rendering prompt templates.
It supports variable substitution, template inheritance, and dynamic content generation.

Author: Brij Kishore Pandey
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Any, Union
from string import Template
from pathlib import Path
import logging


logger = logging.getLogger(__name__)


class PromptTemplate:
    """
    A prompt template with variable substitution capabilities.
    
    This class allows you to create reusable prompt templates with placeholders
    that can be filled with specific values at runtime.
    
    Example:
        >>> template = PromptTemplate(
        ...     "Hello {name}, please {action} the following: {content}"
        ... )
        >>> prompt = template.render(
        ...     name="Alice",
        ...     action="analyze",
        ...     content="market data"
        ... )
        >>> print(prompt)
        Hello Alice, please analyze the following: market data
    """
    
    def __init__(
        self,
        template: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        variables: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a prompt template.
        
        Args:
            template (str): Template string with {variable} placeholders
            name (Optional[str]): Name of the template
            description (Optional[str]): Description of the template's purpose
            variables (Optional[List[str]]): List of expected variable names
            metadata (Optional[Dict[str, Any]]): Additional metadata
        """
        self.template = template
        self.name = name or "unnamed_template"
        self.description = description or ""
        self.variables = variables or self._extract_variables()
        self.metadata = metadata or {}
        
        # Create Python Template object for substitution
        self._template_obj = Template(template)
    
    def _extract_variables(self) -> List[str]:
        """
        Extract variable names from the template string.
        
        Returns:
            List[str]: List of variable names found in the template
        """
        # Use Template to find variables
        template_obj = Template(self.template)
        
        # Get identifiers from the template
        variables = []
        try:
            # This will raise KeyError for missing variables, helping us find them
            template_obj.safe_substitute({})
        except (KeyError, ValueError):
            pass
        
        # Extract variables using regex (simpler approach)
        import re
        pattern = r'\{([^}]+)\}'
        variables = re.findall(pattern, self.template)
        
        return list(set(variables))  # Remove duplicates
    
    def render(self, **kwargs) -> str:
        """
        Render the template with provided variables.
        
        Args:
            **kwargs: Variable values to substitute in the template
            
        Returns:
            str: Rendered prompt with variables substituted
            
        Raises:
            KeyError: If required variables are missing
            ValueError: If template rendering fails
        """
        try:
            # Check for missing required variables
            missing_vars = set(self.variables) - set(kwargs.keys())
            if missing_vars:
                logger.warning(f"Missing variables in template '{self.name}': {missing_vars}")
            
            # Render using string format
            rendered = self.template.format(**kwargs)
            
            logger.debug(f"Template '{self.name}' rendered successfully")
            return rendered
            
        except KeyError as e:
            raise KeyError(f"Missing required variable: {e}")
        except Exception as e:
            raise ValueError(f"Template rendering failed: {str(e)}")
    
    def safe_render(self, **kwargs) -> str:
        """
        Safely render the template, leaving unmatched variables as placeholders.
        
        Args:
            **kwargs: Variable values to substitute in the template
            
        Returns:
            str: Rendered prompt with available variables substituted
        """
        try:
            # Use safe_substitute to leave missing variables as-is
            rendered = self._template_obj.safe_substitute(kwargs)
            return rendered
        except Exception as e:
            logger.error(f"Safe template rendering failed: {str(e)}")
            return self.template
    
    def validate_variables(self, variables: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that all required variables are provided.
        
        Args:
            variables (Dict[str, Any]): Variables to validate
            
        Returns:
            Dict[str, Any]: Validation result with missing and extra variables
        """
        provided_vars = set(variables.keys())
        required_vars = set(self.variables)
        
        missing = required_vars - provided_vars
        extra = provided_vars - required_vars
        
        return {
            "valid": len(missing) == 0,
            "missing": list(missing),
            "extra": list(extra),
            "required": list(required_vars)
        }
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the template.
        
        Returns:
            Dict[str, Any]: Template information
        """
        return {
            "name": self.name,
            "description": self.description,
            "variables": self.variables,
            "template_length": len(self.template),
            "metadata": self.metadata
        }
    
    def copy(self, **override_params) -> 'PromptTemplate':
        """
        Create a copy of the template with optional parameter overrides.
        
        Args:
            **override_params: Parameters to override in the copy
            
        Returns:
            PromptTemplate: New template instance
        """
        params = {
            "template": self.template,
            "name": self.name,
            "description": self.description,
            "variables": self.variables.copy(),
            "metadata": self.metadata.copy()
        }
        params.update(override_params)
        
        return PromptTemplate(**params)
    
    def __str__(self) -> str:
        """String representation of the template."""
        return f"PromptTemplate(name='{self.name}', variables={self.variables})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the template."""
        return (
            f"PromptTemplate(name='{self.name}', "
            f"description='{self.description[:50]}...', "
            f"variables={self.variables})"
        )


class TemplateManager:
    """
    Manager for loading, storing, and organizing prompt templates.
    
    This class provides a centralized way to manage multiple prompt templates,
    load them from files, and organize them by categories or use cases.
    
    Example:
        >>> manager = TemplateManager()
        >>> manager.load_from_file("templates.yaml")
        >>> template = manager.get_template("greeting")
        >>> prompt = template.render(name="World")
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the template manager.
        
        Args:
            templates_dir (Optional[str]): Directory containing template files
        """
        self.templates: Dict[str, PromptTemplate] = {}
        self.categories: Dict[str, List[str]] = {}
        self.templates_dir = templates_dir
        
        if templates_dir and os.path.exists(templates_dir):
            self.load_directory(templates_dir)
    
    def add_template(
        self,
        template: PromptTemplate,
        category: Optional[str] = None
    ) -> None:
        """
        Add a template to the manager.
        
        Args:
            template (PromptTemplate): Template to add
            category (Optional[str]): Category to organize the template under
        """
        self.templates[template.name] = template
        
        if category:
            if category not in self.categories:
                self.categories[category] = []
            
            if template.name not in self.categories[category]:
                self.categories[category].append(template.name)
        
        logger.info(f"Added template '{template.name}' to manager")
    
    def get_template(self, name: str) -> PromptTemplate:
        """
        Get a template by name.
        
        Args:
            name (str): Name of the template
            
        Returns:
            PromptTemplate: The requested template
            
        Raises:
            KeyError: If template is not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        
        return self.templates[name]
    
    def remove_template(self, name: str) -> None:
        """
        Remove a template from the manager.
        
        Args:
            name (str): Name of the template to remove
            
        Raises:
            KeyError: If template is not found
        """
        if name not in self.templates:
            raise KeyError(f"Template '{name}' not found")
        
        del self.templates[name]
        
        # Remove from categories
        for category, template_names in self.categories.items():
            if name in template_names:
                template_names.remove(name)
        
        logger.info(f"Removed template '{name}' from manager")
    
    def list_templates(self, category: Optional[str] = None) -> List[str]:
        """
        List available templates.
        
        Args:
            category (Optional[str]): Filter by category
            
        Returns:
            List[str]: List of template names
        """
        if category:
            return self.categories.get(category, [])
        
        return list(self.templates.keys())
    
    def list_categories(self) -> List[str]:
        """
        List available categories.
        
        Returns:
            List[str]: List of category names
        """
        return list(self.categories.keys())
    
    def load_from_file(self, file_path: str) -> None:
        """
        Load templates from a YAML or JSON file.
        
        Args:
            file_path (str): Path to the template file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Template file not found: {file_path}")
        
        file_extension = Path(file_path).suffix.lower()
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_extension in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                elif file_extension == '.json':
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported file format: {file_extension}")
            
            self._load_templates_from_data(data)
            logger.info(f"Loaded templates from {file_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to load templates from {file_path}: {str(e)}")
    
    def _load_templates_from_data(self, data: Dict[str, Any]) -> None:
        """
        Load templates from parsed data structure.
        
        Args:
            data (Dict[str, Any]): Parsed template data
        """
        templates = data.get('templates', {})
        categories = data.get('categories', {})
        
        # Load individual templates
        for name, template_data in templates.items():
            if isinstance(template_data, str):
                # Simple string template
                template = PromptTemplate(template_data, name=name)
            elif isinstance(template_data, dict):
                # Detailed template with metadata
                template = PromptTemplate(
                    template=template_data.get('template', ''),
                    name=name,
                    description=template_data.get('description', ''),
                    variables=template_data.get('variables'),
                    metadata=template_data.get('metadata', {})
                )
            else:
                logger.warning(f"Skipping invalid template '{name}': unsupported format")
                continue
            
            self.add_template(template)
        
        # Load categories
        for category, template_names in categories.items():
            self.categories[category] = template_names
    
    def load_directory(self, directory: str) -> None:
        """
        Load all template files from a directory.
        
        Args:
            directory (str): Directory path containing template files
        """
        if not os.path.exists(directory):
            logger.warning(f"Templates directory not found: {directory}")
            return
        
        for file_path in Path(directory).glob("*.yaml"):
            try:
                self.load_from_file(str(file_path))
            except Exception as e:
                logger.error(f"Failed to load template file {file_path}: {str(e)}")
        
        for file_path in Path(directory).glob("*.yml"):
            try:
                self.load_from_file(str(file_path))
            except Exception as e:
                logger.error(f"Failed to load template file {file_path}: {str(e)}")
        
        for file_path in Path(directory).glob("*.json"):
            try:
                self.load_from_file(str(file_path))
            except Exception as e:
                logger.error(f"Failed to load template file {file_path}: {str(e)}")
    
    def save_to_file(self, file_path: str) -> None:
        """
        Save all templates to a YAML file.
        
        Args:
            file_path (str): Path where to save the templates
        """
        data = {
            'templates': {},
            'categories': self.categories
        }
        
        # Convert templates to saveable format
        for name, template in self.templates.items():
            data['templates'][name] = {
                'template': template.template,
                'description': template.description,
                'variables': template.variables,
                'metadata': template.metadata
            }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            logger.info(f"Saved templates to {file_path}")
            
        except Exception as e:
            raise ValueError(f"Failed to save templates to {file_path}: {str(e)}")
    
    def create_template(
        self,
        name: str,
        template: str,
        description: str = "",
        category: Optional[str] = None,
        **metadata
    ) -> PromptTemplate:
        """
        Create and add a new template.
        
        Args:
            name (str): Template name
            template (str): Template string
            description (str): Template description
            category (Optional[str]): Category for the template
            **metadata: Additional metadata
            
        Returns:
            PromptTemplate: Created template
        """
        prompt_template = PromptTemplate(
            template=template,
            name=name,
            description=description,
            metadata=metadata
        )
        
        self.add_template(prompt_template, category)
        return prompt_template
    
    def render_template(self, name: str, **kwargs) -> str:
        """
        Render a template by name with provided variables.
        
        Args:
            name (str): Template name
            **kwargs: Variables for template rendering
            
        Returns:
            str: Rendered template
            
        Raises:
            KeyError: If template is not found
        """
        template = self.get_template(name)
        return template.render(**kwargs)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the template manager.
        
        Returns:
            Dict[str, Any]: Statistics including counts and categories
        """
        return {
            "total_templates": len(self.templates),
            "total_categories": len(self.categories),
            "categories": {
                category: len(templates)
                for category, templates in self.categories.items()
            },
            "templates_by_name": list(self.templates.keys())
        }


# Predefined common templates
COMMON_TEMPLATES = {
    "basic_completion": PromptTemplate(
        template="Complete the following text: {text}",
        name="basic_completion",
        description="Basic text completion template"
    ),
    
    "summarization": PromptTemplate(
        template="Summarize the following text in {max_sentences} sentences:\n\n{text}",
        name="summarization",
        description="Text summarization template",
        variables=["text", "max_sentences"]
    ),
    
    "question_answering": PromptTemplate(
        template="Based on the following context, answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
        name="question_answering",
        description="Question answering with context template"
    ),
    
    "translation": PromptTemplate(
        template="Translate the following text from {source_language} to {target_language}:\n\n{text}",
        name="translation",
        description="Text translation template"
    ),
    
    "creative_writing": PromptTemplate(
        template="Write a {genre} story about {topic}. The story should be {length} and include {elements}.",
        name="creative_writing",
        description="Creative writing prompt template"
    ),
}


def get_common_template(name: str) -> PromptTemplate:
    """
    Get a predefined common template.
    
    Args:
        name (str): Name of the common template
        
    Returns:
        PromptTemplate: The requested template
        
    Raises:
        KeyError: If template is not found
    """
    if name not in COMMON_TEMPLATES:
        available = list(COMMON_TEMPLATES.keys())
        raise KeyError(f"Common template '{name}' not found. Available: {available}")
    
    return COMMON_TEMPLATES[name].copy()


def create_template_manager_with_common() -> TemplateManager:
    """
    Create a template manager pre-loaded with common templates.
    
    Returns:
        TemplateManager: Manager with common templates loaded
    """
    manager = TemplateManager()
    
    for template in COMMON_TEMPLATES.values():
        manager.add_template(template.copy(), category="common")
    
    return manager