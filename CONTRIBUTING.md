# Contributing to Generative AI Project

Thank you for your interest in contributing to the Generative AI Project! This document provides guidelines and instructions for contributors to help maintain code quality and project consistency.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Code Standards](#code-standards)
4. [Testing Guidelines](#testing-guidelines)
5. [Documentation](#documentation)
6. [Submitting Changes](#submitting-changes)
7. [Issue Guidelines](#issue-guidelines)
8. [Code Review Process](#code-review-process)

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Virtual environment (recommended)

### Setting Up Your Development Environment

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/generative_ai_project.git
   cd generative_ai_project
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt  # When available
   pip install -e .  # Install in development mode
   ```

4. **Set Up Environment Variables**
   ```bash
   cp .env.example .env  # When available
   # Edit .env with your API keys and configuration
   ```

5. **Verify Installation**
   ```bash
   python examples/basic_completion.py
   ```

## Development Setup

### Project Structure

```
generative_ai_project/
├── src/                     # Source code
│   ├── llm/                # LLM client implementations
│   ├── prompt_engineering/ # Prompt engineering tools
│   ├── utils/              # Utility functions
│   └── handlers/           # Error handling
├── config/                 # Configuration files
├── examples/               # Example implementations
├── tests/                  # Test files (when added)
├── docs/                   # Documentation (when added)
└── notebooks/              # Jupyter notebooks
```

### Branch Naming Convention

- `feature/feature-name` - New features
- `bugfix/issue-description` - Bug fixes
- `docs/documentation-update` - Documentation updates
- `refactor/component-name` - Code refactoring
- `test/test-description` - Test additions

## Code Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

#### General Guidelines

- **Line Length**: Maximum 88 characters (Black formatter standard)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Use absolute imports, group imports logically
- **Naming Conventions**:
  - Classes: `PascalCase`
  - Functions/variables: `snake_case`
  - Constants: `UPPER_SNAKE_CASE`
  - Private members: `_leading_underscore`

#### Code Formatting

We use **Black** for code formatting:

```bash
# Install Black
pip install black

# Format your code
black src/ examples/ tests/

# Check formatting without making changes
black --check src/ examples/ tests/
```

#### Import Organization

Organize imports in the following order:

1. Standard library imports
2. Third-party library imports
3. Local application imports

```python
import os
import sys
from typing import Dict, List, Optional

import yaml
import openai

from src.llm.base import BaseLLMClient
from src.utils.logger import get_logger
```

#### Type Hints

Use type hints for all public functions and methods:

```python
from typing import Dict, List, Optional, Any

def process_text(
    text: str,
    model_name: str = "gpt-3.5-turbo",
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """Process text with the specified model."""
    # Implementation here
    pass
```

### Documentation Standards

#### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    Brief description of the function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1 (str): Description of param1
        param2 (int, optional): Description of param2. Defaults to 10.
        
    Returns:
        bool: Description of return value
        
    Raises:
        ValueError: Description of when this exception is raised
        
    Example:
        >>> result = example_function("test", 20)
        >>> print(result)
        True
    """
    # Implementation here
    pass
```

#### Class Documentation

```python
class ExampleClass:
    """
    Brief description of the class.
    
    Longer description explaining the purpose and usage of the class.
    Include information about key methods, attributes, and usage patterns.
    
    Attributes:
        attribute1 (str): Description of attribute1
        attribute2 (int): Description of attribute2
        
    Example:
        >>> obj = ExampleClass("value")
        >>> result = obj.process()
    """
    
    def __init__(self, param: str):
        """
        Initialize the class.
        
        Args:
            param (str): Description of the parameter
        """
        self.attribute1 = param
        self.attribute2 = 0
```

### Error Handling

#### Custom Exceptions

Use the project's custom exception hierarchy:

```python
from src.handlers import ValidationError, APIError, handle_error

def example_function(data: str) -> str:
    """Example with proper error handling."""
    try:
        if not data:
            raise ValidationError("Data cannot be empty", field="data")
        
        # Process data
        result = process_data(data)
        return result
        
    except Exception as e:
        error_info = handle_error(e, {"function": "example_function"})
        logger.error(f"Function failed: {error_info.message}")
        raise
```

#### Logging

Use the project's logging utilities:

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)

def example_function():
    """Example with proper logging."""
    logger.info("Starting function execution")
    
    try:
        # Do work
        logger.debug("Processing step completed")
        
    except Exception as e:
        logger.error(f"Function failed: {str(e)}")
        raise
    
    logger.info("Function completed successfully")
```

## Testing Guidelines

### Writing Tests

When the testing framework is added, follow these guidelines:

#### Test Structure

```python
import pytest
from src.llm.openai_client import OpenAIClient

class TestOpenAIClient:
    """Test cases for OpenAI client."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OpenAIClient(api_key="test-key")
    
    def test_initialization(self):
        """Test client initialization."""
        assert self.client.model_name == "gpt-3.5-turbo"
        assert self.client.api_key == "test-key"
    
    def test_generate_text_with_valid_input(self):
        """Test text generation with valid input."""
        # Mock the API response
        with patch.object(self.client, 'generate_text') as mock_generate:
            mock_generate.return_value = "Generated text"
            
            result = self.client.generate_text("Test prompt")
            
            assert result == "Generated text"
            mock_generate.assert_called_once_with("Test prompt")
    
    def test_generate_text_with_invalid_input(self):
        """Test text generation with invalid input."""
        with pytest.raises(ValidationError):
            self.client.generate_text("")
```

#### Test Naming

- Test files: `test_module_name.py`
- Test classes: `TestClassName`
- Test methods: `test_specific_behavior`

#### Mock External Dependencies

Always mock external API calls:

```python
from unittest.mock import patch, MagicMock

@patch('openai.ChatCompletion.create')
def test_api_call(mock_create):
    """Test API call with mocked response."""
    mock_create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Response"))]
    )
    
    # Test your code here
```

## Documentation

### Code Comments

- Use comments sparingly and only when necessary
- Explain **why**, not **what**
- Keep comments up-to-date with code changes

```python
# Good: Explains why
def calculate_cost(tokens: int, model: str) -> float:
    # Apply 10% discount for high-volume usage
    if tokens > 100000:
        return base_cost * 0.9
    return base_cost

# Bad: Explains what (obvious from code)
def calculate_cost(tokens: int, model: str) -> float:
    # Check if tokens is greater than 100000
    if tokens > 100000:
        # Multiply base_cost by 0.9
        return base_cost * 0.9
    return base_cost
```

### README Updates

When adding new features, update relevant documentation:

- Update feature lists
- Add new configuration options
- Include usage examples
- Update installation instructions if needed

## Submitting Changes

### Pull Request Process

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow code standards
   - Add tests (when framework is available)
   - Update documentation

3. **Test Your Changes**
   ```bash
   # Run examples to ensure they still work
   python examples/basic_completion.py
   python examples/chat_session.py
   python examples/chain_prompts.py
   
   # Run tests when available
   pytest tests/
   
   # Check code formatting
   black --check src/ examples/
   ```

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

5. **Push and Create Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Use conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add Claude client implementation
fix: resolve rate limiting issue in OpenAI client
docs: update installation instructions
refactor: improve error handling in prompt chains
```

### Pull Request Description

Include in your PR description:

- **Summary**: Brief description of changes
- **Motivation**: Why this change is needed
- **Changes**: Detailed list of what was changed
- **Testing**: How the changes were tested
- **Breaking Changes**: Any breaking changes (if applicable)

Example:
```markdown
## Summary
Add support for custom retry policies in LLM clients.

## Motivation
Users need more control over retry behavior for different use cases.

## Changes
- Added RetryPolicy class in utils module
- Updated BaseLLMClient to support custom retry policies
- Added configuration options for retry behavior
- Updated documentation and examples

## Testing
- Tested with OpenAI and Claude clients
- Verified retry behavior with network failures
- All existing examples continue to work

## Breaking Changes
None
```

## Issue Guidelines

### Reporting Bugs

Include the following information:

- **Environment**: Python version, OS, dependency versions
- **Steps to Reproduce**: Clear steps to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Error Messages**: Full error messages and stack traces
- **Code Examples**: Minimal code to reproduce the issue

### Feature Requests

Include the following information:

- **Use Case**: Describe your use case and why this feature is needed
- **Proposed Solution**: Your ideas for implementation
- **Alternatives**: Alternative solutions you've considered
- **Additional Context**: Any other relevant information

### Questions and Discussions

- Check existing issues and documentation first
- Use clear, descriptive titles
- Provide context and examples
- Be respectful and constructive

## Code Review Process

### What We Look For

- **Code Quality**: Follows style guidelines and best practices
- **Documentation**: Well-documented code with clear docstrings
- **Testing**: Adequate test coverage (when framework is available)
- **Performance**: Efficient implementation
- **Compatibility**: Works with existing code and doesn't break functionality

### Review Timeline

- Initial review within 2-3 days
- Follow-up reviews within 1-2 days
- Approval and merge after all feedback is addressed

### Addressing Feedback

- Respond to all review comments
- Make requested changes in additional commits
- Ask for clarification if feedback is unclear
- Be open to suggestions and constructive criticism

## Additional Guidelines

### Security Considerations

- Never commit API keys or sensitive information
- Use environment variables for configuration
- Validate all user inputs
- Handle errors securely (don't expose sensitive information)

### Performance Guidelines

- Use appropriate data structures and algorithms
- Implement caching where beneficial
- Consider memory usage for large datasets
- Profile code for performance bottlenecks

### Backward Compatibility

- Maintain backward compatibility when possible
- Clearly document breaking changes
- Provide migration guides for major changes
- Use deprecation warnings before removing features

## Getting Help

If you need help or have questions:

- Check the documentation and examples
- Search existing issues
- Create a new issue with the "question" label
- Reach out to maintainers

## Recognition

We appreciate all contributions! Contributors will be:

- Added to the contributor list
- Mentioned in release notes for significant contributions
- Recognized in project documentation

Thank you for contributing to the Generative AI Project! 🚀