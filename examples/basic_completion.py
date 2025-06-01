#!/usr/bin/env python3
"""
Basic Completion Example

This example demonstrates how to use the generative AI project for basic text completion
tasks using different LLM providers and models.

Author: Brij Kishore Pandey
"""

import os
import sys
import asyncio
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import OpenAIClient, ClaudeClient, LLMManager
from prompt_engineering import PromptTemplate, get_common_template
from utils import setup_logging, get_logger, rate_limit, count_tokens
from handlers import handle_error, ValidationError


# Setup logging
setup_logging(level="INFO", console_colors=True)
logger = get_logger(__name__)


class BasicCompletionExample:
    """
    Example class demonstrating basic text completion functionality.
    
    This class shows how to:
    - Set up LLM clients
    - Use prompt templates
    - Handle errors gracefully
    - Apply rate limiting
    - Track token usage
    """
    
    def __init__(self):
        """Initialize the example with LLM clients and templates."""
        self.manager = LLMManager()
        self.setup_clients()
        self.setup_templates()
    
    def setup_clients(self):
        """Set up LLM clients for different providers."""
        try:
            # Setup OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                openai_client = OpenAIClient(
                    model_name="gpt-3.5-turbo",
                    api_key=openai_api_key,
                    max_tokens=500,
                    temperature=0.7
                )
                self.manager.add_client("openai", openai_client)
                logger.info("OpenAI client added successfully")
            else:
                logger.warning("OPENAI_API_KEY not found in environment variables")
            
            # Setup Claude client
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            if anthropic_api_key:
                claude_client = ClaudeClient(
                    model_name="claude-3-haiku-20240307",
                    api_key=anthropic_api_key,
                    max_tokens=500,
                    temperature=0.7
                )
                self.manager.add_client("claude", claude_client)
                logger.info("Claude client added successfully")
            else:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
            
            if not self.manager.list_clients():
                raise ValueError("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
                
        except Exception as e:
            error_info = handle_error(e, {"context": "client_setup"})
            logger.error(f"Failed to setup clients: {error_info.message}")
            raise
    
    def setup_templates(self):
        """Set up prompt templates for different completion tasks."""
        self.templates = {
            "basic": PromptTemplate(
                "Complete the following text: {text}",
                name="basic_completion",
                description="Simple text completion"
            ),
            
            "creative": PromptTemplate(
                "Write a creative continuation of this story:\n\n{text}\n\nContinuation:",
                name="creative_completion",
                description="Creative story continuation"
            ),
            
            "professional": PromptTemplate(
                "Complete this professional email in a {tone} tone:\n\n{text}",
                name="professional_completion",
                description="Professional email completion"
            ),
            
            "technical": PromptTemplate(
                "Complete this technical documentation:\n\n{text}\n\nRequirements:\n- Be precise and accurate\n- Use appropriate technical terminology\n- Maintain professional tone\n\nCompletion:",
                name="technical_completion",
                description="Technical documentation completion"
            )
        }
        
        logger.info(f"Set up {len(self.templates)} prompt templates")
    
    @rate_limit(calls_per_minute=30)
    def complete_text(
        self,
        text: str,
        template_name: str = "basic",
        client_name: str = None,
        **template_vars
    ) -> Dict[str, Any]:
        """
        Complete text using the specified template and client.
        
        Args:
            text (str): Text to complete
            template_name (str): Name of the template to use
            client_name (str): Name of the client to use (optional)
            **template_vars: Additional variables for the template
            
        Returns:
            Dict[str, Any]: Completion result with metadata
        """
        try:
            # Validate inputs
            if not text.strip():
                raise ValidationError("Text cannot be empty", field="text", value=text)
            
            if template_name not in self.templates:
                raise ValidationError(
                    f"Template '{template_name}' not found. Available: {list(self.templates.keys())}",
                    field="template_name",
                    value=template_name
                )
            
            # Get template and prepare prompt
            template = self.templates[template_name]
            template_vars["text"] = text
            
            # Validate template variables
            validation = template.validate_variables(template_vars)
            if not validation["valid"]:
                raise ValidationError(
                    f"Missing template variables: {validation['missing']}",
                    field="template_variables",
                    value=validation
                )
            
            prompt = template.render(**template_vars)
            
            # Count input tokens
            input_tokens = count_tokens(prompt, self.manager.default_client or "gpt-3.5-turbo")
            logger.info(f"Input tokens: {input_tokens}")
            
            # Generate completion
            response = self.manager.generate(prompt, client_name)
            
            # Count output tokens
            output_tokens = count_tokens(response, self.manager.default_client or "gpt-3.5-turbo")
            logger.info(f"Output tokens: {output_tokens}")
            
            result = {
                "original_text": text,
                "template_used": template_name,
                "client_used": client_name or self.manager.default_client,
                "prompt": prompt,
                "completion": response,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "template_info": template.get_info()
            }
            
            logger.info(f"Successfully completed text using {template_name} template")
            return result
            
        except Exception as e:
            error_info = handle_error(e, {
                "text_length": len(text) if text else 0,
                "template_name": template_name,
                "client_name": client_name
            })
            logger.error(f"Text completion failed: {error_info.message}")
            return {"error": error_info.to_dict()}
    
    def batch_complete(
        self,
        texts: List[str],
        template_name: str = "basic",
        client_name: str = None
    ) -> List[Dict[str, Any]]:
        """
        Complete multiple texts in batch.
        
        Args:
            texts (List[str]): List of texts to complete
            template_name (str): Template to use for all completions
            client_name (str): Client to use for all completions
            
        Returns:
            List[Dict[str, Any]]: List of completion results
        """
        results = []
        
        logger.info(f"Starting batch completion of {len(texts)} texts")
        
        for i, text in enumerate(texts):
            logger.info(f"Processing text {i+1}/{len(texts)}")
            
            try:
                result = self.complete_text(text, template_name, client_name)
                results.append(result)
                
            except Exception as e:
                error_info = handle_error(e, {
                    "batch_index": i,
                    "text_preview": text[:50] + "..." if len(text) > 50 else text
                })
                results.append({"error": error_info.to_dict()})
        
        successful = sum(1 for r in results if "error" not in r)
        logger.info(f"Batch completion finished: {successful}/{len(texts)} successful")
        
        return results
    
    def demonstrate_different_templates(self):
        """Demonstrate different completion templates with example texts."""
        examples = [
            {
                "template": "basic",
                "text": "The future of artificial intelligence is",
                "description": "Basic completion"
            },
            {
                "template": "creative", 
                "text": "Sarah opened the mysterious door and found herself in a room filled with",
                "description": "Creative story continuation"
            },
            {
                "template": "professional",
                "text": "Dear Mr. Johnson,\n\nI hope this email finds you well. I am writing to follow up on our conversation about",
                "template_vars": {"tone": "formal"},
                "description": "Professional email completion"
            },
            {
                "template": "technical",
                "text": "API Rate Limiting\n\nRate limiting is a technique used to control the rate of requests sent to an API. The implementation involves",
                "description": "Technical documentation completion"
            }
        ]
        
        logger.info("Demonstrating different completion templates")
        
        for i, example in enumerate(examples, 1):
            print(f"\n{'='*60}")
            print(f"Example {i}: {example['description']}")
            print(f"{'='*60}")
            
            template_vars = example.get("template_vars", {})
            result = self.complete_text(
                example["text"],
                example["template"],
                **template_vars
            )
            
            if "error" in result:
                print(f"Error: {result['error']['message']}")
                continue
            
            print(f"Template: {result['template_used']}")
            print(f"Client: {result['client_used']}")
            print(f"Tokens: {result['tokens']['total']} ({result['tokens']['input']} input + {result['tokens']['output']} output)")
            print(f"\nOriginal text:")
            print(f"'{result['original_text']}'")
            print(f"\nCompletion:")
            print(f"'{result['completion']}'")
    
    def demonstrate_client_comparison(self):
        """Demonstrate completion using different LLM clients."""
        text = "The impact of machine learning on modern society includes"
        
        print(f"\n{'='*60}")
        print("Client Comparison")
        print(f"{'='*60}")
        print(f"Text to complete: '{text}'")
        
        for client_name in self.manager.list_clients():
            print(f"\n--- Using {client_name.upper()} ---")
            
            result = self.complete_text(text, "basic", client_name)
            
            if "error" in result:
                print(f"Error: {result['error']['message']}")
                continue
            
            print(f"Completion: '{result['completion']}'")
            print(f"Tokens: {result['tokens']['total']}")


def main():
    """Main function to run the basic completion examples."""
    try:
        # Create example instance
        example = BasicCompletionExample()
        
        print("🚀 Basic Completion Example")
        print("=" * 60)
        
        # Demonstrate different templates
        example.demonstrate_different_templates()
        
        # Demonstrate client comparison (if multiple clients available)
        if len(example.manager.list_clients()) > 1:
            example.demonstrate_client_comparison()
        
        # Demonstrate batch processing
        batch_texts = [
            "The benefits of renewable energy include",
            "In the world of quantum computing,",
            "The role of education in society is"
        ]
        
        print(f"\n{'='*60}")
        print("Batch Processing Example")
        print(f"{'='*60}")
        
        batch_results = example.batch_complete(batch_texts)
        
        for i, result in enumerate(batch_results, 1):
            if "error" not in result:
                print(f"\nBatch {i}: '{result['completion'][:100]}...'")
            else:
                print(f"\nBatch {i}: Error - {result['error']['message']}")
        
        # Print client information
        print(f"\n{'='*60}")
        print("Client Information")
        print(f"{'='*60}")
        
        for client_name in example.manager.list_clients():
            info = example.manager.get_client_info(client_name)
            print(f"\n{client_name.upper()}:")
            print(f"  Model: {info['model_name']}")
            print(f"  Provider: {info['provider']}")
            print(f"  Max tokens: {info['max_tokens']}")
            print(f"  Temperature: {info['temperature']}")
        
        print(f"\n{'='*60}")
        print("✅ Basic completion example completed successfully!")
        print(f"{'='*60}")
        
    except Exception as e:
        error_info = handle_error(e, {"context": "main_execution"})
        logger.error(f"Example execution failed: {error_info.message}")
        
        if error_info.suggestions:
            print("\n💡 Suggestions:")
            for suggestion in error_info.suggestions:
                print(f"  • {suggestion}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()