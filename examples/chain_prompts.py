#!/usr/bin/env python3
"""
Prompt Chaining Example

This example demonstrates advanced prompt chaining capabilities, including
sequential processing, conditional branching, and complex multi-step workflows.

Author: Brij Kishore Pandey
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import OpenAIClient, ClaudeClient, LLMManager
from prompt_engineering import (
    PromptChain, ChainBuilder, LLMStep, ProcessingStep, ConditionalStep,
    create_analysis_chain, create_qa_chain
)
from utils import setup_logging, get_logger, perf_logger
from handlers import handle_error, ValidationError


# Setup logging
setup_logging(level="INFO", console_colors=True)
logger = get_logger(__name__)


class PromptChainDemo:
    """
    Demonstration of prompt chaining capabilities.
    
    This class shows how to:
    - Create sequential prompt chains
    - Use conditional branching
    - Process data between steps
    - Monitor chain execution
    - Handle errors in chains
    """
    
    def __init__(self):
        """Initialize the demo with LLM clients."""
        self.manager = LLMManager()
        self.setup_clients()
        self.client = self.manager.get_client(self.manager.default_client)
    
    def setup_clients(self):
        """Set up LLM clients."""
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            openai_client = OpenAIClient(api_key=openai_key, model_name="gpt-3.5-turbo")
            self.manager.add_client("openai", openai_client)
            logger.info("OpenAI client added")
        
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            claude_client = ClaudeClient(api_key=anthropic_key, model_name="claude-3-haiku-20240307")
            self.manager.add_client("claude", claude_client)
            logger.info("Claude client added")
        
        if not self.manager.list_clients():
            raise ValueError("No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    def demo_basic_sequential_chain(self):
        """Demonstrate a basic sequential prompt chain."""
        print("\n🔗 Basic Sequential Chain Demo")
        print("=" * 50)
        
        # Create a simple sequential chain for content analysis
        builder = ChainBuilder("content_analysis", "Analyze and summarize content")
        
        chain = (builder
            .add_llm_step(
                "extract_topics",
                self.client,
                "Extract the main topics from this text:\n\n{text}\n\nMain topics (list format):",
                "Extract main topics from the input text"
            )
            .add_llm_step(
                "analyze_sentiment",
                self.client,
                "Analyze the sentiment of this text:\n\n{text}\n\nSentiment analysis:",
                "Analyze the sentiment of the original text"
            )
            .add_llm_step(
                "create_summary",
                self.client,
                "Create a comprehensive summary based on:\n\nOriginal text: {text}\n\nTopics: {response}\n\nSentiment: {step_1_result}\n\nSummary:",
                "Create final summary incorporating all analysis"
            )
            .build())
        
        # Test input
        test_text = """
        Artificial intelligence is revolutionizing industries across the globe. From healthcare 
        to finance, AI technologies are enabling unprecedented efficiency and innovation. 
        However, this rapid advancement also brings challenges including job displacement, 
        privacy concerns, and the need for new regulatory frameworks. Despite these challenges, 
        the potential benefits of AI in solving complex global problems like climate change 
        and disease make it one of the most promising technologies of our time.
        """
        
        print(f"Input text: {test_text.strip()[:100]}...")
        print("\nExecuting chain...")
        
        # Execute the chain
        with perf_logger.measure("sequential_chain_execution"):
            result = chain.execute({"text": test_text.strip()})
        
        if "error" not in result:
            print(f"\n✅ Chain executed successfully!")
            print(f"Total steps: {result['execution_summary']['total_steps']}")
            print(f"Execution time: {result['execution_summary']['total_time']:.2f}s")
            
            # Show final result
            final_data = result['final_data']
            if 'response' in final_data:
                print(f"\nFinal Summary:")
                print(f"{final_data['response']}")
        else:
            print(f"❌ Chain execution failed: {result.get('error', 'Unknown error')}")
        
        return result
    
    def demo_conditional_chain(self):
        """Demonstrate conditional branching in chains."""
        print("\n🔀 Conditional Chain Demo")
        print("=" * 50)
        
        # Classification function for branching
        def classify_content_type(input_data: Dict[str, Any], context: Dict[str, Any]) -> str:
            """Classify content type based on text characteristics."""
            text = input_data.get("text", "").lower()
            
            if any(word in text for word in ["study", "research", "analysis", "hypothesis"]):
                return "academic"
            elif any(word in text for word in ["product", "buy", "price", "offer"]):
                return "commercial"
            elif any(word in text for word in ["story", "character", "plot", "novel"]):
                return "creative"
            else:
                return "general"
        
        # Create branches for different content types
        branches = {
            "academic": LLMStep(
                "academic_analysis",
                self.client,
                "Provide an academic analysis of this text:\n\n{text}\n\nAcademic Analysis:",
                "Academic analysis of the content"
            ),
            "commercial": LLMStep(
                "commercial_analysis",
                self.client,
                "Analyze this commercial content:\n\n{text}\n\nCommercial Analysis:",
                "Commercial analysis of the content"
            ),
            "creative": LLMStep(
                "creative_analysis",
                self.client,
                "Provide creative feedback on this text:\n\n{text}\n\nCreative Feedback:",
                "Creative analysis of the content"
            ),
            "general": LLMStep(
                "general_analysis",
                self.client,
                "Provide a general analysis of this text:\n\n{text}\n\nGeneral Analysis:",
                "General analysis of the content"
            )
        }
        
        # Build conditional chain
        builder = ChainBuilder("conditional_analysis", "Conditional content analysis")
        
        chain = (builder
            .add_conditional_step(
                "classify_and_analyze",
                classify_content_type,
                branches,
                default_branch="general",
                "Classify content type and analyze accordingly"
            )
            .add_llm_step(
                "summarize_analysis",
                self.client,
                "Create a final summary of this analysis:\n\n{response}\n\nContent type: {branch_taken}\n\nFinal Summary:",
                "Create final summary of the analysis"
            )
            .build())
        
        # Test different content types
        test_cases = [
            {
                "type": "Academic",
                "text": "This research study examines the hypothesis that machine learning algorithms can improve diagnostic accuracy in medical imaging. Our analysis of 10,000 patient scans shows significant improvements."
            },
            {
                "type": "Commercial", 
                "text": "Introducing our new AI-powered productivity suite! Increase your team's efficiency by 40% with our innovative tools. Special launch price of $99/month - limited time offer!"
            },
            {
                "type": "Creative",
                "text": "The protagonist walked through the misty forest, her heart pounding as shadows danced between the ancient trees. She knew the dragon was near, its presence felt in every whisper of wind."
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i}: {test_case['type']} Content ---")
            print(f"Text: {test_case['text'][:80]}...")
            
            result = chain.execute({"text": test_case["text"]})
            
            if "error" not in result:
                step_results = result.get("step_results", [])
                if step_results:
                    branch_taken = step_results[0].get("output_data", {}).get("branch_taken", "unknown")
                    print(f"Classification: {branch_taken}")
                
                final_response = result.get("final_data", {}).get("response", "No response")
                print(f"Analysis: {final_response[:150]}...")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    def demo_data_processing_chain(self):
        """Demonstrate chain with data processing steps."""
        print("\n⚙️ Data Processing Chain Demo")
        print("=" * 50)
        
        # Custom processing functions
        def extract_keywords(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Extract keywords from text using simple heuristics."""
            text = input_data.get("response", input_data.get("text", ""))
            
            # Simple keyword extraction (in practice, you might use NLP libraries)
            words = text.lower().split()
            
            # Filter for meaningful words (simple approach)
            keywords = [
                word.strip(".,!?;:")
                for word in words
                if len(word) > 4 and word.isalpha()
            ]
            
            # Count frequency and get top keywords
            from collections import Counter
            keyword_counts = Counter(keywords)
            top_keywords = [word for word, count in keyword_counts.most_common(10)]
            
            return {
                "keywords": top_keywords,
                "keyword_count": len(top_keywords),
                "total_words": len(words)
            }
        
        def format_results(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """Format the final results into a structured report."""
            analysis = input_data.get("response", "")
            keywords_data = context.get("step_1_result", {})
            
            report = {
                "analysis": analysis,
                "keywords": keywords_data.get("keywords", []),
                "statistics": {
                    "total_words": keywords_data.get("total_words", 0),
                    "key_terms": keywords_data.get("keyword_count", 0)
                },
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            return {"formatted_report": json.dumps(report, indent=2)}
        
        # Build processing chain
        builder = ChainBuilder("data_processing", "Text analysis with data processing")
        
        chain = (builder
            .add_llm_step(
                "analyze_text",
                self.client,
                "Provide a detailed analysis of this text:\n\n{text}\n\nDetailed Analysis:",
                "Analyze the input text"
            )
            .add_processing_step(
                "extract_keywords",
                extract_keywords,
                "Extract keywords from the analysis"
            )
            .add_processing_step(
                "format_report",
                format_results,
                "Format the final report"
            )
            .build())
        
        # Test input
        test_text = """
        Climate change represents one of the most pressing challenges of our time. Rising global 
        temperatures, melting ice caps, and extreme weather events are clear indicators of 
        environmental transformation. Sustainable energy solutions, including solar and wind power, 
        offer promising pathways to reduce carbon emissions. However, implementing these technologies 
        requires significant investment, policy changes, and international cooperation.
        """
        
        print(f"Input: {test_text.strip()[:100]}...")
        print("\nExecuting data processing chain...")
        
        result = chain.execute({"text": test_text.strip()})
        
        if "error" not in result:
            print(f"\n✅ Processing completed successfully!")
            
            # Show formatted report
            formatted_report = result.get("final_data", {}).get("formatted_report")
            if formatted_report:
                print("\nFormatted Report:")
                print(formatted_report)
        else:
            print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
    
    def demo_predefined_chains(self):
        """Demonstrate predefined chain patterns."""
        print("\n📋 Predefined Chain Patterns Demo")
        print("=" * 50)
        
        # Demo analysis chain
        print("\n--- Analysis Chain ---")
        analysis_chain = create_analysis_chain(self.client, "demo_analysis")
        
        analysis_text = """
        The emergence of remote work has fundamentally changed how businesses operate. 
        Companies have discovered increased productivity, reduced overhead costs, and 
        access to global talent. However, challenges include maintaining team cohesion, 
        ensuring cybersecurity, and managing work-life balance for employees.
        """
        
        print("Executing analysis chain...")
        analysis_result = analysis_chain.execute({"text": analysis_text})
        
        if "error" not in analysis_result:
            print("✅ Analysis chain completed")
            final_summary = analysis_result.get("final_data", {}).get("response", "")
            print(f"Final summary: {final_summary[:200]}...")
        
        # Demo Q&A chain
        print("\n--- Q&A Chain ---")
        qa_chain = create_qa_chain(self.client, "demo_qa")
        
        context = """
        Quantum computing is a revolutionary technology that uses quantum mechanical phenomena 
        to process information. Unlike classical computers that use bits (0 or 1), quantum 
        computers use quantum bits or qubits that can exist in multiple states simultaneously. 
        This allows quantum computers to potentially solve certain problems exponentially 
        faster than classical computers.
        """
        
        question = "How do quantum computers differ from classical computers?"
        
        print(f"Question: {question}")
        print("Executing Q&A chain...")
        
        qa_result = qa_chain.execute({
            "context": context,
            "question": question
        })
        
        if "error" not in qa_result:
            print("✅ Q&A chain completed")
            final_answer = qa_result.get("final_data", {}).get("response", "")
            print(f"Answer: {final_answer[:200]}...")
    
    def demo_error_handling_and_recovery(self):
        """Demonstrate error handling in chains."""
        print("\n🛡️ Error Handling Demo")
        print("=" * 50)
        
        # Create a chain that might fail
        def potentially_failing_process(input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
            """A processing step that might fail."""
            text = input_data.get("text", "")
            
            # Simulate failure condition
            if "error" in text.lower():
                raise ValueError("Simulated processing error")
            
            return {"processed": f"Processed: {text}"}
        
        # Build chain with fail_fast=False to continue on errors
        builder = ChainBuilder("error_demo", "Error handling demonstration")
        
        chain = (builder
            .set_fail_fast(False)  # Continue on errors
            .add_llm_step(
                "first_step",
                self.client,
                "Process this text: {text}",
                "First processing step"
            )
            .add_processing_step(
                "potentially_failing",
                potentially_failing_process,
                "Step that might fail"
            )
            .add_llm_step(
                "final_step",
                self.client,
                "Finalize processing: {response}",
                "Final step (should still execute even if previous step failed)"
            )
            .build())
        
        # Test cases: one that works, one that fails
        test_cases = [
            {"text": "This is normal text that should work fine."},
            {"text": "This text contains an error keyword that will trigger failure."}
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            print(f"Input: {test_case['text']}")
            
            result = chain.execute(test_case)
            
            print(f"Success rate: {result['execution_summary']['success_rate']:.2%}")
            print(f"Successful steps: {result['execution_summary']['successful_steps']}/{result['execution_summary']['total_steps']}")
            
            # Show step details
            if result.get("step_results"):
                for step_result in result["step_results"]:
                    status = step_result["status"]
                    step_name = step_result["step_name"]
                    
                    if status == "failed":
                        error = step_result.get("error", "Unknown error")
                        print(f"  ❌ {step_name}: {error}")
                    else:
                        print(f"  ✅ {step_name}: {status}")
    
    def demo_chain_visualization_and_monitoring(self):
        """Demonstrate chain visualization and monitoring."""
        print("\n📊 Chain Monitoring Demo")
        print("=" * 50)
        
        # Create a complex chain for demonstration
        builder = ChainBuilder("monitored_chain", "Complex chain with monitoring")
        
        chain = (builder
            .add_llm_step("step1", self.client, "Analyze: {text}", "First analysis")
            .add_llm_step("step2", self.client, "Expand on: {response}", "Second analysis")
            .add_llm_step("step3", self.client, "Summarize: {response}", "Final summary")
            .build())
        
        # Show chain visualization
        print("Chain Structure:")
        print(chain.visualize_chain())
        
        # Execute with monitoring
        test_text = "Renewable energy adoption is accelerating globally due to technological advances and policy support."
        
        print(f"\nExecuting chain with monitoring...")
        print(f"Input: {test_text}")
        
        start_time = time.time()
        result = chain.execute({"text": test_text})
        total_duration = time.time() - start_time
        
        # Detailed monitoring output
        if "error" not in result:
            print(f"\n📈 Execution Metrics:")
            summary = result["execution_summary"]
            print(f"  Total time: {summary['total_time']:.2f}s")
            print(f"  Steps completed: {summary['successful_steps']}/{summary['total_steps']}")
            print(f"  Success rate: {summary['success_rate']:.2%}")
            
            print(f"\n⏱️ Step-by-step timing:")
            if result.get("step_results"):
                for step_result in result["step_results"]:
                    step_name = step_result["step_name"]
                    duration = step_result["execution_time"]
                    status = step_result["status"]
                    print(f"  {step_name}: {duration:.2f}s ({status})")
            
            # Show chain info
            print(f"\n🔧 Chain Configuration:")
            chain_info = chain.get_chain_info()
            print(f"  Name: {chain_info['name']}")
            print(f"  Steps: {chain_info['step_count']}")
            print(f"  Fail fast: {chain_info['fail_fast']}")
            print(f"  Save intermediate: {chain_info['save_intermediate']}")


def main():
    """Main function to run prompt chaining examples."""
    try:
        print("🔗 Prompt Chaining Example")
        print("=" * 60)
        
        demo = PromptChainDemo()
        
        # Run all demonstrations
        demo.demo_basic_sequential_chain()
        demo.demo_conditional_chain()
        demo.demo_data_processing_chain()
        demo.demo_predefined_chains()
        demo.demo_error_handling_and_recovery()
        demo.demo_chain_visualization_and_monitoring()
        
        print(f"\n{'='*60}")
        print("✅ Prompt chaining example completed successfully!")
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