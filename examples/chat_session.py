#!/usr/bin/env python3
"""
Chat Session Example

This example demonstrates how to create interactive chat sessions with different
LLM providers, including conversation history management, context preservation,
and advanced chat features.

Author: Brij Kishore Pandey
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm import OpenAIClient, ClaudeClient, LLMManager
from prompt_engineering import PromptTemplate
from utils import setup_logging, get_logger, count_message_tokens, llm_cache
from handlers import handle_error, ValidationError


# Setup logging
setup_logging(level="INFO", console_colors=True)
logger = get_logger(__name__)


class ChatSession:
    """
    A chat session that maintains conversation history and context.
    
    This class provides:
    - Conversation history management
    - Context preservation across messages
    - Token counting and management
    - Chat caching for repeated sessions
    - Export/import functionality
    """
    
    def __init__(
        self,
        client_name: str,
        llm_manager: LLMManager,
        system_prompt: Optional[str] = None,
        max_history: int = 50,
        max_tokens_per_message: int = 1000
    ):
        """
        Initialize a chat session.
        
        Args:
            client_name (str): Name of the LLM client to use
            llm_manager (LLMManager): LLM manager instance
            system_prompt (Optional[str]): System prompt to set context
            max_history (int): Maximum number of messages to keep in history
            max_tokens_per_message (int): Maximum tokens per message
        """
        self.client_name = client_name
        self.llm_manager = llm_manager
        self.max_history = max_history
        self.max_tokens_per_message = max_tokens_per_message
        
        # Initialize conversation history
        self.messages: List[Dict[str, str]] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
        
        # Session metadata
        self.session_id = self._generate_session_id()
        self.created_at = datetime.now()
        self.total_tokens = 0
        self.message_count = 0
        
        logger.info(f"Chat session created with {client_name} client")
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        import uuid
        return str(uuid.uuid4())[:12]
    
    def send_message(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message in the chat session.
        
        Args:
            user_message (str): User's message
            
        Returns:
            Dict[str, Any]: Response with assistant's message and metadata
        """
        try:
            # Validate input
            if not user_message.strip():
                raise ValidationError("Message cannot be empty", field="user_message")
            
            # Add user message to history
            self.messages.append({"role": "user", "content": user_message.strip()})
            
            # Check if we need to truncate history
            self._manage_history()
            
            # Count tokens before making request
            token_count = count_message_tokens(
                self.messages,
                self.llm_manager.get_client(self.client_name).model_name
            )
            
            logger.info(f"Sending message with {token_count.total_tokens} total tokens")
            
            # Generate response
            response = self.llm_manager.chat(
                self.messages,
                self.client_name,
                max_tokens=self.max_tokens_per_message
            )
            
            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": response})
            
            # Update session metadata
            self.message_count += 1
            self.total_tokens += token_count.total_tokens
            
            # Prepare response
            result = {
                "session_id": self.session_id,
                "message_id": self.message_count,
                "user_message": user_message,
                "assistant_response": response,
                "tokens_used": token_count.total_tokens,
                "total_session_tokens": self.total_tokens,
                "timestamp": datetime.now().isoformat(),
                "client_used": self.client_name
            }
            
            logger.info(f"Message processed successfully (ID: {self.message_count})")
            return result
            
        except Exception as e:
            error_info = handle_error(e, {
                "session_id": self.session_id,
                "message_count": self.message_count,
                "user_message_length": len(user_message) if user_message else 0
            })
            logger.error(f"Failed to process message: {error_info.message}")
            return {"error": error_info.to_dict()}
    
    def _manage_history(self):
        """Manage conversation history to stay within limits."""
        # Keep system message if present
        system_messages = [msg for msg in self.messages if msg["role"] == "system"]
        other_messages = [msg for msg in self.messages if msg["role"] != "system"]
        
        # Truncate other messages if needed
        if len(other_messages) > self.max_history:
            # Keep the most recent messages
            other_messages = other_messages[-(self.max_history - len(system_messages)):]
            logger.info(f"Truncated history to {len(other_messages)} messages")
        
        # Rebuild messages list
        self.messages = system_messages + other_messages
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Returns:
            Dict[str, Any]: Conversation summary
        """
        user_messages = [msg for msg in self.messages if msg["role"] == "user"]
        assistant_messages = [msg for msg in self.messages if msg["role"] == "assistant"]
        
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "duration_minutes": (datetime.now() - self.created_at).total_seconds() / 60,
            "total_messages": len(self.messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "total_tokens": self.total_tokens,
            "client_used": self.client_name,
            "average_tokens_per_exchange": self.total_tokens / max(1, len(user_messages))
        }
    
    def export_conversation(self, file_path: Optional[str] = None) -> str:
        """
        Export conversation to JSON file.
        
        Args:
            file_path (Optional[str]): Path to save file (auto-generated if not provided)
            
        Returns:
            str: Path to the exported file
        """
        if not file_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"chat_session_{self.session_id}_{timestamp}.json"
        
        export_data = {
            "session_metadata": self.get_conversation_summary(),
            "messages": self.messages,
            "export_timestamp": datetime.now().isoformat()
        }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Conversation exported to {file_path}")
            return file_path
            
        except Exception as e:
            error_info = handle_error(e, {"file_path": file_path})
            logger.error(f"Failed to export conversation: {error_info.message}")
            raise
    
    @classmethod
    def import_conversation(
        cls,
        file_path: str,
        llm_manager: LLMManager,
        client_name: Optional[str] = None
    ) -> 'ChatSession':
        """
        Import conversation from JSON file.
        
        Args:
            file_path (str): Path to the JSON file
            llm_manager (LLMManager): LLM manager instance
            client_name (Optional[str]): Client name (uses original if not provided)
            
        Returns:
            ChatSession: Restored chat session
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = data["session_metadata"]
            messages = data["messages"]
            
            # Use original client or provided one
            original_client = metadata.get("client_used")
            client_to_use = client_name or original_client
            
            if client_to_use not in llm_manager.list_clients():
                raise ValueError(f"Client '{client_to_use}' not available")
            
            # Create new session
            system_prompt = None
            if messages and messages[0]["role"] == "system":
                system_prompt = messages[0]["content"]
            
            session = cls(client_to_use, llm_manager, system_prompt)
            
            # Restore messages and metadata
            session.messages = messages
            session.session_id = metadata["session_id"]
            session.total_tokens = metadata["total_tokens"]
            session.message_count = metadata["user_messages"]
            
            logger.info(f"Conversation imported from {file_path}")
            return session
            
        except Exception as e:
            error_info = handle_error(e, {"file_path": file_path})
            logger.error(f"Failed to import conversation: {error_info.message}")
            raise


class ChatSessionManager:
    """
    Manager for multiple chat sessions with different personas and contexts.
    """
    
    def __init__(self, llm_manager: LLMManager):
        """
        Initialize the chat session manager.
        
        Args:
            llm_manager (LLMManager): LLM manager instance
        """
        self.llm_manager = llm_manager
        self.sessions: Dict[str, ChatSession] = {}
        self.personas = self._setup_personas()
        
        logger.info("Chat session manager initialized")
    
    def _setup_personas(self) -> Dict[str, str]:
        """Set up different chat personas with system prompts."""
        return {
            "assistant": "You are a helpful, knowledgeable, and friendly AI assistant. Provide clear, accurate, and helpful responses to user questions.",
            
            "tutor": "You are an experienced tutor who explains complex topics in simple terms. Use examples, analogies, and step-by-step explanations to help users understand.",
            
            "creative_writer": "You are a creative writing assistant. Help users with storytelling, character development, plot ideas, and creative expression. Be imaginative and inspiring.",
            
            "technical_expert": "You are a technical expert with deep knowledge in software development, engineering, and technology. Provide precise, detailed technical information and best practices.",
            
            "researcher": "You are a research assistant who helps with information gathering, analysis, and synthesis. Provide well-sourced, objective information and help structure research.",
            
            "business_advisor": "You are a business advisor with expertise in strategy, operations, and management. Provide practical business insights and actionable recommendations.",
            
            "casual_chat": "You are a friendly conversational partner. Engage in casual, natural conversation while being helpful and interesting."
        }
    
    def create_session(
        self,
        persona: str = "assistant",
        client_name: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a new chat session.
        
        Args:
            persona (str): Persona to use for the session
            client_name (Optional[str]): LLM client to use
            custom_system_prompt (Optional[str]): Custom system prompt (overrides persona)
            
        Returns:
            str: Session ID
        """
        if persona not in self.personas and not custom_system_prompt:
            raise ValidationError(
                f"Unknown persona '{persona}'. Available: {list(self.personas.keys())}",
                field="persona",
                value=persona
            )
        
        system_prompt = custom_system_prompt or self.personas.get(persona)
        client_name = client_name or self.llm_manager.default_client
        
        session = ChatSession(
            client_name=client_name,
            llm_manager=self.llm_manager,
            system_prompt=system_prompt
        )
        
        self.sessions[session.session_id] = session
        
        logger.info(f"Created chat session {session.session_id} with {persona} persona")
        return session.session_id
    
    def send_message(self, session_id: str, message: str) -> Dict[str, Any]:
        """
        Send a message to a specific session.
        
        Args:
            session_id (str): Session ID
            message (str): User message
            
        Returns:
            Dict[str, Any]: Response from the session
        """
        if session_id not in self.sessions:
            raise ValidationError(f"Session '{session_id}' not found", field="session_id")
        
        return self.sessions[session_id].send_message(message)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary for a specific session."""
        if session_id not in self.sessions:
            raise ValidationError(f"Session '{session_id}' not found", field="session_id")
        
        return self.sessions[session_id].get_conversation_summary()
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with their summaries."""
        return [
            {
                "session_id": session_id,
                **session.get_conversation_summary()
            }
            for session_id, session in self.sessions.items()
        ]
    
    def close_session(self, session_id: str, export_path: Optional[str] = None) -> Optional[str]:
        """
        Close a session and optionally export it.
        
        Args:
            session_id (str): Session ID to close
            export_path (Optional[str]): Path to export conversation
            
        Returns:
            Optional[str]: Export file path if exported
        """
        if session_id not in self.sessions:
            raise ValidationError(f"Session '{session_id}' not found", field="session_id")
        
        session = self.sessions[session_id]
        exported_path = None
        
        if export_path or session.message_count > 0:
            exported_path = session.export_conversation(export_path)
        
        del self.sessions[session_id]
        logger.info(f"Closed session {session_id}")
        
        return exported_path


def interactive_chat_demo():
    """Interactive chat demonstration."""
    print("🤖 Interactive Chat Demo")
    print("=" * 50)
    print("Commands:")
    print("  /help - Show help")
    print("  /personas - List available personas")
    print("  /switch <persona> - Switch to a different persona")
    print("  /summary - Show conversation summary")
    print("  /export - Export conversation")
    print("  /quit - Quit the demo")
    print("=" * 50)
    
    # Initialize manager
    manager = LLMManager()
    
    # Setup clients
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        manager.add_client("openai", OpenAIClient(api_key=openai_key))
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        manager.add_client("claude", ClaudeClient(api_key=anthropic_key))
    
    if not manager.list_clients():
        print("❌ No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        return
    
    # Create session manager
    chat_manager = ChatSessionManager(manager)
    
    # Start with default persona
    current_persona = "assistant"
    session_id = chat_manager.create_session(current_persona)
    
    print(f"\n✅ Chat session started with '{current_persona}' persona")
    print(f"Session ID: {session_id}")
    print("\nYou can start chatting now!\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.startswith("/"):
                command_parts = user_input[1:].split()
                command = command_parts[0].lower()
                
                if command == "help":
                    print("\nCommands:")
                    print("  /help - Show this help")
                    print("  /personas - List available personas")
                    print("  /switch <persona> - Switch to a different persona")
                    print("  /summary - Show conversation summary")
                    print("  /export - Export conversation")
                    print("  /quit - Quit the demo")
                    continue
                
                elif command == "personas":
                    print("\nAvailable personas:")
                    for persona in chat_manager.personas.keys():
                        mark = " (current)" if persona == current_persona else ""
                        print(f"  • {persona}{mark}")
                    continue
                
                elif command == "switch":
                    if len(command_parts) < 2:
                        print("Usage: /switch <persona>")
                        continue
                    
                    new_persona = command_parts[1]
                    if new_persona not in chat_manager.personas:
                        print(f"Unknown persona '{new_persona}'. Use /personas to see available options.")
                        continue
                    
                    # Close current session and start new one
                    chat_manager.close_session(session_id)
                    current_persona = new_persona
                    session_id = chat_manager.create_session(current_persona)
                    print(f"✅ Switched to '{current_persona}' persona")
                    continue
                
                elif command == "summary":
                    summary = chat_manager.get_session_summary(session_id)
                    print(f"\n📊 Conversation Summary:")
                    print(f"  Messages: {summary['total_messages']}")
                    print(f"  Duration: {summary['duration_minutes']:.1f} minutes")
                    print(f"  Total tokens: {summary['total_tokens']}")
                    print(f"  Client: {summary['client_used']}")
                    continue
                
                elif command == "export":
                    try:
                        file_path = chat_manager.sessions[session_id].export_conversation()
                        print(f"✅ Conversation exported to: {file_path}")
                    except Exception as e:
                        print(f"❌ Export failed: {str(e)}")
                    continue
                
                elif command == "quit":
                    print("\n👋 Goodbye!")
                    chat_manager.close_session(session_id)
                    break
                
                else:
                    print(f"Unknown command: /{command}. Type /help for available commands.")
                    continue
            
            # Send regular message
            response = chat_manager.send_message(session_id, user_input)
            
            if "error" in response:
                print(f"❌ Error: {response['error']['message']}")
            else:
                print(f"Assistant: {response['assistant_response']}")
                print(f"(Tokens: {response['tokens_used']})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            chat_manager.close_session(session_id)
            break
        except Exception as e:
            error_info = handle_error(e)
            print(f"❌ Unexpected error: {error_info.message}")


def automated_chat_demo():
    """Automated chat demonstration with different personas."""
    print("\n🤖 Automated Chat Demo")
    print("=" * 50)
    
    # Initialize manager
    manager = LLMManager()
    
    # Setup at least one client
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        manager.add_client("openai", OpenAIClient(api_key=openai_key))
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        manager.add_client("claude", ClaudeClient(api_key=anthropic_key))
    
    if not manager.list_clients():
        print("❌ No API keys found. Skipping automated demo.")
        return
    
    chat_manager = ChatSessionManager(manager)
    
    # Test scenarios for different personas
    scenarios = [
        {
            "persona": "tutor",
            "messages": [
                "Can you explain how photosynthesis works?",
                "What are the main steps involved?"
            ]
        },
        {
            "persona": "creative_writer",
            "messages": [
                "I need help developing a character for my story.",
                "The character is a detective in a futuristic city."
            ]
        },
        {
            "persona": "technical_expert",
            "messages": [
                "What's the difference between REST and GraphQL APIs?",
                "When would you choose one over the other?"
            ]
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n--- Scenario {i}: {scenario['persona'].title()} ---")
        
        # Create session
        session_id = chat_manager.create_session(scenario["persona"])
        
        # Send messages
        for message in scenario["messages"]:
            print(f"\nUser: {message}")
            
            response = chat_manager.send_message(session_id, message)
            
            if "error" in response:
                print(f"Error: {response['error']['message']}")
            else:
                # Show truncated response
                assistant_response = response["assistant_response"]
                if len(assistant_response) > 200:
                    assistant_response = assistant_response[:200] + "..."
                
                print(f"Assistant: {assistant_response}")
                print(f"(Tokens: {response['tokens_used']})")
        
        # Show summary
        summary = chat_manager.get_session_summary(session_id)
        print(f"\nSession Summary: {summary['total_messages']} messages, {summary['total_tokens']} tokens")
        
        # Close session
        chat_manager.close_session(session_id)


def main():
    """Main function to run chat session examples."""
    try:
        print("💬 Chat Session Example")
        print("=" * 60)
        
        # Run automated demo first
        automated_chat_demo()
        
        # Ask if user wants interactive demo
        if input("\nWould you like to try the interactive chat demo? (y/N): ").strip().lower() == 'y':
            interactive_chat_demo()
        
        print(f"\n{'='*60}")
        print("✅ Chat session example completed!")
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