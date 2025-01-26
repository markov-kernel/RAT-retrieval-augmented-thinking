"""
RAT Research - Enhanced Retrieval Augmented Thinking with Research Capabilities
This module combines DeepSeek reasoning with Perplexity web search and Jina Reader
for comprehensive research paper generation.
"""

from openai import OpenAI
import os
from dotenv import load_dotenv
from rich import print as rprint
from rich.panel import Panel
from rich.markdown import Markdown
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import time
from typing import Dict, Any

from rat.research.session import ResearchSession

# Load environment variables
load_dotenv()

class ModelChain:
    def __init__(self):
        # Initialize DeepSeek client for reasoning
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Initialize research session
        self.research_session = ResearchSession()
        self.show_reasoning = True
        
    def get_deepseek_reasoning(self, user_input: str) -> str:
        """
        Get reasoning from DeepSeek about the research question.
        
        Args:
            user_input: The research question
            
        Returns:
            Reasoning about the research approach
        """
        start_time = time.time()
        
        if self.show_reasoning:
            rprint("\n[blue]Research Approach Reasoning[/blue]")
        
        try:
            # Send a single message with clear instructions
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{
                    "role": "system",
                    "content": (
                        "You are a research assistant. When given a research question, "
                        "analyze it and explain your detailed approach to finding accurate "
                        "information about it. Break down the research into logical steps "
                        "and explain what specific information you will look for."
                    )
                }, {
                    "role": "user",
                    "content": user_input
                }],
                temperature=0.7,
                stream=True
            )
            
            # Initialize content buffer
            content_buffer = []
            
            # Process the stream
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content_piece = chunk.choices[0].delta.content
                    content_buffer.append(content_piece)
                    if self.show_reasoning:
                        print(content_piece, end="", flush=True)
            
            # Combine all content
            full_content = "".join(content_buffer)

            elapsed_time = time.time() - start_time
            if elapsed_time >= 60:
                time_str = f"{elapsed_time/60:.1f} minutes"
            else:
                time_str = f"{elapsed_time:.1f} seconds"
            rprint(f"\n\n[yellow]Thought for {time_str}[/]")

            if self.show_reasoning:
                print("\n")
            return full_content
            
        except Exception as e:
            rprint(f"\n[red]Error in DeepSeek reasoning: {str(e)}[/red]")
            return ""
            
    def conduct_research(self, question: str) -> None:
        """
        Conduct comprehensive research on a question.
        
        Args:
            question: The research question to investigate
        """
        try:
            # Get initial reasoning about the research approach
            reasoning = self.get_deepseek_reasoning(question)
            
            # Start research session
            self.research_session.start_research(question)
            
            # Conduct initial research using Perplexity
            self.research_session.conduct_initial_research()
            
            # Extract content from discovered sources
            self.research_session.extract_content()
            
            # Generate and display the research paper
            paper = self.research_session.generate_paper()
            self.research_session.display_paper()
            
        except Exception as e:
            rprint(f"\n[red]Error conducting research: {str(e)}[/red]")
            
def main():
    """
    Main entry point for the RAT Research application.
    """
    chain = ModelChain()
    
    # Initialize prompt session with styling
    style = Style.from_dict({
        'prompt': 'orange bold',
    })
    session = PromptSession(style=style)
    
    # Display welcome message
    rprint(Panel.fit(
        "[bold cyan]RAT Research - Enhanced Retrieval Augmented Thinking[/]",
        title="[bold cyan]🧠 RAT Research[/]",
        border_style="cyan"
    ))
    
    rprint("[yellow]Commands:[/]")
    rprint(" • Type [bold red]'quit'[/] to exit")
    rprint(" • Type [bold magenta]'reasoning'[/] to toggle reasoning visibility")
    rprint(" • Type [bold magenta]'export'[/] to export the last generated paper")
    rprint(" • Type [bold magenta]'clear'[/] to clear research history\n")
    
    while True:
        try:
            user_input = session.prompt("\nResearch Question: ", style=style).strip()
            
            if user_input.lower() == 'quit':
                print("\nGoodbye! 👋")
                break
                
            elif user_input.lower() == 'clear':
                chain = ModelChain()  # Reset the chain
                rprint("\n[magenta]Research history cleared![/]\n")
                continue
                
            elif user_input.lower() == 'reasoning':
                chain.show_reasoning = not chain.show_reasoning
                status = "visible" if chain.show_reasoning else "hidden"
                rprint(f"\n[magenta]Reasoning process is now {status}[/]\n")
                continue
                
            elif user_input.lower() == 'export':
                try:
                    filepath = chain.research_session.export_paper()
                    rprint(f"\n[green]Paper exported to: {filepath}[/green]")
                except ValueError as e:
                    rprint(f"\n[yellow]{str(e)}[/yellow]")
                continue
                
            # Conduct research on the question
            chain.conduct_research(user_input)
            
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
            
if __name__ == "__main__":
    main() 