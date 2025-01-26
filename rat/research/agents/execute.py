"""
Execution agent for generating code and structured output using Claude.
Handles code generation, data formatting, and output validation.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import anthropic
import os
import json

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class ExecutionTask:
    """
    Represents a code or structured output task.
    
    Attributes:
        task_type: Type of task (code/json/etc)
        content: Content to process
        priority: Task priority (0-1)
        rationale: Why this task is needed
    """
    task_type: str
    content: str
    priority: float
    rationale: str
    timestamp: float = time.time()

class ExecutionAgent(BaseAgent):
    """
    Agent responsible for generating code and structured output using Claude.
    
    Handles code generation, data formatting, and output validation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution agent.
        
        Args:
            config: Optional configuration parameters
        """
        super().__init__("execute", config)
        
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        
        # Configuration
        self.model = self.config.get("model", "claude-3-5-sonnet-20241022")
        self.min_priority = self.config.get("min_priority", 0.3)
        self.max_retries = self.config.get("max_retries", 2)
        
        # Tracking
        self.execution_tasks: Dict[str, ExecutionTask] = {}
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Analyze the research context and decide on execution tasks.
        
        Args:
            context: Current research context
            
        Returns:
            List of execution-related decisions
        """
        decisions = []
        
        # Get analyzed content that might need structured output
        analyzed_content = context.get_content(
            "main",
            ContentType.ANALYSIS
        )
        
        for content in analyzed_content:
            # Check if content needs code generation
            if self._needs_code_generation(content):
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXECUTE,
                        priority=content.priority * 0.9,
                        context={
                            "task_type": "code",
                            "content": content.content,
                            "metadata": content.metadata
                        },
                        rationale="Generate code implementation"
                    )
                )
                
            # Check if content needs JSON formatting
            if self._needs_json_formatting(content):
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXECUTE,
                        priority=content.priority * 0.8,
                        context={
                            "task_type": "json",
                            "content": content.content,
                            "metadata": content.metadata
                        },
                        rationale="Convert analysis to structured JSON"
                    )
                )
                
        return decisions
        
    def can_handle(self, decision: ResearchDecision) -> bool:
        """
        Check if this agent can handle a decision.
        
        Args:
            decision: Decision to evaluate
            
        Returns:
            True if this agent can handle the decision
        """
        return decision.decision_type == DecisionType.EXECUTE
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a code or structured output decision.
        
        Args:
            decision: Execution decision to execute
            
        Returns:
            Generated output and metadata
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            task_type = decision.context["task_type"]
            content = decision.context["content"]
            metadata = decision.context.get("metadata", {})
            
            # Track the task
            task_id = str(len(self.execution_tasks) + 1)
            self.execution_tasks[task_id] = ExecutionTask(
                task_type=task_type,
                content=content,
                priority=decision.priority,
                rationale=decision.rationale
            )
            
            # Execute with retries
            for attempt in range(self.max_retries + 1):
                try:
                    if task_type == "code":
                        results = self._generate_code(content, metadata)
                    elif task_type == "json":
                        results = self._format_json(content, metadata)
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")
                        
                    success = True
                    break
                    
                except Exception as e:
                    if attempt == self.max_retries:
                        raise
                    rprint(f"[yellow]Attempt {attempt + 1} failed: {str(e)}[/yellow]")
                    time.sleep(1)  # Brief delay before retry
                    
            if success:
                rprint(f"[green]{task_type.title()} generation completed[/green]")
            else:
                rprint(f"[yellow]No output generated for {task_type}[/yellow]")
                
        except Exception as e:
            rprint(f"[red]Execution error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "output": "",
                "metadata": {
                    "task_type": decision.context.get("task_type", "unknown")
                }
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            
        return results
        
    def _needs_code_generation(self, content: ContentItem) -> bool:
        """
        Check if content needs code generation.
        
        Args:
            content: Content item to check
            
        Returns:
            True if code generation is needed
        """
        # Look for code-related keywords
        code_indicators = [
            "implementation",
            "code",
            "function",
            "class",
            "algorithm",
            "script"
        ]
        
        text = content.content.lower()
        return any(indicator in text for indicator in code_indicators)
        
    def _needs_json_formatting(self, content: ContentItem) -> bool:
        """
        Check if content needs JSON formatting.
        
        Args:
            content: Content item to check
            
        Returns:
            True if JSON formatting is needed
        """
        # Look for structured data indicators
        json_indicators = [
            "structured",
            "json",
            "data format",
            "schema",
            "key-value",
            "mapping"
        ]
        
        text = content.content.lower()
        return any(indicator in text for indicator in json_indicators)
        
    def _generate_code(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate code using Claude.
        
        Args:
            content: Content to generate code from
            metadata: Additional context
            
        Returns:
            Generated code and metadata
        """
        messages = [{
            "role": "system",
            "content": (
                "You are an expert code generator. Generate clean, efficient, "
                "and well-documented code based on the provided description."
            )
        }, {
            "role": "user",
            "content": f"Generate code for:\n\n{content}"
        }]
        
        response = self.claude_client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=4000
        )
        
        generated_code = response.content[0].text
        
        return {
            "output": generated_code,
            "language": self._detect_language(generated_code),
            "metadata": metadata
        }
        
    def _format_json(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Convert content to structured JSON using Claude.
        
        Args:
            content: Content to convert to JSON
            metadata: Additional context
            
        Returns:
            Formatted JSON and metadata
        """
        messages = [{
            "role": "system",
            "content": (
                "You are an expert at converting unstructured text into clean, "
                "well-structured JSON format."
            )
        }, {
            "role": "user",
            "content": f"Convert to JSON:\n\n{content}"
        }]
        
        response = self.claude_client.messages.create(
            model=self.model,
            messages=messages,
            max_tokens=4000
        )
        
        json_str = response.content[0].text
        
        # Validate JSON
        try:
            json_data = json.loads(json_str)
            return {
                "output": json_data,
                "format": "json",
                "metadata": metadata
            }
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON generated: {str(e)}")
            
    def _detect_language(self, code: str) -> str:
        """
        Detect the programming language of generated code.
        
        Args:
            code: Generated code to analyze
            
        Returns:
            Detected language name
        """
        # TODO: Implement smarter language detection
        # For now, use simple keyword matching
        language_indicators = {
            "python": ["def ", "import ", "class ", "print("],
            "javascript": ["function", "const ", "let ", "var "],
            "java": ["public class", "private ", "void ", "String"],
            "cpp": ["#include", "int main", "std::", "void"]
        }
        
        for lang, indicators in language_indicators.items():
            if any(indicator in code for indicator in indicators):
                return lang
                
        return "unknown"
