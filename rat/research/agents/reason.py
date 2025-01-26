"""
Reasoning agent for analyzing research content using DeepSeek (deepseek-reasoner).
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

@dataclass
class AnalysisTask:
    """
    Represents a content analysis task.
    
    Attributes:
        content: The textual content to analyze
        priority: Analysis priority (0-1)
        rationale: Why this analysis is needed
        chunk_index: If chunked, the index of this chunk
    """
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Agent responsible for analyzing content using the DeepSeek API (deepseek-reasoner).
    
    - Splits content into chunks if exceeding 64k tokens
    - Runs parallel analysis with multiple deepseek calls
    - Merges results and can produce further decisions (SEARCH, EXPLORE) if needed
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning agent.
        
        Args:
            config: Optional configuration parameters
        """
        super().__init__("reason", config)
        
        # Initialize DeepSeek client
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        # Configuration
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        # We enforce the official 64k limit for deepseek reasoning calls
        self.max_tokens_per_call = 64000
        self.min_priority = self.config.get("min_priority", 0.3)
        
        # Tracking
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Primary entry point for making decisions about next research steps.
        The ReasoningAgent is now the primary driver of research, deciding:
        1. What searches to run
        2. Which URLs to explore
        3. When to analyze content
        4. When to terminate research
        """
        decisions = []
        
        # 1. If no search results yet, start with a broad search
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results:
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.SEARCH,
                    priority=1.0,
                    context={
                        "query": context.initial_question,
                        "rationale": "Initial broad search for the research question"
                    },
                    rationale="Starting research with a broad search query"
                )
            )
            return decisions
        
        # 2. Process any unanalyzed content (from search or exploration)
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        unprocessed_explored = [
            item for item in explored_content
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        # Create REASON decisions for unprocessed content
        for item in unprocessed_search + unprocessed_explored:
            if item.priority < self.min_priority:
                continue
                
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.REASON,
                    priority=0.9,
                    context={
                        "content": item.content,
                        "content_type": item.content_type.value,
                        "item_id": item.id
                    },
                    rationale=f"Analyze new {item.content_type.value} content"
                )
            )
        
        # 3. Check existing analysis to identify knowledge gaps
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        combined_analysis = " ".join(str(a.content) for a in analysis_items)
        
        # Example knowledge gap checks (customize based on your needs):
        gaps = self._identify_knowledge_gaps(
            context.initial_question,
            combined_analysis
        )
        
        # Create new SEARCH or EXPLORE decisions for gaps
        for gap in gaps:
            if gap["type"] == "search":
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=0.8,
                        context={
                            "query": gap["query"],
                            "rationale": gap["rationale"]
                        },
                        rationale=f"Fill knowledge gap: {gap['rationale']}"
                    )
                )
            elif gap["type"] == "explore":
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=0.75,
                        context={
                            "url": gap["url"],
                            "rationale": gap["rationale"]
                        },
                        rationale=f"Explore URL for more details: {gap['rationale']}"
                    )
                )
        
        # 4. Check if we should terminate research
        if self._should_terminate(context):
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.TERMINATE,
                    priority=1.0,
                    context={},
                    rationale="Research question appears to be sufficiently answered"
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
        return decision.decision_type == DecisionType.REASON
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a reasoning decision using deepseek-reasoner.
        Potentially split content into multiple chunks and run them in parallel, then merge.
        """
        start_time = time.time()
        success = False
        results = {}
        
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
            
            # Convert content to string for token counting
            content_str = str(content)
            tokens_estimated = len(content_str) // 4
            
            if tokens_estimated > self.max_tokens_per_call:
                # Chunked parallel analysis
                chunk_results = self._parallel_analyze_content(content_str, content_type)
                combined = self._combine_chunk_results(chunk_results)
                results = combined
            else:
                # Single chunk
                single_result = self._analyze_content_chunk(content_str, content_type)
                results = single_result
            
            success = bool(results.get("analysis"))
            
            # Tag the original content item as "analyzed_by_reasoner"
            decision.context["analyzed_by_reasoner"] = True
            results["analyzed_item_id"] = item_id
            
            if success:
                rprint(f"[green]ReasoningAgent: Analysis completed for content type '{content_type}'[/green]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
                
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            results = {
                "error": str(e),
                "analysis": "",
                "insights": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
        
        return results
        
    def _parallel_analyze_content(self, content: str, content_type: str) -> List[Dict[str, Any]]:
        """
        Splits large text into ~64k token chunks, then spawns parallel requests to deepseek-reasoner.
        """
        words = content.split()
        chunk_size_words = self.max_tokens_per_call * 4  # approximate word limit
        chunks = []
        
        idx = 0
        while idx < len(words):
            chunk = words[idx: idx + chunk_size_words]
            chunks.append(" ".join(chunk))
            idx += chunk_size_words
        
        chunk_results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
            future_map = {
                executor.submit(self._analyze_content_chunk, chunk, f"{content_type}_chunk_{i}"): i
                for i, chunk in enumerate(chunks)
            }
            for future in as_completed(future_map):
                chunk_index = future_map[future]
                try:
                    res = future.result()
                    res["chunk_index"] = chunk_index
                    chunk_results.append(res)
                except Exception as e:
                    rprint(f"[red]Error in chunk {chunk_index}: {e}[/red]")
        
        return chunk_results
        
    def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Calls deepseek to analyze a single chunk of content.
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advanced deepseek-reasoner model. "
                            "Analyze the content for key insights, patterns, or relevant facts. "
                            "You can elaborate on major findings."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Content type: {content_type}\n\nContent:\n{content}"
                    }
                ],
                temperature=0.7
            )
            analysis = response.choices[0].message.content
            
            return {
                "analysis": analysis,
                "insights": self._extract_insights(analysis),
                "content_type": content_type
            }
        except Exception as e:
            rprint(f"[red]DeepSeek API error: {str(e)}[/red]")
            return {
                "analysis": "",
                "insights": [],
                "content_type": content_type,
                "error": str(e)
            }
            
    def _extract_insights(self, analysis: str) -> List[str]:
        """
        Basic extraction of bullet points or numbered lines from the analysis text.
        """
        lines = analysis.split("\n")
        insights = []
        
        for line in lines:
            line = line.strip()
            # Look for bullet or numbered items
            if (line.startswith("-") or line.startswith("*") or 
                line.startswith("•") or (line[:2].isdigit() and line[2] in [".", ")"])):
                insights.append(line.lstrip("-*•").strip())
        
        return insights
        
    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merges analysis from multiple chunk results into a single structure.
        """
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        
        combined_analysis = "\n\n".join([res["analysis"] for res in sorted_chunks if res["analysis"]])
        combined_insights = []
        for res in sorted_chunks:
            combined_insights.extend(res.get("insights", []))
        
        # Remove duplicates
        unique_insights = list(dict.fromkeys(combined_insights))
        
        return {
            "analysis": combined_analysis,
            "insights": unique_insights,
            "chunk_count": len(chunk_results)
        }

    def _identify_knowledge_gaps(
        self,
        question: str,
        current_analysis: str
    ) -> List[Dict[str, Any]]:
        """
        Analyze current findings to identify what information is still missing.
        Uses deepseek to compare the question against current analysis.
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an advanced research analyst. "
                            "Compare the research question against current findings "
                            "to identify what key information is still missing. "
                            "Return a JSON array of knowledge gaps, where each gap has:"
                            "- type: 'search' or 'explore'"
                            "- query: search query to fill gap (if type=search)"
                            "- url: URL to explore (if type=explore)"
                            "- rationale: why this information is needed"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nCurrent Analysis:\n{current_analysis}"
                    }
                ],
                temperature=0.7
            )
            
            # Parse the response into a list of gaps
            try:
                import json
                gaps = json.loads(response.choices[0].message.content)
                if isinstance(gaps, list):
                    return gaps
            except:
                # If parsing fails, return an empty list
                return []
                
        except Exception as e:
            rprint(f"[red]Error identifying knowledge gaps: {e}[/red]")
            return []
    
    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Decide if we have sufficient information to answer the question.
        """
        # Get all analysis content
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if not analysis_items:
            return False
        
        # Combine all analysis
        combined_analysis = " ".join(str(a.content) for a in analysis_items)
        
        try:
            # Ask deepseek if we have enough info
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research completion analyst. "
                            "Evaluate if we have sufficient information to answer "
                            "the original question. Consider:"
                            "1. Are all key aspects addressed?"
                            "2. Is the information detailed enough?"
                            "3. Are there any major gaps?"
                            "Return a JSON with:"
                            "- complete: true/false"
                            "- confidence: 0-1"
                            "- missing: list of missing elements (if any)"
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {context.initial_question}\n\nFindings:\n{combined_analysis}"
                    }
                ],
                temperature=0.7
            )
            
            try:
                result = json.loads(response.choices[0].message.content)
                return result.get("complete", False) and result.get("confidence", 0) >= 0.8
            except:
                return False
                
        except Exception as e:
            rprint(f"[red]Error checking termination: {e}[/red]")
            return False
