"""
Reasoning agent for analyzing research content using the Gemini 2.0 Flash Thinking model.
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
import os
import json
import logging

import google.generativeai as genai

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)

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
    Agent responsible for analyzing content using the Gemini 2.0 Flash Thinking model
    and for driving the overall research flow.
    
    - Splits content into chunks if exceeding the input token limit (1,048,576).
    - Runs parallel analysis with multiple calls to Gemini if needed.
    - Merges results and can produce further decisions (SEARCH, EXPLORE) if needed.
    - Potentially decides when to TERMINATE.
    """
    
    def __init__(self, config=None):
        super().__init__("reason", config)

        # Configure the Gemini client
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # We'll store these generation settings to pass each call
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 65536,       # Up to 65k tokens out
            "response_mime_type": "text/plain"
        }
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"

        # We'll chunk input up to a 1,048,576 token estimate
        self.max_tokens_per_call = 1048576  
        self.max_output_tokens = 65536
        self.request_timeout = self.config.get("gemini_timeout", 180)

        # For concurrency chunk splitting
        self.chunk_margin = 5000   # Safety margin

        # Priority thresholds
        self.min_priority = self.config.get("min_priority", 0.3)

        logger.info("ReasoningAgent initialized to use Gemini model: %s", self.model_name)
        
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
        Execute a reasoning decision using Gemini.
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
        Splits large text into ~64k token chunks, then spawns parallel requests to Gemini.
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
        Calls Gemini with a single-turn prompt to analyze the content.
        """
        # Create a chat session
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        chat_session = model.start_chat(history=[])

        # We treat it as: "Analyze the following content for key insights"
        prompt = (
            "You are an advanced reasoning model (Gemini 2.0 Flash Thinking). "
            "Analyze the following text for key insights, patterns, or relevant facts. "
            "Elaborate on major findings.\n\n"
            f"CONTENT:\n{content}\n\n"
            "Please provide your analysis below:"
        )
        response = chat_session.send_message(prompt)
        return {
            "analysis": response.text,
            "insights": self._extract_insights(response.text)
        }

    def _extract_insights(self, analysis_text: str) -> List[str]:
        """
        Basic extraction of bullet points or lines from the analysis text.
        """
        lines = analysis_text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if (
                line.startswith("-") or line.startswith("*") or 
                line.startswith("•") or (len(line) > 2 and line[:2].isdigit())
            ):
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

    def _identify_knowledge_gaps(self, question: str, current_analysis: str) -> List[Dict[str, Any]]:
        """
        Use Gemini to identify gaps in current knowledge and suggest next steps.
        """
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        chat_session = model.start_chat(history=[])

        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "identify gaps in knowledge and suggest specific next steps (either search queries or URLs to explore).\n\n"
            f"QUESTION: {question}\n\n"
            f"CURRENT ANALYSIS:\n{current_analysis}\n\n"
            "Please provide your suggestions in valid JSON format with this structure:\n"
            "[{\"type\": \"search\"|\"explore\", \"query\"|\"url\": \"...\", \"rationale\": \"...\"}]\n"
            "No extra text, only the JSON array."
        )
        response = chat_session.send_message(prompt)
        content_str = response.text.strip()

        try:
            gaps = json.loads(content_str)
            if not isinstance(gaps, list):
                raise ValueError("Expected a JSON list.")
            return gaps
        except Exception as e:
            logger.error("Error parsing knowledge gaps from Gemini: %s", e)
            return []
    
    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Use Gemini to decide if the research question has been sufficiently answered.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if len(analysis_items) < 3:  # Need some minimum analysis
            return False

        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config
        )
        chat_session = model.start_chat(history=[])

        # Combine all analysis
        combined_analysis = "\n".join(
            str(item.content.get("analysis", "")) for item in analysis_items
        )

        prompt = (
            "You are an advanced research assistant. Given a research question and the current analysis, "
            "determine if the question has been sufficiently answered.\n\n"
            f"QUESTION: {context.initial_question}\n\n"
            f"CURRENT ANALYSIS:\n{combined_analysis}\n\n"
            "Respond with a single word: YES if the question is sufficiently answered, NO if not."
        )
        response = chat_session.send_message(prompt)
        answer = response.text.strip().upper()

        return answer == "YES"
