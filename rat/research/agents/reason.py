"""
Reasoning agent for analyzing research content using the Gemini 2.0 Flash Thinking model.
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.

Key responsibilities:
1. Analyzing content using Gemini
2. Deciding which URLs to explore
3. Identifying knowledge gaps
4. Determining when to terminate research
"""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
import os
import json
import logging
import re
from urllib.parse import urlparse
import threading
from time import sleep

import google.generativeai as genai

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

# Get loggers
logger = logging.getLogger(__name__)
api_logger = logging.getLogger('api.gemini')

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
    Reasoning agent for analyzing research content using the Gemini 2.0 Flash Thinking model.
    Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
    Supports parallel processing of content analysis and decision making.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the reasoning agent."""
        super().__init__("reason", config)
        
        # Configure Gemini model
        self.generation_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_output_tokens": 50000,
            "response_mime_type": "text/plain"
        }
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"

        # Initialize the model
        self._model = None

        # Initialize logger
        self.logger = logging.getLogger(__name__)

        # Configure the Gemini client
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])

        # We'll chunk input up to a 1,048,576 token estimate
        self.max_tokens_per_call = 1048576  
        self.max_output_tokens = 65536
        self.request_timeout = self.config.get("gemini_timeout", 180)

        # For concurrency chunk splitting
        self.chunk_margin = 5000   # Safety margin
        
        # Add max_parallel_tasks from config
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)

        # Priority thresholds
        self.min_priority = self.config.get("min_priority", 0.3)
        self.min_url_relevance = self.config.get("min_url_relevance", 0.6)

        # URL tracking
        self.explored_urls: Set[str] = set()
        
        # Flash-fix rate limiting
        self.flash_fix_rate_limit = self.config.get("flash_fix_rate_limit", 10)
        self._flash_fix_last_time = 0.0
        self._flash_fix_lock = threading.Lock()
        
        logger.info("ReasoningAgent initialized to use Gemini model: %s", self.model_name)
        
        # Tracking
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
    @property
    def model(self):
        """Lazy initialization of the Gemini model."""
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=self.generation_config
            )
        return self._model

    def _get_model(self) -> genai.GenerativeModel:
        """Get or create the Gemini model instance."""
        return self.model  # Use the property

    def _enforce_flash_fix_limit(self):
        """
        Ensure we do not exceed flash_fix_rate_limit requests per minute.
        """
        if self.flash_fix_rate_limit <= 0:
            return
            
        with self._flash_fix_lock:
            current_time = time.time()
            elapsed = current_time - self._flash_fix_last_time
            min_interval = 60.0 / self.flash_fix_rate_limit
            
            if elapsed < min_interval:
                sleep_time = min_interval - elapsed
                time.sleep(sleep_time)
                self.metrics["rate_limit_delays"] += 1
            
            self._flash_fix_last_time = time.time()
            
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Primary entry point for making decisions about next research steps.
        Now also responsible for URL exploration decisions.
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
        
        # 2. Process unvisited URLs from search results
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        explored_urls = {
            item.content.get("url", "") for item in explored_content
            if isinstance(item.content, dict)
        }
        self.explored_urls.update(explored_urls)
        
        # Collect unvisited URLs from search results
        unvisited_urls = set()
        for result in search_results:
            if isinstance(result.content, dict):
                urls = result.content.get("urls", [])
                unvisited_urls.update(
                    url for url in urls 
                    if url not in self.explored_urls
                )
        
        # Filter and prioritize URLs
        relevant_urls = self._filter_relevant_urls(
            list(unvisited_urls), 
            context
        )
        
        # Create EXPLORE decisions for relevant URLs
        for url, relevance in relevant_urls:
            if relevance >= self.min_url_relevance:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.EXPLORE,
                        priority=relevance,
                        context={
                            "url": url,
                            "relevance": relevance,
                            "rationale": "URL deemed relevant to research question"
                        },
                        rationale=f"URL relevance score: {relevance:.2f}"
                    )
                )
        
        # 3. Process any unanalyzed content
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
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
        
        # 4. Check for knowledge gaps - but only if we have context
        # Build combined text from search results, explored content, and existing analysis
        search_text = "\n".join(
            str(item.content) 
            for item in search_results 
            if isinstance(item.content, str)
        )
        explored_text = "\n".join(
            str(item.content) 
            for item in explored_content 
            if isinstance(item.content, str)
        )
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_text = "\n".join(
            item.content.get("analysis", "")
            if isinstance(item.content, dict) else str(item.content)
            for item in analysis_items
        )

        combined_analysis = f"{search_text}\n\n{explored_text}\n\n{analysis_text}".strip()
        
        if combined_analysis:  # Only check for gaps if we have some real context
            gaps = self._identify_knowledge_gaps(
                context.initial_question,
                combined_analysis
            )
            
            # Filter out any gaps with placeholders
            filtered_gaps = []
            for gap in gaps:
                query_str = gap.get("query", "")
                url_str = gap.get("url", "")
                # Skip if the LLM hallucinated placeholders
                if "[" in query_str or "]" in query_str or "[" in url_str or "]" in url_str:
                    self.logger.warning(f"Skipping gap with placeholders: {gap}")
                    continue
                filtered_gaps.append(gap)
            
            # Create new SEARCH or EXPLORE decisions for filtered gaps
            for gap in filtered_gaps:
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
                    if gap["url"] not in self.explored_urls:
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
        else:
            # No real context yet, skip knowledge gap detection
            self.logger.info("Skipping knowledge gap analysis because we have no contextual text.")
        
        # 5. Check if we should terminate
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
        Only stores the actual analysis text, not any suggestions or placeholders.
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
            
            # We only consider it successful if we got actual analysis text
            success = bool(results.get("analysis", "").strip())
            
            # Tag the original content item as "analyzed_by_reasoner"
            decision.context["analyzed_by_reasoner"] = True
            
            # Package only the analysis-related content, no suggestions or placeholders
            final_results = {
                "analysis": results.get("analysis", ""),
                "insights": results.get("insights", []),
                "analyzed_item_id": item_id
            }
            
            if success:
                rprint(f"[green]ReasoningAgent: Analysis completed for content type '{content_type}'[/green]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
                
            return final_results
            
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            return {
                "error": str(e),
                "analysis": "",
                "insights": []
            }
            
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
        
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
        We'll store only the final 'analysis' portion, ignoring any next-step JSON the LLM appends.
        """
        # Enforce main rate limit
        self._enforce_rate_limit()
        
        # Create a chat session
        model = self._get_model()
        chat_session = model.start_chat(history=[])

        # We explicitly instruct the model to avoid placeholders and next steps
        prompt = (
            "You are an advanced reasoning model (Gemini 2.0 Flash Thinking). "
            "Analyze the following text for key insights, patterns, or relevant facts. "
            "Provide ONLY factual analysis and insights. "
            "DO NOT include any placeholders (like [company name] or [person]).\n"
            "DO NOT suggest next steps or additional searches.\n"
            "DO NOT output JSON or structured data.\n\n"
            f"CONTENT:\n{content}\n\n"
            "Please provide your analysis below (plain text only):"
        )
        
        response = chat_session.send_message(prompt)
        analysis_text = response.text.strip()
        
        # Extract insights from the analysis text
        insights = self._extract_insights(analysis_text)
        
        return {
            "analysis": analysis_text,
            "insights": insights
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
        Only combines actual analysis text and insights, not suggestions or placeholders.
        """
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        
        # Only combine actual analysis text, not any suggestions
        combined_analysis = "\n\n".join([
            res["analysis"] for res in sorted_chunks 
            if res.get("analysis", "").strip()
        ])
        
        # Combine insights, removing duplicates
        combined_insights = []
        for res in sorted_chunks:
            insights = res.get("insights", [])
            combined_insights.extend(insight for insight in insights if insight.strip())
        
        # Remove duplicates while preserving order
        unique_insights = list(dict.fromkeys(combined_insights))
        
        return {
            "analysis": combined_analysis,
            "insights": unique_insights,
            "chunk_count": len(chunk_results)
        }

    def _fix_json_with_gemini_exp(self, malformed_json_str: str) -> str:
        """
        Attempts to fix malformed JSON by calling a second Gemini model
        (gemini-2.0-flash-exp) that is instructed to output valid JSON only.
        """
        # Enforce flash-fix rate limit
        self._enforce_flash_fix_limit()
        
        # We can define a separate model name for the "JSON-fixer" step.
        fix_model_name = "gemini-2.0-flash-exp"
        
        # Provide generation settings suitable for a JSON-fixing prompt.
        # Typically you want a low creativity / temperature so it just
        # cleans up the JSON and doesn't hallucinate extra structure.
        fix_generation_config = {
            "temperature": 0.0,
            "top_p": 0.0,
            "top_k": 1,
            "max_output_tokens": 1024,
        }
        
        fix_model = genai.GenerativeModel(
            model_name=fix_model_name,
            generation_config=fix_generation_config
        )
        
        chat_session = fix_model.start_chat(history=[])
        
        # In the prompt, we strongly instruct that it must return valid JSON only
        # with no extra commentary or text.
        prompt = (
            "You are an expert at transforming malformed JSON into correct JSON.\n\n"
            "Your job is to fix any invalid or malformed JSON so it can be parsed.\n"
            "Output *only* the corrected JSON—no extra commentary or text.\n\n"
            "Here is the malformed JSON:\n"
            f"{malformed_json_str}\n\n"
            "Now return only valid JSON."
        )
        
        response = chat_session.send_message(prompt)
        return response.text

    def _call_gemini(self, prompt: str, context: str = "") -> str:
        """Helper method to call Gemini API with logging."""
        api_logger.info(f"Gemini API Request - Context length: {len(context)}")
        api_logger.debug(f"Prompt: {prompt}")
        
        try:
            model = self._get_model()
            chat = model.start_chat(history=[])
            response = chat.send_message(prompt)
            
            api_logger.debug(f"Response: {response.text}")
            return response.text.strip()
            
        except Exception as e:
            api_logger.error(f"Gemini API error: {str(e)}")
            raise

    def _identify_knowledge_gaps(self, question: str, current_analysis: str) -> List[Dict[str, Any]]:
        """
        Identify missing information and suggest next steps.
        Skip any suggestions containing placeholder text in brackets.
        
        Args:
            question: The research question being investigated
            current_analysis: Combined text from search results, explored content, and analysis
            
        Returns:
            List of gap dictionaries, each containing type (search/explore), query/url, and rationale
        """
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "identify specific missing information and suggest concrete next steps.\n\n"
            "IMPORTANT RULES:\n"
            "1. DO NOT use placeholders like [company name] or [person] - only suggest specific, concrete queries/URLs\n"
            "2. Base your suggestions ONLY on the actual content provided\n"
            "3. If you can't identify specific gaps, return an empty array\n"
            "4. Each suggestion must be actionable and clearly related to the research question\n\n"
            f"RESEARCH QUESTION: {question}\n\n"
            f"CURRENT ANALYSIS:\n{current_analysis}\n\n"
            "Respond with a JSON array of gaps in this format:\n"
            '[{"type": "search"|"explore", "query"|"url": "specific text", "rationale": "why needed"}]\n'
            "Return ONLY the JSON array, no other text."
        )
        
        try:
            content_str = self._call_gemini(prompt, current_analysis)
            if not content_str:
                return []
            
            # Clean up the response
            content_str = self._clean_json_response(content_str)
            
            try:
                gaps = json.loads(content_str)
                if not isinstance(gaps, list):
                    return []
                
                # Filter out any gaps with placeholders or invalid structure
                filtered_gaps = []
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                        
                    # Validate required fields
                    if "type" not in gap or gap["type"] not in ["search", "explore"]:
                        continue
                    
                    # Get the relevant field based on type
                    content_field = "query" if gap["type"] == "search" else "url"
                    if content_field not in gap or "rationale" not in gap:
                        continue
                    
                    content_str = gap[content_field]
                    rationale_str = gap["rationale"]
                    
                    # Skip if any field contains placeholders
                    if (
                        "[" in content_str or "]" in content_str or 
                        "[" in rationale_str or "]" in rationale_str
                    ):
                        self.logger.warning(
                            f"Skipping gap with placeholders: {content_str}"
                        )
                        continue
                    
                    filtered_gaps.append(gap)
                
                return filtered_gaps
                
            except json.JSONDecodeError:
                self.logger.error("Failed to parse knowledge gaps JSON response")
                return []
                
        except Exception as e:
            self.logger.error(f"Error identifying knowledge gaps: {str(e)}")
            return []

    def _clean_json_response(self, content: str) -> str:
        """Helper to clean up JSON responses from the model."""
        content = content.strip()
        
        # Remove markdown code blocks if present
        if content.startswith("```"):
            start_idx = content.find("\n", content.find("```")) + 1
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
            else:
                content = content.replace("```", "").strip()
            
        # Remove any "json" language identifier
        content = content.replace("json", "").strip()
        
        return content
    
    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Use Gemini to decide if the research question has been sufficiently answered.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if len(analysis_items) < 3:  # Need some minimum analysis
            return False

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
        
        try:
            answer = self._call_gemini(prompt, combined_analysis)
            return answer.strip().upper() == "YES"
        except Exception:
            return False

    def _filter_relevant_urls(
        self, 
        urls: List[str], 
        context: ResearchContext
    ) -> List[tuple[str, float]]:
        """
        Filter and score URLs based on relevance to the research question.
        Returns: List of (url, relevance_score) tuples.
        """
        if not urls:
            return []
            
        # Batch URLs for efficient processing
        batch_size = 5
        url_batches = [urls[i:i + batch_size] for i in range(0, len(urls), batch_size)]
        relevant_urls = []
        
        for batch in url_batches:
            prompt = (
                "You are an expert at determining URL relevance for research questions.\n"
                "For each URL, analyze its potential relevance to the research question "
                "and provide a relevance score between 0.0 and 1.0.\n\n"
                f"RESEARCH QUESTION: {context.initial_question}\n\n"
                "URLs to evaluate:\n"
            )
            
            for url in batch:
                domain = urlparse(url).netloc
                path = urlparse(url).path
                prompt += f"- {domain}{path}\n"
                
            prompt += (
                "\nRespond with a JSON array of scores in this format:\n"
                "[{\"url\": \"...\", \"score\": 0.X, \"reason\": \"...\"}]\n"
                "ONLY return the JSON array, no other text."
            )
            
            try:
                content = self._call_gemini(prompt)
                
                # Clean markdown formatting if present
                if content.startswith("```"):
                    content = content[content.find("\n")+1:content.rfind("```")].strip()
                content = content.replace("json", "").strip()
                
                scores = json.loads(content)
                
                for score_obj in scores:
                    url = score_obj["url"]
                    score = float(score_obj["score"])
                    relevant_urls.append((url, score))
                    
            except Exception as e:
                logger.error(f"Error scoring URLs: {str(e)}")
                # Fall back to basic keyword matching for this batch
                for url in batch:
                    relevance = self._basic_url_relevance(url, context.initial_question)
                    relevant_urls.append((url, relevance))
                
        return sorted(relevant_urls, key=lambda x: x[1], reverse=True)

    def _basic_url_relevance(self, url: str, question: str) -> float:
        """
        Basic fallback method for URL relevance when LLM scoring fails.
        Returns a score between 0.0 and 1.0.
        """
        # Extract keywords from question
        keywords = set(re.findall(r'\w+', question.lower()))
        
        # Parse URL components
        parsed = urlparse(url)
        domain_parts = parsed.netloc.lower().split('.')
        path_parts = parsed.path.lower().split('/')
        
        # Count keyword matches in domain and path
        domain_matches = sum(1 for part in domain_parts if part in keywords)
        path_matches = sum(1 for part in path_parts if part in keywords)
        
        # Weight domain matches more heavily than path matches
        score = (domain_matches * 0.6 + path_matches * 0.4) / max(len(keywords), 1)
        return min(max(score, 0.0), 1.0)
