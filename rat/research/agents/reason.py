"""
Reasoning agent for analyzing research content using DeepSeek (deepseek-reasoner).
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.

Enhancements:
1. Improved chunk splitting to avoid breaking in the middle of words.
2. Automated consolidation of multiple chunk analyses for the same source content
   into a single, unified analysis entry (to avoid fragmented or disjoint results).
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os
import json
import logging

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)

@dataclass
class AnalysisTask:
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()

class ReasoningAgent(BaseAgent):
    """
    Agent responsible for analyzing content using the DeepSeek API (deepseek-reasoner).
    Splits content if it exceeds 64k tokens, calls the API, merges results, etc.
    Also drives the research forward (search, explore, or terminate).
    
    Enhancements:
    - Automatic consolidation of chunk analyses.
    - More careful content chunking to avoid splitting within words.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        
        self.deepseek_client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.max_tokens_per_call = 64000
        self.max_output_tokens = 8192
        self.request_timeout = self.config.get("deepseek_timeout", 180)
        
        self.min_priority = self.config.get("min_priority", 0.3)
        self.analysis_tasks: Dict[str, AnalysisTask] = {}
        
        # Store partial chunked analyses for each (item_id, total_chunks) so we can unify them
        self.chunked_analyses: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

        logger.info(
            "ReasoningAgent initialized with max_parallel_tasks=%d, max_tokens_per_call=%d, max_output_tokens=%d",
            self.max_parallel_tasks,
            self.max_tokens_per_call,
            self.max_output_tokens
        )

    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        Evaluate the current context to decide if new searches, explorations, or deeper reasoning is needed.
        If no search results are available, generate initial search queries.
        Otherwise, look for unprocessed content that needs chunked analysis,
        identify knowledge gaps, and possibly produce a TERMINATE decision.
        """
        decisions = []
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results:
            # No search results => generate initial queries
            initial_queries = self._generate_initial_queries(context.initial_question)
            for query_info in initial_queries:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.SEARCH,
                        priority=1.0,
                        context={
                            "query": query_info["query"],
                            "rationale": query_info["rationale"]
                        },
                        rationale=f"Parallel initial search: {query_info['rationale']}"
                    )
                )
            return decisions
        
        # Identify unprocessed search or explored content that needs reasoning
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        unprocessed_explored = [
            item for item in explored_content
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        # For each piece of unprocessed content, see if we need chunk-based analysis
        for item in unprocessed_search + unprocessed_explored:
            if item.priority < self.min_priority:
                continue
            content_str = str(item.content)
            tokens_estimated = len(content_str) // 4
            # If it's above the threshold, chunk it
            if tokens_estimated > self.max_tokens_per_call:
                chunks = self._split_content_into_chunks(content_str)
                for idx, chunk in enumerate(chunks):
                    decisions.append(
                        ResearchDecision(
                            decision_type=DecisionType.REASON,
                            priority=0.9,
                            context={
                                "content": chunk,
                                "content_type": item.content_type.value,
                                "item_id": item.id,
                                "chunk_info": {
                                    "index": idx,
                                    "total_chunks": len(chunks)
                                }
                            },
                            rationale=f"Analyze chunk {idx+1}/{len(chunks)} of {item.content_type.value} content"
                        )
                    )
            else:
                decisions.append(
                    ResearchDecision(
                        decision_type=DecisionType.REASON,
                        priority=0.9,
                        context={
                            "content": content_str,
                            "content_type": item.content_type.value,
                            "item_id": item.id
                        },
                        rationale=f"Analyze new {item.content_type.value} content"
                    )
                )
        
        # Gather existing analyses to detect knowledge gaps
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_by_source = {}
        for aitem in analysis_items:
            sid = aitem.metadata.get("source_content_id")
            if sid:
                analysis_by_source.setdefault(sid, []).append(aitem)
        
        # Attempt to identify further knowledge gaps (prompt for additional SEARCH or EXPLORE)
        gaps = self._parallel_identify_gaps(context.initial_question, analysis_by_source, context)
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
        
        # Possibly finalize research with a TERMINATE decision if the question is answered
        if self._should_terminate(context):
            decisions.append(
                ResearchDecision(
                    decision_type=DecisionType.TERMINATE,
                    priority=1.0,
                    context={},
                    rationale="Research question appears sufficiently answered"
                )
            )
        
        return decisions
    
    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.REASON
    
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute a REASON decision by analyzing a chunk (or entire content).
        Also merges partial analyses if the content was chunked.
        """
        start_time = time.time()
        success = False
        results = {}
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
            
            tokens_estimated = len(content) // 4
            if tokens_estimated > self.max_tokens_per_call:
                msg = f"Chunk is still too large: {tokens_estimated} tokens > {self.max_tokens_per_call}"
                logger.warning(msg)
                raise ValueError(msg)
            
            single_result = self._analyze_content_chunk(content, content_type)
            # Mark success if we got a valid analysis
            success = bool(single_result.get("analysis"))
            
            # If we have chunk info, store partial results in self.chunked_analyses
            chunk_info = decision.context.get("chunk_info")
            if chunk_info:
                source_key = (item_id, chunk_info["total_chunks"])
                self.chunked_analyses.setdefault(source_key, []).append(single_result)
                
                # If we have all chunk analyses for this source, merge them
                if len(self.chunked_analyses[source_key]) == chunk_info["total_chunks"]:
                    logger.info(
                        "Merging %d chunk analyses for item_id=%s", 
                        chunk_info["total_chunks"], item_id
                    )
                    merged = self._merge_chunk_analyses(self.chunked_analyses[source_key])
                    # Once merged, remove partial results from the dictionary
                    del self.chunked_analyses[source_key]
                    # Our final result is the unified analysis
                    results = merged
                    success = True
                else:
                    # Return empty or partial so the orchestrator doesn't treat
                    # partial chunk results as final
                    # We'll just let the orchestrator store a minimal placeholder
                    results = {
                        "analysis": "",
                        "insights": [],
                        "content_type": content_type,
                        "partial_chunk": True
                    }
            else:
                # No chunking => store single chunk's analysis as final
                results = single_result
            
            # Mark item as analyzed
            decision.context["analyzed_by_reasoner"] = True
            results["analyzed_item_id"] = item_id
            
            if success and not results.get("partial_chunk"):
                rprint(f"[green]ReasoningAgent: Analysis completed for '{content_type}'[/green]")
            elif results.get("partial_chunk"):
                rprint(f"[yellow]ReasoningAgent: Partial chunk analysis stored for '{content_type}'[/yellow]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
                
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            results = {"error": str(e), "analysis": "", "insights": []}
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)
            logger.info("Reasoning decision executed. Success=%s, time=%.2fsec", success, execution_time)
        return results
    
    def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        """
        Analyze a single chunk of content by calling the DeepSeek API.
        Returns a dictionary with 'analysis' text and 'insights'.
        """
        try:
            logger.debug(
                "Sending chunk (length=%d chars) to DeepSeek with max_tokens=%d",
                len(content), self.max_output_tokens
            )
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
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            analysis = response.choices[0].message.content
            return {
                "analysis": analysis,
                "insights": self._extract_insights(analysis),
                "content_type": content_type
            }
        except Exception as e:
            rprint(f"[red]DeepSeek API error: {str(e)}[/red]")
            logger.exception("Error calling DeepSeek for chunk analysis.")
            return {
                "analysis": "",
                "insights": [],
                "content_type": content_type,
                "error": str(e)
            }
    
    def _merge_chunk_analyses(self, partial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple chunk analyses into a single cohesive analysis.
        Simply concatenates the analysis fields and merges insights.
        """
        # Sort partial_results by length or other heuristic if desired;
        # here we simply combine them in the order they arrived
        combined_analysis = []
        combined_insights = []
        
        for result in partial_results:
            if result.get("analysis"):
                combined_analysis.append(result["analysis"])
            if result.get("insights"):
                combined_insights.extend(result["insights"])
        
        # Create a single consolidated analysis
        final_analysis = "\n\n".join(combined_analysis)
        return {
            "analysis": final_analysis,
            "insights": combined_insights,
            "content_type": partial_results[0]["content_type"] if partial_results else "analysis_chunked"
        }
    
    def _extract_insights(self, analysis: str) -> List[str]:
        """
        Extract bullet points or enumerated lines as 'insights' from the analysis text.
        """
        lines = analysis.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if (
                line.startswith("-") or line.startswith("*") or
                line.startswith("•") or (line[:2].isdigit() and line[2] in [".", ")"])
            ):
                insights.append(line.lstrip("-*•").strip())
        return insights
    
    def _split_content_into_chunks(self, content: str) -> List[str]:
        """
        Improved chunk splitting to avoid word-breaks.
        We aim for ~max_tokens_per_call tokens worth of content per chunk.
        We subtract a small buffer to reduce the chance of overshoot.
        """
        safe_limit = self.max_tokens_per_call - 1000
        # Estimate 1 token ~ 4 chars => safe_limit_chars
        safe_limit_chars = safe_limit * 4
        
        words = content.split()
        chunks = []
        current_chunk_words = []
        current_length = 0
        
        for w in words:
            # +1 for space
            token_estimate = len(w) + 1
            if (current_length + token_estimate) >= safe_limit_chars:
                # Close out the current chunk
                chunks.append(" ".join(current_chunk_words))
                current_chunk_words = [w]
                current_length = token_estimate
            else:
                current_chunk_words.append(w)
                current_length += token_estimate
        
        if current_chunk_words:
            chunks.append(" ".join(current_chunk_words))
        
        return chunks
    
    def _generate_initial_queries(self, question: str) -> List[Dict[str, Any]]:
        """
        Uses the deepseek-reasoner to propose multiple initial search queries
        for broad coverage of the user's question.
        """
        try:
            sanitized_question = self._sanitize_and_shorten_question(question)
            logger.debug("Sending sanitized question to DeepSeek: %s", sanitized_question)
            try:
                response = self.deepseek_client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Generate exactly 3 to 5 different search queries in valid JSON format. "
                                "Each item in the JSON array must be an object with 'query' and 'rationale' keys. "
                                "No text outside the JSON array."
                            )
                        },
                        {
                            "role": "user",
                            "content": sanitized_question
                        }
                    ],
                    temperature=0.7,
                    max_tokens=self.max_output_tokens
                )
            except Exception as api_error:
                logger.error("DeepSeek API call failed: %s", str(api_error))
                return [{
                    "query": question,
                    "rationale": f"Direct fallback search - API error: {str(api_error)}"
                }]
            content_str = response.choices[0].message.content.strip()
            logger.debug("DeepSeek API response content: %s", content_str)
            if not content_str:
                logger.error("Empty response from DeepSeek API")
                return [{
                    "query": question,
                    "rationale": "Direct fallback search - empty API response"
                }]
            
            # Strip markdown code block delimiters if present
            if content_str.startswith("```"):
                content_str = content_str.split("\n", 1)[1]  # Remove first line with ```json
            if content_str.endswith("```"):
                content_str = content_str.rsplit("\n", 1)[0]  # Remove last line with ```
            content_str = content_str.strip()
            
            queries = json.loads(content_str)
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON list.")
            for q in queries:
                if not isinstance(q, dict) or "query" not in q or "rationale" not in q:
                    raise ValueError("Must have query + rationale.")
            return queries
        except Exception as e:
            rprint(f"[red]Error generating initial queries: {e}[/red]")
            logger.exception("Error in _generate_initial_queries.")
            return [{
                "query": question,
                "rationale": "Direct fallback search"
            }]

    def _sanitize_and_shorten_question(self, question: str, max_tokens: int = 300) -> str:
        """
        Clean up the user question and limit its size so we don't blow up the model context.
        """
        sanitized = " ".join(question.split())
        words = sanitized.split()
        if len(words) > max_tokens:
            words = words[:max_tokens]
            sanitized = " ".join(words)
        return sanitized
    
    def _parallel_identify_gaps(self, question: str,
                                analysis_by_source: Dict[str, List[ContentItem]],
                                context: ResearchContext) -> List[Dict[str, Any]]:
        """
        Attempt to find knowledge gaps in the analysis so far, which might
        prompt new searches or URL explorations. This is done in parallel
        for each source's combined analysis.
        """
        try:
            tasks = []
            for source_id, items in analysis_by_source.items():
                combined_analysis = " ".join(str(item.content) for item in items)
                tasks.append({
                    "question": question,
                    "analysis": combined_analysis,
                    "source_id": source_id
                })
            with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
                futures = []
                for task in tasks:
                    fut = executor.submit(
                        self._analyze_single_source_gaps,
                        task["question"],
                        task["analysis"],
                        task["source_id"]
                    )
                    futures.append(fut)
                all_gaps = []
                for future in as_completed(futures):
                    try:
                        gaps = future.result()
                        all_gaps.extend(gaps)
                    except Exception as e:
                        rprint(f"[red]Error in gap analysis: {e}[/red]")
                        logger.exception("Error analyzing knowledge gaps.")
            return self._deduplicate_gaps(all_gaps)
        except Exception as e:
            rprint(f"[red]Error in parallel gap identification: {e}[/red]")
            logger.exception("Error in _parallel_identify_gaps.")
            return []
    
    def _analyze_single_source_gaps(self, question: str, analysis: str, source_id: str) -> List[Dict[str, Any]]:
        """
        For a single chunk of combined analysis from a source, ask the deepseek-reasoner
        to identify knowledge gaps that might prompt a new search or exploration.
        """
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Identify knowledge gaps in the current analysis. Output valid JSON array. "
                            "Each item must have: 'type' (search|explore), 'query' or 'url', 'rationale'. "
                            "No text outside the JSON array."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\n\nCurrent Analysis:\n{analysis}"
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            content_str = response.choices[0].message.content.strip()
            gaps = json.loads(content_str)
            if not isinstance(gaps, list):
                raise ValueError("Expected JSON list.")
            for gap in gaps:
                if gap["type"] not in ["search", "explore"]:
                    raise ValueError("type must be search or explore.")
                if gap["type"] == "search" and "query" not in gap:
                    raise ValueError("search item missing query.")
                if gap["type"] == "explore" and "url" not in gap:
                    raise ValueError("explore item missing url.")
                if "rationale" not in gap:
                    raise ValueError("gap missing rationale.")
                gap["source_id"] = source_id
            return gaps
        except Exception as e:
            rprint(f"[red]Error analyzing source gaps: {e}[/red]")
            logger.exception("Error analyzing source gaps for source_id=%s", source_id)
            return []

    def _deduplicate_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        When multiple sources identify the same or very similar gaps,
        deduplicate them so we don't produce repeated queries/explorations.
        """
        unique_gaps = {}
        for gap in gaps:
            key = gap["query"] if gap["type"] == "search" else gap["url"]
            if key not in unique_gaps:
                unique_gaps[key] = gap
            else:
                # Keep the gap with the longer rationale for clarity
                if len(gap["rationale"]) > len(unique_gaps[key]["rationale"]):
                    unique_gaps[key] = gap
        return list(unique_gaps.values())
    
    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Simple heuristic that calls deepseek-reasoner to see if the question is answered
        and if confidence >= 0.8 => we propose termination.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if not analysis_items:
            return False
        combined_analysis = " ".join(str(a.content) for a in analysis_items)
        try:
            response = self.deepseek_client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research completion analyst. Evaluate if we have enough info. "
                            "Return JSON with {complete: bool, confidence: 0..1, missing: []}."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {context.initial_question}\n\nFindings:\n{combined_analysis}"
                    }
                ],
                temperature=0.7,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            try:
                result = json.loads(response.choices[0].message.content)
                return result.get("complete", False) and (result.get("confidence", 0) >= 0.8)
            except Exception:
                return False
        except Exception as e:
            rprint(f"[red]Error checking termination: {e}[/red]")
            logger.exception("Error in _should_terminate check.")
            return False