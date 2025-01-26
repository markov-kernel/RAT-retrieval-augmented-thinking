"""
Reasoning agent for analyzing research content using DeepSeek (deepseek-reasoner).
Now it also acts as the "lead agent" that decides next steps (search, explore, etc.).
Supports parallel processing of content analysis and decision making.

Enhancement:
1. We still instruct deepseek-reasoner to produce JSON, but it may be malformed.
2. We do a "second pass" with deepseek-chat if parsing fails, passing `response_format={'type':'json_object'}` to obtain valid JSON.
3. This preserves all existing flows while guaranteeing structured JSON where needed.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich import print as rprint
from openai import OpenAI
import os
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

    - We instruct the deepseek-reasoner to produce JSON, then attempt to parse it.
    - If parsing fails, we call deepseek-chat with response_format={'type':'json_object'}
      to repair/morph the text into valid JSON.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        
        # Primary client for deepseek-reasoner calls
        self.deepseek_reasoner = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )

        # Secondary client for deepseek-chat calls (to fix malformed JSON)
        # You can reuse the same DEEPSEEK_API_KEY or have a separate one.
        self.deepseek_chat = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.max_tokens_per_call = 64000
        self.max_output_tokens = 8192
        self.request_timeout = self.config.get("deepseek_timeout", 180)
        
        self.min_priority = self.config.get("min_priority", 0.3)

        # For chunk merges
        self.chunked_analyses: Dict[Tuple[str,int], List[Dict[str,Any]]] = {}

        logger.info(
            "ReasoningAgent initialized with max_parallel_tasks=%d, max_tokens_per_call=%d, max_output_tokens=%d",
            self.max_parallel_tasks,
            self.max_tokens_per_call,
            self.max_output_tokens
        )

    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
        search_results = context.get_content("main", ContentType.SEARCH_RESULT)
        if not search_results:
            # Generate initial queries if no search results
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
        
        # Identify unprocessed search or explored content
        unprocessed_search = [
            item for item in search_results
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        unprocessed_explored = [
            item for item in explored_content
            if not item.metadata.get("analyzed_by_reasoner")
        ]
        
        for item in unprocessed_search + unprocessed_explored:
            if item.priority < self.min_priority:
                continue
            content_str = str(item.content)
            tokens_estimated = len(content_str) // 4
            if tokens_estimated > self.max_tokens_per_call:
                # Chunk-based approach
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
                # Single chunk
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
        
        # Additional knowledge gap detection
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_by_source = {}
        for aitem in analysis_items:
            sid = aitem.metadata.get("source_content_id")
            if sid:
                analysis_by_source.setdefault(sid, []).append(aitem)
        
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
            
            # Instruct deepseek-reasoner to produce JSON in the analysis
            # (Might be malformed; we will fix it with deepseek-chat if needed.)
            single_result = self._analyze_content_chunk(content, content_type)
            success = bool(single_result.get("analysis"))

            chunk_info = decision.context.get("chunk_info")
            if chunk_info:
                # If chunk, store partial
                source_key = (item_id, chunk_info["total_chunks"])
                self.chunked_analyses.setdefault(source_key, []).append(single_result)
                
                # If we now have all chunk analyses, unify them
                if len(self.chunked_analyses[source_key]) == chunk_info["total_chunks"]:
                    merged = self._merge_chunk_analyses(self.chunked_analyses[source_key])
                    del self.chunked_analyses[source_key]
                    results = merged
                    success = True
                else:
                    # Partial chunk result
                    results = {
                        "analysis": "",
                        "insights": [],
                        "content_type": content_type,
                        "partial_chunk": True
                    }
            else:
                # Single chunk, final
                results = single_result
            
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
        Analyze a chunk of content using DeepSeek.
        Handles both single-page and batch exploration results.
        """
        try:
            # Prepare the content for analysis
            if content_type == ContentType.EXPLORED_CONTENT.value:
                # Parse the content if it's a string
                if isinstance(content, str):
                    content = json.loads(content)
                
                # Handle batch exploration results
                if isinstance(content, dict) and content.get("type") == "batch_exploration":
                    # Special handling for batch results
                    domain = content.get("domain", "")
                    urls = content.get("urls", [])
                    batch_results = content.get("results", [])
                    
                    # Analyze each result in the batch
                    combined_analysis = []
                    combined_insights = []
                    
                    for idx, (url, result) in enumerate(zip(urls, batch_results)):
                        # Skip empty or error results
                        if not result or "error" in result:
                            continue
                            
                        # Analyze the individual page
                        page_analysis = self._analyze_single_page(result, url)
                        if page_analysis:
                            combined_analysis.append(
                                f"Page {idx + 1} ({url}):\n{page_analysis['analysis']}"
                            )
                            combined_insights.extend(page_analysis.get("insights", []))
                    
                    # Combine all analyses
                    return {
                        "analysis": "\n\n".join(combined_analysis),
                        "insights": combined_insights,
                        "content_type": content_type,
                        "domain": domain,
                        "url_count": len(urls),
                        "successful_analyses": len(combined_analysis)
                    }
                else:
                    # Single page result
                    return self._analyze_single_page(content, content.get("metadata", {}).get("url", ""))
            else:
                # Handle search results or other content types
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert research analyst. Analyze the following content "
                            "and provide insights. Format your response as JSON with 'analysis' "
                            "and 'insights' fields. Keep insights concise and focused."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Content type: {content_type}\n\nContent to analyze:\n{content}"
                    }
                ]
                
                response = self.deepseek_reasoner.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=messages,
                    max_tokens=self.max_output_tokens,
                    temperature=0.3,
                    timeout=self.request_timeout
                )
                
                try:
                    result_text = response.choices[0].message.content
                    result_json = json.loads(result_text)
                    return {
                        **result_json,
                        "content_type": content_type
                    }
                except json.JSONDecodeError:
                    # If JSON is malformed, try to fix it with deepseek-chat
                    fixed_json = self._transform_malformed_json_with_chat(result_text)
                    result_json = json.loads(fixed_json)
                    return {
                        **result_json,
                        "content_type": content_type
                    }
                    
        except Exception as e:
            logger.exception("Error analyzing content chunk")
            return {
                "error": str(e),
                "analysis": "",
                "insights": [],
                "content_type": content_type
            }
            
    def _analyze_single_page(self, content: Dict[str, Any], url: str) -> Dict[str, Any]:
        """
        Analyze a single webpage's content.
        
        Args:
            content: The page content and metadata
            url: The source URL
            
        Returns:
            Analysis results for the page
        """
        try:
            # Extract the relevant content
            title = content.get("title", "")
            text = content.get("text", "")
            metadata = content.get("metadata", {})
            
            # Prepare the content for analysis
            page_content = f"Title: {title}\n\nURL: {url}\n\nContent:\n{text}"
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert research analyst. Analyze the following webpage "
                        "content and provide insights. Format your response as JSON with "
                        "'analysis' and 'insights' fields. Keep insights concise and focused."
                    )
                },
                {
                    "role": "user",
                    "content": page_content
                }
            ]
            
            response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=messages,
                max_tokens=self.max_output_tokens,
                temperature=0.3,
                timeout=self.request_timeout
            )
            
            try:
                result_text = response.choices[0].message.content
                result_json = json.loads(result_text)
                return {
                    **result_json,
                    "url": url,
                    "metadata": metadata
                }
            except json.JSONDecodeError:
                # If JSON is malformed, try to fix it with deepseek-chat
                fixed_json = self._transform_malformed_json_with_chat(result_text)
                result_json = json.loads(fixed_json)
                return {
                    **result_json,
                    "url": url,
                    "metadata": metadata
                }
                
        except Exception as e:
            logger.exception("Error analyzing single page: %s", url)
            return {
                "error": str(e),
                "analysis": "",
                "insights": [],
                "url": url
            }

    def _transform_malformed_json_with_chat(self, possibly_malformed: str) -> str:
        """
        Call deepseek-chat to fix malformed JSON text.
        Returns the corrected JSON string or "[]" if it cannot repair it.
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an assistant that strictly outputs valid JSON. "
                        "Take the user's input (which may be malformed JSON or random text) "
                        "and produce well-formed JSON that best represents it. "
                        "If the input appears to be an array, output a JSON array. "
                        "Otherwise output a JSON object. "
                        "If you cannot parse it meaningfully, output an empty array."
                    )
                },
                {
                    "role": "user",
                    "content": possibly_malformed
                }
            ]
            response = self.deepseek_chat.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                max_tokens=self.max_output_tokens,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.warning("Unable to fix malformed JSON with deepseek-chat: %s", e)
            return "[]"

    def _merge_chunk_analyses(self, partials: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple chunk analyses into a single cohesive analysis, 
        combining their 'analysis' text and 'insights'.
        """
        combined_analysis = []
        combined_insights = []
        
        for result in partials:
            if result.get("analysis"):
                combined_analysis.append(result["analysis"])
            if result.get("insights"):
                combined_insights.extend(result["insights"])
        
        final_analysis = "\n\n".join(combined_analysis)
        return {
            "analysis": final_analysis,
            "insights": combined_insights,
            "content_type": partials[0]["content_type"] if partials else "analysis_chunked"
        }

    def _split_content_into_chunks(self, content: str) -> List[str]:
        """
        Split large content into chunks that are below the max token limit.
        Roughly: 1 token ~ 4 chars. We subtract ~1000 tokens as a safety buffer.
        """
        safe_limit = self.max_tokens_per_call - 1000
        safe_limit_chars = safe_limit * 4
        
        words = content.split()
        chunks = []
        current_chunk_words = []
        current_length = 0
        
        for w in words:
            token_estimate = len(w) + 1  # approximate
            if (current_length + token_estimate) >= safe_limit_chars:
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
        We instruct deepseek-reasoner to produce multiple queries in JSON. 
        If it's malformed, we again correct with deepseek-chat for JSON.
        """
        try:
            sanitized_question = " ".join(question.split())
            reasoner_response = self.deepseek_reasoner.chat.completions.create(
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
                max_tokens=self.max_output_tokens,
                stream=False
            )
            raw_text = reasoner_response.choices[0].message.content.strip()
            # Try to parse
            try:
                queries = json.loads(raw_text)
            except json.JSONDecodeError:
                # Fix with deepseek-chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    queries = json.loads(corrected)
                except:
                    # fallback
                    return [{
                        "query": question,
                        "rationale": "Direct fallback search - JSON fix failed"
                    }]
            # Validate minimal structure
            if not isinstance(queries, list):
                raise ValueError("Expected a JSON list of queries.")
            for q in queries:
                if not isinstance(q, dict) or "query" not in q or "rationale" not in q:
                    raise ValueError("Each query must have 'query' and 'rationale'.")
            return queries
        except Exception as e:
            rprint(f"[red]Error generating initial queries: {e}[/red]")
            logger.exception("Error in _generate_initial_queries.")
            return [{
                "query": question,
                "rationale": "Direct fallback search"
            }]

    def _parallel_identify_gaps(self,
                                question: str,
                                analysis_by_source: Dict[str, List[ContentItem]],
                                context: ResearchContext
                               ) -> List[Dict[str, Any]]:
        """
        Attempt to find knowledge gaps in the analysis, possibly creating new search or explore tasks.
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
            
            all_gaps = []
            with ThreadPoolExecutor(max_workers=self.max_parallel_tasks) as executor:
                futures = [
                    executor.submit(
                        self._analyze_single_source_gaps,
                        t["question"],
                        t["analysis"],
                        t["source_id"]
                    )
                    for t in tasks
                ]
                for fut in as_completed(futures):
                    try:
                        all_gaps.extend(fut.result())
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
        Ask deepseek-reasoner to produce JSON gap info. If malformed, fix with chat.
        """
        try:
            response = self.deepseek_reasoner.chat.completions.create(
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
            raw_text = response.choices[0].message.content.strip()
            
            try:
                gaps = json.loads(raw_text)
            except json.JSONDecodeError:
                # fix with chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    gaps = json.loads(corrected)
                except:
                    # fallback
                    return []
            
            if not isinstance(gaps, list):
                raise ValueError("Expected JSON list.")
            # validate
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
        unique_gaps = {}
        for gap in gaps:
            key = gap["query"] if gap["type"] == "search" else gap["url"]
            if key not in unique_gaps:
                unique_gaps[key] = gap
            else:
                # keep the gap with the longer rationale
                if len(gap["rationale"]) > len(unique_gaps[key]["rationale"]):
                    unique_gaps[key] = gap
        return list(unique_gaps.values())

    def _should_terminate(self, context: ResearchContext) -> bool:
        """
        Asks deepseek-reasoner for a JSON {complete: bool, confidence: float, missing: []}.
        If malformed, we fix it with chat. If we get `complete==true and confidence>=0.8`, we return True.
        """
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if not analysis_items:
            return False

        combined_analysis = " ".join(str(a.content) for a in analysis_items)
        try:
            response = self.deepseek_reasoner.chat.completions.create(
                model="deepseek-reasoner",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a research completion analyst. Evaluate if we have enough info. "
                            "Return JSON with {\"complete\": bool, \"confidence\": 0..1, \"missing\": []}."
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
            raw_text = response.choices[0].message.content
            # try parse
            try:
                result = json.loads(raw_text)
            except json.JSONDecodeError:
                # fix with chat
                corrected = self._transform_malformed_json_with_chat(raw_text)
                try:
                    result = json.loads(corrected)
                except:
                    return False

            complete = bool(result.get("complete", False))
            confidence = float(result.get("confidence", 0))
            return (complete and confidence >= 0.8)
        except Exception as e:
            rprint(f"[red]Error checking termination: {e}[/red]")
            logger.exception("Error in _should_terminate check.")
            return False