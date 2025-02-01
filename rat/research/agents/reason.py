"""
Reasoning agent for analyzing research content using the o3-mini model with high reasoning effort.
All decisions are now generated solely by O3 mini.
All next–step decisions are derived from a structured JSON response.
All local heuristics or NLP rules have been removed.
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import time
from rich import print as rprint
import logging
import json
from urllib.parse import urlparse
import openai
from openai import OpenAI
import uuid

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem
from ..circuit_breaker import CircuitBreaker
from ..monitoring import MetricsManager

logger = logging.getLogger(__name__)
api_logger = logging.getLogger('api.o3mini')


@dataclass
class AnalysisTask:
    content: str
    priority: float
    rationale: str
    chunk_index: Optional[int] = None
    timestamp: float = time.time()


class ReasoningAgent(BaseAgent):
    """
    Reasoning agent that uses O3 mini exclusively to decide the next research steps.
    It aggregates the current research context and sends it to O3 mini to obtain a JSON array
    of decisions. No local heuristics or fixed rules are applied.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        self.model_name = "o3-mini"
        self.max_output_tokens = self.config.get("max_output_tokens", 50000)
        self.request_timeout = self.config.get("o3_mini_timeout", 180)
        self.reasoning_effort = "high"
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.flash_fix_rate_limit = self.config.get("flash_fix_rate_limit", 10)
        self._flash_fix_last_time = 0.0
        self._flash_fix_lock = asyncio.Lock()
        self.session_id = str(uuid.uuid4())
        self.metrics = MetricsManager().get_metrics(self.session_id)
        
        # Initialize circuit breakers
        self.analysis_circuit = CircuitBreaker(
            name="o3mini_analysis",
            failure_threshold=self.config.get("circuit_failure_threshold", 5),
            reset_timeout=self.config.get("circuit_reset_timeout", 60.0),
            session_id=self.session_id
        )
        self.decision_circuit = CircuitBreaker(
            name="o3mini_decisions",
            failure_threshold=self.config.get("circuit_failure_threshold", 5),
            reset_timeout=self.config.get("circuit_reset_timeout", 60.0),
            session_id=self.session_id
        )
        
        logger.info(f"ReasoningAgent initialized with session {self.session_id}")
        self.analysis_tasks = {}

    async def _enforce_flash_fix_limit(self):
        if self.flash_fix_rate_limit <= 0:
            return
        async with self._flash_fix_lock:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self._flash_fix_last_time
            min_interval = 60.0 / self.flash_fix_rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
                self.metrics.record_api_call(
                    success=True,
                    latency=min_interval - elapsed,
                    metadata={"type": "rate_limit_delay"}
                )
            self._flash_fix_last_time = asyncio.get_event_loop().time()

    async def _call_o3_mini(self, prompt: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "user", "content": prompt})
        api_logger.info(f"o3-mini API Request - Prompt length: {len(prompt)}")
        
        async def _make_api_call():
            await self._enforce_flash_fix_limit()
            client = OpenAI()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model_name,
                messages=messages,
                reasoning_effort=self.reasoning_effort,
                max_completion_tokens=self.max_output_tokens,
                **({"response_format": {"type": "json_object"}} if "json" in prompt.lower() else {})
            )
            return response.choices[0].message.content
        
        try:
            # Use circuit breaker for API calls
            result = await self.analysis_circuit.execute(_make_api_call)
            return result
        except Exception as e:
            api_logger.error(f"o3-mini API error: {str(e)}")
            self.metrics.record_error(
                error_type="api_error",
                error_message=str(e),
                metadata={"model": self.model_name}
            )
            raise

    async def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        # Aggregate the current research context.
        aggregated_text = ""
        for content_item in context.get_content("main"):
            aggregated_text += f"\n[{content_item.content_type.value}] {content_item.content}\n"
        
        prompt = (
            "You are an advanced research assistant. Given the research question and the following aggregated research content, "
            "generate a JSON array of decisions for the next steps in the research process. Each decision must be an object with the following keys:\n"
            " - decision_type: one of 'search', 'explore', 'reason', or 'terminate'\n"
            " - priority: a number between 0 and 1 indicating the importance of the decision\n"
            " - context: an object containing necessary parameters (for example, for a search decision, include a 'query' field; for an explore decision, include a 'url' field)\n"
            " - rationale: a brief explanation for the decision\n\n"
            "The research question is: \"" + context.initial_question + "\"\n\n"
            "The aggregated research content is:\n" + aggregated_text + "\n\n"
            "Return only the JSON array, with no additional text."
        )
        
        try:
            # Use circuit breaker for decision generation
            response = await self.decision_circuit.execute(self._call_o3_mini, prompt)
            try:
                decisions_json = json.loads(response)
                if not isinstance(decisions_json, list):
                    logger.warning("O3 mini response is not a list of decisions")
                    return []
                decisions = []
                for d in decisions_json:
                    if not isinstance(d, dict):
                        continue
                    decision_type_str = d.get("decision_type", "").lower()
                    if decision_type_str not in [dt.value for dt in DecisionType]:
                        continue
                    decision = ResearchDecision(
                        decision_type=DecisionType(decision_type_str),
                        priority=float(d.get("priority", 0.5)),
                        context=d.get("context", {}),
                        rationale=d.get("rationale", "")
                    )
                    decisions.append(decision)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse O3 mini response as JSON: {str(e)}")
                return []
            except Exception as e:
                logger.error(f"Error parsing decisions from O3 mini response: {str(e)}")
                return []
            
            # Record metrics
            self.metrics.record_decision(
                success=True,
                execution_time=time.time() - time.time(),  # Just use current time difference since we don't have last_update_time
                metadata={
                    "decisions_count": len(decisions),
                    "question": context.initial_question
                }
            )
            
            return decisions
            
        except Exception as e:
            self.logger.error(f"Error parsing decisions from O3 mini response: {str(e)}")
            self.metrics.record_decision(
                success=False,
                execution_time=time.time() - context.last_update_time,
                metadata={"error": str(e)}
            )
            return []

    def can_handle(self, decision: ResearchDecision) -> bool:
        return decision.decision_type == DecisionType.REASON

    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = time.time()
        success = False
        results = {}
        try:
            # For a reason decision, we expect a 'content' field in the context.
            content = decision.context.get("content", "")
            content_type = decision.context.get("content_type", "unknown")
            item_id = decision.context.get("item_id", "")
            content_str = str(content)
            tokens_estimated = len(content_str) // 4
            
            if tokens_estimated > self.max_output_tokens:
                chunk_results = await self._parallel_analyze_content(content_str, content_type)
                results = self._combine_chunk_results(chunk_results)
            else:
                results = await self._analyze_content_chunk(content_str, content_type)
            
            success = bool(results.get("analysis", "").strip())
            decision.context["analyzed_by_reasoner"] = True
            
            final_results = {
                "analysis": results.get("analysis", ""),
                "insights": results.get("insights", []),
                "analyzed_item_id": item_id
            }
            
            if success:
                rprint(f"[green]ReasoningAgent: Analysis completed for content type '{content_type}'[/green]")
            else:
                rprint(f"[yellow]ReasoningAgent: No analysis produced for '{content_type}'[/yellow]")
            
            # Record metrics
            execution_time = time.time() - start_time
            self.metrics.record_decision(
                success=success,
                execution_time=execution_time,
                metadata={
                    "content_type": content_type,
                    "tokens": tokens_estimated
                }
            )
            
            return final_results
            
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            self.metrics.record_error(
                error_type="execution_error",
                error_message=str(e),
                metadata={"decision_type": "reason"}
            )
            return {"error": str(e), "analysis": "", "insights": []}
        finally:
            execution_time = time.time() - start_time
            self.log_decision(decision, success, execution_time)

    async def _parallel_analyze_content(self, content: str, content_type: str) -> List[Dict[str, Any]]:
        words = content.split()
        chunk_size_words = self.max_output_tokens * 4
        chunks = []
        idx = 0
        while idx < len(words):
            chunk = words[idx: idx + chunk_size_words]
            chunks.append(" ".join(chunk))
            idx += chunk_size_words
        tasks = []
        for i, chunk in enumerate(chunks):
            tasks.append(asyncio.create_task(self._analyze_content_chunk(chunk, f"{content_type}_chunk_{i}")))
        chunk_results = await asyncio.gather(*tasks, return_exceptions=False)
        for i, res in enumerate(chunk_results):
            res["chunk_index"] = i
        return chunk_results

    async def _analyze_content_chunk(self, content: str, content_type: str) -> Dict[str, Any]:
        await self._enforce_flash_fix_limit()
        prompt = (
            "You are an advanced reasoning model (o3-mini) with high reasoning effort. "
            "Analyze the following text for key insights, patterns, or relevant facts. "
            "Provide ONLY factual analysis and insights without placeholders or next-step suggestions.\n\n"
            f"CONTENT:\n{content}\n\n"
            "Please provide your analysis below (plain text only):"
        )
        
        try:
            # Use circuit breaker for content analysis
            response_text = await self.analysis_circuit.execute(self._call_o3_mini, prompt)
            analysis_text = response_text.strip()
            insights = self._extract_insights(analysis_text)
            
            # Record successful analysis
            self.metrics.record_api_call(
                success=True,
                latency=time.time() - self._flash_fix_last_time,
                metadata={
                    "content_type": content_type,
                    "insights_count": len(insights)
                }
            )
            
            return {"analysis": analysis_text, "insights": insights}
            
        except Exception as e:
            self.metrics.record_error(
                error_type="analysis_error",
                error_message=str(e),
                metadata={"content_type": content_type}
            )
            return {"analysis": "", "insights": []}

    def _extract_insights(self, analysis_text: str) -> List[str]:
        lines = analysis_text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if line:
                insights.append(line)
        return insights

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        combined_analysis = "\n\n".join(res["analysis"] for res in sorted_chunks if res.get("analysis", "").strip())
        combined_insights = []
        for res in sorted_chunks:
            combined_insights.extend(insight for insight in res.get("insights", []) if insight.strip())
        unique_insights = list(dict.fromkeys(combined_insights))
        return {"analysis": combined_analysis, "insights": unique_insights, "chunk_count": len(chunk_results)}

    async def _should_terminate(self, context: ResearchContext) -> bool:
        prompt = (
            "You are an advanced research assistant. Given the research question and the aggregated research content below, "
            "determine if the research question has been sufficiently answered. Respond with a single word: YES if it is answered, NO if it is not.\n\n"
            f"Research Question: {context.initial_question}\n\n"
            "Aggregated Research Content:\n" +
            "\n".join(str(item.content) for item in context.get_content("main"))
        )
        try:
            # Use circuit breaker for termination check
            answer = await self.decision_circuit.execute(self._call_o3_mini, prompt)
            result = answer.strip().upper() == "YES"
            
            # Record the termination decision
            self.metrics.record_decision(
                success=True,
                execution_time=0.0,
                metadata={
                    "decision_type": "terminate",
                    "result": result
                }
            )
            
            return result
        except Exception as e:
            self.metrics.record_error(
                error_type="termination_error",
                error_message=str(e)
            )
            return False