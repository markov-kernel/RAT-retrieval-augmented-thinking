"""
Reasoning agent for analyzing research content using the o3-mini model with high reasoning effort.
Now acts as the "lead agent" that decides next steps.
All methods are now asynchronous.
"""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import time
from rich import print as rprint
import logging
import json
import re
from urllib.parse import urlparse
import openai
from openai import OpenAI

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType, ContentItem

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
    Reasoning agent for analyzing research content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("reason", config)
        self.model_name = "o3-mini"
        self.max_output_tokens = self.config.get("max_output_tokens", 50000)
        self.request_timeout = self.config.get("o3_mini_timeout", 180)
        self.reasoning_effort = "high"
        self.chunk_margin = 5000
        self.max_parallel_tasks = self.config.get("max_parallel_tasks", 3)
        self.min_priority = self.config.get("min_priority", 0.3)
        self.min_url_relevance = self.config.get("min_url_relevance", 0.6)
        self.explored_urls: Set[str] = set()
        self.flash_fix_rate_limit = self.config.get("flash_fix_rate_limit", 10)
        self._flash_fix_last_time = 0.0
        self._flash_fix_lock = asyncio.Lock()
        logger.info("ReasoningAgent initialized to use o3-mini model: %s", self.model_name)
        self.analysis_tasks: Dict[str, AnalysisTask] = {}

    async def _enforce_flash_fix_limit(self):
        if self.flash_fix_rate_limit <= 0:
            return
        async with self._flash_fix_lock:
            current_time = asyncio.get_event_loop().time()
            elapsed = current_time - self._flash_fix_last_time
            min_interval = 60.0 / self.flash_fix_rate_limit
            if elapsed < min_interval:
                await asyncio.sleep(min_interval - elapsed)
                self.metrics["rate_limit_delays"] += 1
            self._flash_fix_last_time = asyncio.get_event_loop().time()

    async def _call_o3_mini(self, prompt: str, context: str = "") -> str:
        messages = []
        if context:
            messages.append({"role": "assistant", "content": context})
        messages.append({"role": "user", "content": prompt})
        api_logger.info(f"o3-mini API Request - Prompt length: {len(prompt)}")
        try:
            # Enforce rate limit before making the API call
            await self._enforce_rate_limit()

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
        except Exception as e:
            api_logger.error(f"o3-mini API error: {str(e)}")
            raise

    async def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        decisions = []
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

        explored_content = context.get_content("main", ContentType.EXPLORED_CONTENT)
        explored_urls = { 
            item.content.get("url", "") for item in explored_content
            if isinstance(item.content, dict)
        }
        self.explored_urls.update(explored_urls)
        unvisited_urls = set()
        for result in search_results:
            if isinstance(result.content, dict):
                urls = result.content.get("urls", [])
                unvisited_urls.update(url for url in urls if url not in self.explored_urls)
        relevant_urls = await self._filter_relevant_urls(list(unvisited_urls), context)
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
        unprocessed_search = [item for item in search_results if not item.metadata.get("analyzed_by_reasoner")]
        unprocessed_explored = [item for item in explored_content if not item.metadata.get("analyzed_by_reasoner")]
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
        search_text = "\n".join(str(item.content) for item in search_results if isinstance(item.content, str))
        explored_text = "\n".join(str(item.content) for item in explored_content if isinstance(item.content, str))
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        analysis_text = "\n".join(
            (item.content.get("analysis", "") if isinstance(item.content, dict) else str(item.content))
            for item in analysis_items
        )
        combined_analysis = f"{search_text}\n\n{explored_text}\n\n{analysis_text}".strip()
        if combined_analysis:
            gaps = await self._identify_knowledge_gaps(context.initial_question, combined_analysis)
            filtered_gaps = []
            for gap in gaps:
                query_str = gap.get("query", "")
                url_str = gap.get("url", "")
                if any(x in query_str or x in url_str for x in ("[", "]")):
                    self.logger.warning(f"Skipping gap with placeholders: {gap}")
                    continue
                filtered_gaps.append(gap)
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
            self.logger.info("Skipping knowledge gap analysis due to lack of context.")
        if await self._should_terminate(context):
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
        return decision.decision_type == DecisionType.REASON

    async def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        start_time = asyncio.get_event_loop().time()
        success = False
        results = {}
        try:
            content = decision.context["content"]
            content_type = decision.context["content_type"]
            item_id = decision.context["item_id"]
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
            return final_results
        except Exception as e:
            rprint(f"[red]ReasoningAgent error: {str(e)}[/red]")
            return {"error": str(e), "analysis": "", "insights": []}
        finally:
            execution_time = asyncio.get_event_loop().time() - start_time
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
        response_text = await self._call_o3_mini(prompt)
        analysis_text = response_text.strip()
        insights = self._extract_insights(analysis_text)
        return {"analysis": analysis_text, "insights": insights}

    def _extract_insights(self, analysis_text: str) -> List[str]:
        lines = analysis_text.split("\n")
        insights = []
        for line in lines:
            line = line.strip()
            if (line.startswith("-") or line.startswith("*") or line.startswith("•") or 
                (len(line) > 2 and line[:2].isdigit())):
                insights.append(line.lstrip("-*•").strip())
        return insights

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        sorted_chunks = sorted(chunk_results, key=lambda x: x.get("chunk_index", 0))
        combined_analysis = "\n\n".join(res["analysis"] for res in sorted_chunks if res.get("analysis", "").strip())
        combined_insights = []
        for res in sorted_chunks:
            combined_insights.extend(insight for insight in res.get("insights", []) if insight.strip())
        unique_insights = list(dict.fromkeys(combined_insights))
        return {"analysis": combined_analysis, "insights": unique_insights, "chunk_count": len(chunk_results)}

    async def _identify_knowledge_gaps(self, question: str, current_analysis: str) -> List[Dict[str, Any]]:
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "identify specific missing information and suggest concrete next steps.\n\n"
            "IMPORTANT RULES:\n"
            "1. DO NOT use placeholders like [company name] or [person].\n"
            "2. Base suggestions solely on the provided content.\n"
            "3. If no specific gaps can be identified, return an empty array.\n"
            "4. Each suggestion must be actionable and clearly linked to the research question.\n\n"
            f"RESEARCH QUESTION: {question}\n\n"
            f"CURRENT ANALYSIS:\n{current_analysis}\n\n"
            "Return a JSON object with a 'gaps' array in this format:\n"
            "{\"gaps\": [{\"type\": \"search\"|\"explore\", \"query\"|\"url\": \"specific text\", \"rationale\": \"why needed\"}]}"
        )
        try:
            response = await self._call_o3_mini(prompt)
            content_str = response.strip()
            if not content_str:
                return []
            try:
                result = json.loads(content_str)
                gaps = result.get("gaps", [])
                filtered_gaps = []
                for gap in gaps:
                    if not isinstance(gap, dict):
                        continue
                    if "type" not in gap or gap["type"] not in ["search", "explore"]:
                        continue
                    content_field = "query" if gap["type"] == "search" else "url"
                    if content_field not in gap or "rationale" not in gap:
                        continue
                    if any(x in gap[content_field] or x in gap["rationale"] for x in ("[", "]")):
                        self.logger.warning(f"Skipping gap with placeholders: {gap}")
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
        content = content.strip()
        if content.startswith("```"):
            start_idx = content.find("\n") + 1
            end_idx = content.rfind("```")
            if end_idx > start_idx:
                content = content[start_idx:end_idx].strip()
            else:
                content = content.replace("```", "").strip()
        content = content.replace("json", "").strip()
        return content

    async def _should_terminate(self, context: ResearchContext) -> bool:
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        if len(analysis_items) < 3:
            return False
        combined_analysis = "\n".join(
            str(item.content.get("analysis", "")) for item in analysis_items if isinstance(item.content, dict)
        )
        prompt = (
            "You are an advanced research assistant. Given a research question and current analysis, "
            "determine if the question has been sufficiently answered.\n\n"
            f"QUESTION: {context.initial_question}\n\n"
            f"CURRENT ANALYSIS:\n{combined_analysis}\n\n"
            "Respond with a single word: YES if answered, NO if not."
        )
        try:
            answer = await self._call_o3_mini(prompt, combined_analysis)
            return answer.strip().upper() == "YES"
        except Exception:
            return False

    async def _filter_relevant_urls(self, urls: List[str], context: ResearchContext) -> List[tuple]:
        if not urls:
            return []
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
                content = await self._call_o3_mini(prompt)
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
                for url in batch:
                    relevance = self._basic_url_relevance(url, context.initial_question)
                    relevant_urls.append((url, relevance))
        return sorted(relevant_urls, key=lambda x: x[1], reverse=True)

    def _basic_url_relevance(self, url: str, question: str) -> float:
        import re
        from urllib.parse import urlparse
        keywords = set(re.findall(r'\w+', question.lower()))
        parsed = urlparse(url)
        domain_parts = parsed.netloc.lower().split('.')
        path_parts = parsed.path.lower().split('/')
        domain_matches = sum(1 for part in domain_parts if part in keywords)
        path_matches = sum(1 for part in path_parts if part in keywords)
        score = (domain_matches * 0.6 + path_matches * 0.4) / max(len(keywords), 1)
        return min(max(score, 0.0), 1.0)