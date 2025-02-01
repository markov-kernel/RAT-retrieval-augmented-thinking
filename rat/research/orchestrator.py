"""
Orchestrator for coordinating the multi-agent research workflow.
Now fully asynchronous.
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from rich import print as rprint
from pathlib import Path
import markdown
from weasyprint import HTML
import openai

from rat.research.agents.search import SearchAgent
from rat.research.agents.explore import ExploreAgent
from rat.research.agents.reason import ReasoningAgent
from rat.research.perplexity_client import PerplexityClient
from rat.research.firecrawl_client import FirecrawlClient
from rat.research.output_manager import OutputManager
from rat.research.agents.base import ResearchDecision, DecisionType
from rat.research.agents.context import ResearchContext, ContentType, ContentItem

logger = logging.getLogger(__name__)


@dataclass
class ResearchIteration:
    iteration_number: int
    decisions_made: List[ResearchDecision]
    content_added: List[ContentItem]
    metrics: Dict[str, Any]
    timestamp: float = time.time()


class ResearchOrchestrator:
    """
    Coordinates the multi-agent research workflow.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.perplexity = PerplexityClient()
        self.firecrawl = FirecrawlClient()
        self.search_agent = SearchAgent(
            self.perplexity,
            {
                **(self.config.get("search_config") or {}),
                "max_workers": self.config.get("max_parallel_searches", 10),
                "rate_limit": self.config.get("search_rate_limit", 100)
            }
        )
        self.explore_agent = ExploreAgent(
            self.firecrawl,
            {
                **(self.config.get("explore_config") or {}),
                "max_workers": self.config.get("max_parallel_explores", 10),
                "rate_limit": self.config.get("explore_rate_limit", 50)
            }
        )
        self.reason_agent = ReasoningAgent(
            {
                **(self.config.get("reason_config") or {}),
                "max_workers": self.config.get("max_parallel_reason", 5),
                "rate_limit": self.config.get("reason_rate_limit", 10),
                "flash_fix_rate_limit": self.config.get("flash_fix_rate_limit", 10)
            }
        )
        self.output_manager = OutputManager()
        self.max_iterations = self.config.get("max_iterations", 5)
        self.min_new_content = self.config.get("min_new_content", 1)
        self.min_confidence = self.config.get("min_confidence", 0.7)
        self.current_context: Optional[ResearchContext] = None
        self.iterations: List[ResearchIteration] = []
        self.research_dir: Optional[Path] = None
        self.previous_searches = set()

    async def start_research(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        self.research_dir = self.output_manager.create_research_dir(question)
        self.current_context = ResearchContext(initial_question=question)
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            rprint(f"\n[bold cyan]Starting iteration {iteration}[/bold cyan]")
            iteration_result = await self._run_iteration(iteration)
            self.iterations.append(iteration_result)
            if self._should_terminate(iteration_result):
                break
        results = await self._generate_final_output()
        total_time = time.time() - start_time
        results["metrics"] = self._calculate_metrics(total_time)
        if self.research_dir:
            self.output_manager.save_research_paper(self.research_dir, results)
        return results

    async def _run_iteration(self, iteration_number: int) -> ResearchIteration:
        iteration_start = time.time()
        all_decisions = []
        content_added = []
        try:
            reason_decisions = await self.reason_agent.analyze(self.current_context)
            if any(d.decision_type == DecisionType.TERMINATE for d in reason_decisions):
                all_decisions.extend(reason_decisions)
            else:
                search_decisions = await self.search_agent.analyze(self.current_context)
                explore_decisions = await self.explore_agent.analyze(self.current_context)
                all_decisions = reason_decisions + search_decisions + explore_decisions
            sorted_decisions = sorted(all_decisions, key=lambda d: d.priority, reverse=True)
            for decision in sorted_decisions:
                if decision.decision_type == DecisionType.TERMINATE:
                    break
                agent = self._get_agent_for_decision(decision)
                if not agent:
                    continue
                if decision.decision_type == DecisionType.SEARCH:
                    query_str = decision.context.get("query", "").strip()
                    if not query_str:
                        continue
                    if query_str in self.previous_searches:
                        rprint(f"[yellow]Skipping duplicate search: '{query_str}'[/yellow]")
                        continue
                    else:
                        self.previous_searches.add(query_str)
                try:
                    result = await agent.execute_decision(decision)
                    if result:
                        content_item = self._create_content_item(decision, result, iteration_number)
                        self.current_context.add_content("main", content_item=content_item)
                        content_added.append(content_item)
                except Exception as e:
                    rprint(f"[red]Error executing decision: {str(e)}[/red]")
        except Exception as e:
            rprint(f"[red]Iteration error: {str(e)}[/red]")
        metrics = {
            "iteration_time": time.time() - iteration_start,
            "decisions_made": len(all_decisions),
            "content_added": len(content_added),
            "agent_metrics": self._get_agent_metrics()
        }
        return ResearchIteration(
            iteration_number=iteration_number,
            decisions_made=all_decisions,
            content_added=content_added,
            metrics=metrics
        )

    def _create_content_item(self, decision: ResearchDecision, result: Dict[str, Any], iteration_number: int) -> ContentItem:
        if decision.decision_type == DecisionType.SEARCH:
            content_str = result.get('content', '')
            urls = result.get('urls', [])
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=self._get_content_type(decision),
                content=content_str,
                metadata={"decision_type": decision.decision_type.value, "iteration": iteration_number, "urls": urls},
                token_count=token_count,
                priority=decision.priority
            )
        else:
            content_str = result if isinstance(result, str) else json.dumps(result)
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=self._get_content_type(decision),
                content=result,
                metadata={"decision_type": decision.decision_type.value, "iteration": iteration_number},
                token_count=token_count,
                priority=decision.priority
            )

    def _should_terminate(self, iteration: ResearchIteration) -> bool:
        terminate_decision = any(d.decision_type == DecisionType.TERMINATE for d in iteration.decisions_made)
        if terminate_decision:
            rprint("[green]Terminating: ReasoningAgent indicated completion.[/green]")
            return True
        try:
            reason_decisions = asyncio.run(self.reason_agent.analyze(self.current_context))
            search_decisions = asyncio.run(self.search_agent.analyze(self.current_context))
            explore_decisions = asyncio.run(self.explore_agent.analyze(self.current_context))
            valid_decisions = [
                d for d in (reason_decisions + search_decisions + explore_decisions)
                if (d.decision_type != DecisionType.SEARCH or 
                    d.context.get("query", "").strip() not in self.previous_searches)
            ]
            if not valid_decisions:
                rprint("[yellow]Terminating: No further valid decisions from any agent.[/yellow]")
                return True
        except Exception as e:
            rprint(f"[red]Error checking for new decisions: {str(e)}[/red]")
            return False
        return False

    def _get_agent_for_decision(self, decision: ResearchDecision) -> Optional[Any]:
        agent_map = {
            DecisionType.SEARCH: self.search_agent,
            DecisionType.EXPLORE: self.explore_agent,
            DecisionType.REASON: self.reason_agent
        }
        return agent_map.get(decision.decision_type)

    def _get_content_type(self, decision: ResearchDecision) -> ContentType:
        type_map = {
            DecisionType.SEARCH: ContentType.SEARCH_RESULT,
            DecisionType.EXPLORE: ContentType.EXPLORED_CONTENT,
            DecisionType.REASON: ContentType.ANALYSIS,
            DecisionType.TERMINATE: ContentType.OTHER
        }
        return type_map.get(decision.decision_type, ContentType.OTHER)

    async def _call_o3_mini_for_report(self, prompt: str) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",
                max_completion_tokens=self.reason_agent.max_output_tokens
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                start_idx = text.find("\n") + 1
                end_idx = text.rfind("```")
                if end_idx > start_idx:
                    text = text[start_idx:end_idx].strip()
                else:
                    text = text.replace("```", "").strip()
            return text
        except Exception as e:
            logger.error(f"Error in final paper LLM call: {e}")
            return "## Error generating comprehensive paper"

    async def _generate_comprehensive_paper_markdown(self, context: ResearchContext) -> str:
        search_items = context.get_content("main", ContentType.SEARCH_RESULT)
        explored_items = context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis_items = context.get_content("main", ContentType.ANALYSIS)
        combined_corpus = []
        combined_corpus.append("### Final Consolidated Research\n")
        combined_corpus.append("[SEARCH RESULTS]\n")
        for s in search_items:
            combined_corpus.append(str(s.content))
        combined_corpus.append("\n[EXPLORED CONTENT]\n")
        for e in explored_items:
            combined_corpus.append(str(e.content))
        combined_corpus.append("\n[ANALYSIS TEXT]\n")
        for a in analysis_items:
            if isinstance(a.content, dict):
                combined_corpus.append(a.content.get("analysis", ""))
            else:
                combined_corpus.append(str(a.content))
        big_text = "\n\n".join(combined_corpus)
        prompt = (
            "You are an advanced AI that just completed a comprehensive multi-step research.\n"
            "Now produce a SINGLE, richly detailed research paper in valid Markdown.\n"
            "Incorporate all relevant facts, context, analysis, and insights from the text below.\n\n"
            "Provide a thorough, well-structured breakdown:\n"
            "- Large headings\n"
            "- Subheadings\n"
            "- Bullet points\n"
            "- Tables if relevant\n"
            "- Detailed comparisons and references\n\n"
            "Return ONLY Markdown. RULE: ensure that all tables are valid Markdown tables. No extra JSON or placeholders.\n\n"
            "RESEARCH CORPUS:\n"
            f"{big_text}\n\n"
            "Please produce the final research paper in Markdown now:"
        ).strip()
        final_markdown = await self._call_o3_mini_for_report(prompt)
        return final_markdown

    async def _convert_markdown_to_pdf(self, markdown_text: str, out_path: Path):
        import markdown
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2em; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #eee; }}
                h2 {{ color: #34495e; margin-top: 2em; }}
                h3 {{ color: #7f8c8d; }}
                code {{ background: #f8f9fa; padding: 0.2em 0.4em; border-radius: 3px; }}
                pre {{ background: #f8f9fa; padding: 1em; border-radius: 5px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #ddd; margin: 0; padding-left: 1em; color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f8f9fa; }}
            </style>
        </head>
        <body>
            {markdown.markdown(markdown_text, extensions=['fenced_code', 'tables'])}
        </body>
        </html>
        """
        HTML(string=html_content).write_pdf(str(out_path))

    async def _generate_final_output(self) -> Dict[str, Any]:
        comprehensive_md = await self._generate_comprehensive_paper_markdown(self.current_context)
        if self.research_dir:
            pdf_path = self.research_dir / "research_paper.pdf"
            await self._convert_markdown_to_pdf(comprehensive_md, pdf_path)
        return {
            "paper": comprehensive_md,
            "title": self.current_context.initial_question,
            "sources": []
        }

    def _calculate_metrics(self, total_time: float) -> Dict[str, Any]:
        metrics = {
            "total_time": total_time,
            "iterations": len(self.iterations),
            "total_decisions": sum(len(it.decisions_made) for it in self.iterations),
            "total_content": sum(len(it.content_added) for it in self.iterations),
            "agent_metrics": self._get_agent_metrics()
        }
        metrics["iterations_data"] = [
            {
                "number": it.iteration_number,
                "time": it.metrics["iteration_time"],
                "decisions": len(it.decisions_made),
                "content": len(it.content_added)
            }
            for it in self.iterations
        ]
        return metrics

    def _get_agent_metrics(self) -> Dict[str, Any]:
        return {
            "search": self.search_agent.get_metrics(),
            "explore": self.explore_agent.get_metrics(),
            "reason": self.reason_agent.get_metrics()
        }