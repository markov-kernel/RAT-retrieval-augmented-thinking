"""
Manager for coordinating the multi-agent research workflow.
This module implements a central ResearchManager that:
  - Initializes a ResearchContext and persists it to a JSON file.
  - Dispatches agent decisions as concurrent tasks (using AgentTask wrappers).
  - Updates the context as agent tasks complete.
  - Generates a comprehensive research paper upon termination.
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from openai import OpenAI
from rich import print as rprint

from rat.research.agents.search import SearchAgent
from rat.research.agents.explore import ExploreAgent
from rat.research.agents.reason import ReasoningAgent
from rat.research.agents.base import ResearchDecision, DecisionType
from rat.research.agents.context import ResearchContext, ContentType, ContentItem
from rat.research.perplexity_client import PerplexityClient
from rat.research.firecrawl_client import FirecrawlClient
from rat.research.output_manager import OutputManager

logger = logging.getLogger(__name__)


class AgentTask:
    """
    Wrapper for an agent decision execution.
    Executes the decision via the appropriate agent and calls a callback
    to update the research context.
    """
    def __init__(self, decision: ResearchDecision, agent, callback):
        self.decision = decision
        self.agent = agent
        self.callback = callback

    async def run(self):
        """Execute the decision asynchronously."""
        try:
            result = await self.agent.execute_decision(self.decision)
            self.callback(self.decision, result)
            return result
        except Exception as e:
            logger.error(f"Error executing decision: {str(e)}")
            return {"error": str(e)}


class ResearchManager:
    """
    Central manager for the multi-agent research process.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.perplexity = PerplexityClient()
        self.firecrawl = FirecrawlClient()
        self.search_agent = SearchAgent(self.perplexity, self.config.get("search_config", {}))
        self.explore_agent = ExploreAgent(self.firecrawl, self.config.get("explore_config", {}))
        self.reason_agent = ReasoningAgent(self.config.get("reason_config", {}))
        self.output_manager = OutputManager()
        self.max_iterations = self.config.get("max_iterations", 5)
        self.current_context: Optional[ResearchContext] = None
        self.research_dir: Optional[Path] = None
        self.previous_searches = set()

    async def start_research(self, question: str) -> Dict[str, Any]:
        start_time = time.time()
        self.research_dir = self.output_manager.create_research_dir(question)
        self.current_context = ResearchContext(initial_question=question)
        self.persist_context()
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"Starting iteration {iteration}")
            decisions = await self.collect_decisions()
            if not decisions:
                logger.info("No new decisions, terminating research")
                break
            if any(d.decision_type == DecisionType.TERMINATE for d in decisions):
                logger.info("Terminate decision received")
                break
            await self.dispatch_decisions(decisions, iteration)
            self.persist_context()
            if await self._should_terminate():
                logger.info("Termination condition met based on context")
                break
        final_output = await self._generate_final_output()
        total_time = time.time() - start_time
        final_output["metrics"] = {
            "total_time": total_time,
            "iterations": iteration
        }
        if self.research_dir:
            self.output_manager.save_research_paper(self.research_dir, final_output)
        return final_output

    async def collect_decisions(self) -> List[ResearchDecision]:
        decisions = []
        try:
            reason_decisions = await self.reason_agent.analyze(self.current_context)
        except Exception as e:
            logger.error(f"Error in reason agent analysis: {e}")
            reason_decisions = []
        try:
            search_decisions = await self.search_agent.analyze(self.current_context)
        except Exception as e:
            logger.error(f"Error in search agent analysis: {e}")
            search_decisions = []
        try:
            explore_decisions = await self.explore_agent.analyze(self.current_context)
        except Exception as e:
            logger.error(f"Error in explore agent analysis: {e}")
            explore_decisions = []
        decisions.extend(reason_decisions)
        decisions.extend(search_decisions)
        decisions.extend(explore_decisions)
        filtered = []
        for d in decisions:
            if d.decision_type == DecisionType.SEARCH:
                query = d.context.get("query", "").strip()
                if query in self.previous_searches:
                    logger.info(f"Skipping duplicate search: {query}")
                    continue
                else:
                    self.previous_searches.add(query)
            filtered.append(d)
        filtered.sort(key=lambda d: d.priority, reverse=True)
        return filtered

    async def dispatch_decisions(self, decisions: List[ResearchDecision], iteration: int):
        tasks = []
        for decision in decisions:
            if decision.decision_type == DecisionType.TERMINATE:
                continue
            agent = self._select_agent(decision)
            if not agent:
                continue
            task = AgentTask(decision, agent, self.update_context_with_result)
            tasks.append(asyncio.create_task(task.run()))
        if tasks:
            await asyncio.gather(*tasks)

    def update_context_with_result(self, decision: ResearchDecision, result: Dict[str, Any]):
        content_item = self._create_content_item(decision, result)
        if content_item and self.current_context:
            try:
                self.current_context.add_content("main", content_item=content_item)
            except Exception as e:
                logger.error(f"Error updating context: {e}")

    def _create_content_item(self, decision: ResearchDecision, result: Dict[str, Any]) -> ContentItem:
        if decision.decision_type == DecisionType.SEARCH:
            content_str = result.get("content", "")
            urls = result.get("urls", [])
            token_count = self.current_context._estimate_tokens(content_str)
            return ContentItem(
                content_type=ContentType.SEARCH_RESULT,
                content=content_str,
                metadata={"decision_type": decision.decision_type.value, "urls": urls},
                token_count=token_count,
                priority=decision.priority
            )
        else:
            if isinstance(result, dict):
                content_str = result.get("analysis", json.dumps(result))
            else:
                content_str = str(result)
            token_count = self.current_context._estimate_tokens(content_str)
            content_type = (ContentType.EXPLORED_CONTENT if decision.decision_type == DecisionType.EXPLORE 
                            else ContentType.ANALYSIS)
            return ContentItem(
                content_type=content_type,
                content=result,
                metadata={"decision_type": decision.decision_type.value},
                token_count=token_count,
                priority=decision.priority
            )

    def _select_agent(self, decision: ResearchDecision):
        if decision.decision_type == DecisionType.SEARCH:
            return self.search_agent
        elif decision.decision_type == DecisionType.EXPLORE:
            return self.explore_agent
        elif decision.decision_type == DecisionType.REASON:
            return self.reason_agent
        else:
            return None

    async def _should_terminate(self) -> bool:
        if self.current_context:
            contents = self.current_context.get_content("main")
            if len(contents) >= self.config.get("min_new_content", 3):
                return True
        try:
            reason_decisions = await self.reason_agent.analyze(self.current_context)
            search_decisions = await self.search_agent.analyze(self.current_context)
            explore_decisions = await self.explore_agent.analyze(self.current_context)
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

    def persist_context(self):
        if self.current_context and self.research_dir:
            context_file = self.research_dir / "research_context.json"
            self.current_context.save_to_file(str(context_file))
            logger.info(f"Context persisted to {context_file}")

    async def _generate_comprehensive_markdown(self) -> str:
        search_items = self.current_context.get_content("main", ContentType.SEARCH_RESULT)
        explored_items = self.current_context.get_content("main", ContentType.EXPLORED_CONTENT)
        analysis_items = self.current_context.get_content("main", ContentType.ANALYSIS)
        corpus = []
        corpus.append("### Consolidated Research\n")
        corpus.append("#### Search Results\n")
        for item in search_items:
            corpus.append(str(item.content))
        corpus.append("\n#### Explored Content\n")
        for item in explored_items:
            corpus.append(str(item.content))
        corpus.append("\n#### Analysis\n")
        for item in analysis_items:
            if isinstance(item.content, dict):
                corpus.append(item.content.get("analysis", ""))
            else:
                corpus.append(str(item.content))
        big_text = "\n\n".join(corpus)
        prompt = (
            "You are an advanced AI tasked with generating a comprehensive research paper in Markdown. "
            "Using the following research corpus, produce a detailed, well-structured paper with headings, subheadings, bullet points, and tables if necessary.\n\n"
            "RESEARCH CORPUS:\n"
            f"{big_text}\n\n"
            "Please produce the final research paper in Markdown:"
        )
        
        try:
            # Use the new OpenAI API format
            client = OpenAI()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="o3-mini",
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="high",
                max_completion_tokens=50000
            )
            final_markdown = response.choices[0].message.content.strip()
            return final_markdown
        except Exception as e:
            logger.error(f"Error generating comprehensive markdown: {str(e)}")
            return "Error generating research paper. Please try again."

    async def _generate_final_output(self) -> Dict[str, Any]:
        comprehensive_md = await self._generate_comprehensive_markdown()
        if self.research_dir:
            pdf_path = self.research_dir / "research_paper.pdf"
            await self._convert_markdown_to_pdf(comprehensive_md, pdf_path)
        return {
            "paper": comprehensive_md,
            "title": self.current_context.initial_question,
            "sources": []
        }

    async def _convert_markdown_to_pdf(self, markdown_text: str, out_path: Path):
        import markdown
        from weasyprint import HTML
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