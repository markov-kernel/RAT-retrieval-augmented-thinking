"""
Explore agent for extracting content from URLs.
Now acts as a simple executor that processes EXPLORE decisions from the ReasoningAgent.
"""

from typing import List, Dict, Any, Optional
import logging
from rich import print as rprint

from .base import BaseAgent, ResearchDecision, DecisionType
from .context import ResearchContext, ContentType
from ..firecrawl_client import FirecrawlClient

logger = logging.getLogger(__name__)

class ExploreAgent(BaseAgent):
    """
    Agent responsible for extracting content from URLs.
    Acts as an executor for EXPLORE decisions made by the ReasoningAgent.
    """
    
    def __init__(self, firecrawl_client: FirecrawlClient, config=None):
        super().__init__("explore", config)
        self.firecrawl = firecrawl_client
        self.logger = logging.getLogger(__name__)
        
    def analyze(self, context: ResearchContext) -> List[ResearchDecision]:
        """
        No longer generates decisions - URL selection is handled by ReasoningAgent.
        """
        return []
        
    def can_handle(self, decision: ResearchDecision) -> bool:
        """Check if this agent can handle a decision."""
        return decision.decision_type == DecisionType.EXPLORE
        
    def execute_decision(self, decision: ResearchDecision) -> Dict[str, Any]:
        """
        Execute an EXPLORE decision by scraping the URL.
        
        Args:
            decision: The EXPLORE decision containing the URL to scrape
            
        Returns:
            Dict containing the scraped content
        """
        url = decision.context["url"]
        self.logger.info(f"Exploring URL: {url}")
        
        try:
            scrape_result = self.firecrawl.extract_content(url)
            
            if not scrape_result:
                self.logger.warning(f"No content extracted from URL: {url}")
                return {}
                
            return {
                "url": url,
                "title": scrape_result.get("title", ""),
                "text": scrape_result.get("text", ""),
                "metadata": {
                    **scrape_result.get("metadata", {}),
                    "relevance": decision.context.get("relevance", 0.0),
                    "rationale": decision.context.get("rationale", "")
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error exploring URL {url}: {str(e)}")
            return {
                "url": url,
                "error": str(e)
            }
