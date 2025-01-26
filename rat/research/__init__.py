"""
RAT Research package initialization.
This module provides research capabilities for the RAT system.
"""

from .perplexity_client import PerplexityClient
from .firecrawl_client import FirecrawlClient
from .orchestrator import ResearchOrchestrator
from .output_manager import OutputManager

__all__ = [
    'PerplexityClient',
    'FirecrawlClient',
    'ResearchOrchestrator',
    'OutputManager'
]
