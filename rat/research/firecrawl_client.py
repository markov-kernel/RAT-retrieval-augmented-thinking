"""
Firecrawl client for web scraping functionality.
This module handles interactions with the Firecrawl API for extracting content
from web pages and processing the extracted data.

Key Features:
1. Single-page extraction (primary /scrape endpoint)
2. Batch scraping for multiple URLs
3. Optional LLM-based structured data extraction
4. Content cleaning and formatting
"""

from typing import Dict, List, Any, Optional
import os
import logging
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

logger = logging.getLogger(__name__)

class FirecrawlClient:
    """
    Client for interacting with the Firecrawl API.
    Handles webpage scraping and content extraction.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Firecrawl client.
        
        Args:
            api_key: Optional API key for Firecrawl. If not provided, will look for FIRECRAWL_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("Firecrawl API key is required. Set FIRECRAWL_API_KEY env var or pass to constructor.")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single URL.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing the extracted content and metadata
        """
        try:
            result = self.app.scrape_url(url, params={
                'formats': ['markdown', 'html']
            })
            
            # Convert to our expected format
            return {
                "text": result.get("markdown", ""),
                "html": result.get("html", ""),
                "metadata": result.get("metadata", {})
            }
            
        except Exception as e:
            logger.exception(f"Error extracting content from {url}")
            return {
                "error": str(e),
                "text": "",
                "html": "",
                "metadata": {"url": url}
            }
    
    def scrape_urls_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape multiple URLs in batch mode.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of results, one per URL
        """
        try:
            # Use the batch scraping endpoint
            results = self.app.batch_scrape_urls(urls, {
                'formats': ['markdown', 'html']
            })
            
            # Convert to list of our expected format
            processed_results = []
            for result in results.get("data", []):
                processed_results.append({
                    "text": result.get("markdown", ""),
                    "html": result.get("html", ""),
                    "metadata": result.get("metadata", {})
                })
            
            return processed_results
            
        except Exception as e:
            logger.exception(f"Error batch scraping URLs")
            # Return error results for each URL
            return [
                {
                    "error": str(e),
                    "text": "",
                    "html": "",
                    "metadata": {"url": url}
                }
                for url in urls
            ]

    def extract_data(self, url: str, prompt: str) -> Dict[str, Any]:
        """
        Extract structured data from a webpage using LLM processing.
        
        Args:
            url: The URL to extract data from
            prompt: The prompt for structured data extraction
            
        Returns:
            Dict containing extracted structured data
        """
        logger.info("Extracting structured data from %s with prompt", url)
        try:
            response = self.app.extract(url, prompt, timeout=self.request_timeout)
            logger.debug("Successfully extracted structured data from %s", url)
            return response
        except Exception as e:
            logger.error("Failed to extract structured data from %s: %s", url, str(e))
            raise

    def _process_extracted_content(self, data: Dict[str, Any], original_url: str) -> Dict[str, Any]:
        """Process and clean the extracted content."""
        if not data or not isinstance(data, dict):
            logger.warning("Invalid data received for URL %s", original_url)
            return {
                "url": original_url,
                "content": "",
                "success": False,
                "error": "Invalid response data"
            }

        content = data.get("content", "")
        if not content:
            logger.warning("No content extracted from %s", original_url)
            return {
                "url": original_url,
                "content": "",
                "success": False,
                "error": "No content extracted"
            }

        cleaned_content = self._clean_text(content)
        logger.debug("Processed content from %s: %d chars -> %d chars", 
                    original_url, len(content), len(cleaned_content))
        
        return {
            "url": original_url,
            "content": cleaned_content,
            "success": True,
            "metadata": data.get("metadata", {})
        }

    def _clean_text(self, text: str) -> str:
        """Clean and format extracted text content."""
        if not text:
            return ""
            
        # Basic cleaning operations
        cleaned = text.strip()
        
        logger.debug("Cleaned text: %d chars -> %d chars", 
                    len(text), len(cleaned))
        return cleaned
