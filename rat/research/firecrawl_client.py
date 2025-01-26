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

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class FirecrawlClient:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        # Configuration
        self.config = config or {}
        self.request_timeout = self.config.get("request_timeout", 60)  # 60s default
        
        # Initialize client (timeout will be used in requests)
        self.app = FirecrawlApp(api_key=self.api_key)
        
        logger.info(
            "FirecrawlClient initialized with timeout=%d",
            self.request_timeout
        )
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a single webpage using Firecrawl's /scrape endpoint.
        This is the primary method for single-page extraction, returning markdown.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info("Extracting content from URL: %s (single scrape)", url)
            
            # Make the request to scrape the URL with timeout
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                    'request_timeout': self.request_timeout
                }
            )
            
            return self._process_extracted_content(result.get('data', {}), url)
            
        except Exception as e:
            rprint(f"[red]Firecrawl API request failed (single scrape) for {url}: {str(e)}[/red]")
            logger.exception("Error extracting content from URL (single scrape): %s", url)
            return {
                "title": "",
                "text": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
            }

    def scrape_urls_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Batch scrape multiple URLs in one Firecrawl call using /batch/scrape.
        More efficient than multiple single-page scrapes when you have several URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of processed results, one per URL (preserving order)
        """
        if not urls:
            return []
            
        logger.info("Batch-scraping %d URLs", len(urls))
        
        try:
            # Ensure all URLs have protocols
            urls = [
                f"https://{url}" if not url.startswith(('http://', 'https://')) else url
                for url in urls
            ]
            
            # Call Firecrawl batch scrape
            result = self.app.batch_scrape_urls(
                urls,
                params={
                    'formats': ['markdown'],
                    'request_timeout': self.request_timeout
                }
            )
            
            # Process each page in the batch
            batch_data = result.get('data', [])
            processed_results = []
            
            for page_data in batch_data:
                # Each page_data is similar to a single scrape result
                processed = self._process_extracted_content(
                    page_data,
                    page_data.get('metadata', {}).get('sourceURL', '')
                )
                processed_results.append(processed)
                
            return processed_results
            
        except Exception as e:
            rprint(f"[red]Firecrawl batch scrape failed: {str(e)}[/red]")
            logger.exception("Error in batch_scrape_urls for URLs: %s", urls)
            # Return empty results for all URLs
            return [
                {
                    "title": "",
                    "text": "",
                    "metadata": {
                        "url": url,
                        "error": str(e)
                    }
                }
                for url in urls
            ]

    def extract_data(self, url: str, prompt: str) -> Dict[str, Any]:
        """
        Extract structured data from a webpage using Firecrawl's LLM capabilities.
        Uses the /scrape endpoint with formats=['json'] and a custom prompt.
        
        Args:
            url: The URL to extract data from
            prompt: Instructions for the LLM about what data to extract
            
        Returns:
            Dict containing the extracted structured data and metadata
        """
        try:
            logger.info(
                "Extracting structured data from URL: %s with prompt='%s'",
                url, prompt
            )
            
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['json'],
                    'jsonOptions': {
                        'prompt': prompt
                    },
                    'request_timeout': self.request_timeout
                }
            )
            
            data = result.get('data', {})
            extracted = data.get('json', {})
            meta = data.get('metadata', {})
            
            return {
                'url': meta.get('sourceURL', url),
                'extracted_fields': extracted,
                'metadata': meta
            }
            
        except Exception as e:
            rprint(f"[red]Firecrawl structured extraction failed for {url}: {str(e)}[/red]")
            logger.exception("Error extracting structured data from URL: %s", url)
            return {
                'url': url,
                'extracted_fields': {},
                'metadata': {'error': str(e)}
            }
            
    def _process_extracted_content(self, data: Dict[str, Any], original_url: str) -> Dict[str, Any]:
        """
        Process and clean the extracted content.
        
        Args:
            data: Raw API response data
            original_url: The original URL that was scraped
            
        Returns:
            Processed and cleaned content
        """
        metadata = data.get("metadata", {})
        markdown_content = data.get("markdown", "")
        
        processed = {
            "title": metadata.get("title", metadata.get("ogTitle", "")),
            "text": self._clean_text(markdown_content),
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": metadata.get("publishedDate", ""),
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        
        return processed
        
    def _clean_text(self, text: str) -> str:
        """
        Clean and format extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and formatted text
        """
        if not text:
            return ""
            
        # Remove extra whitespace while preserving markdown formatting
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            # Preserve markdown headings and lists
            if line.strip().startswith(("#", "-", "*", "1.", ">")):
                cleaned_lines.append(line)
            else:
                # Clean normal text lines
                cleaned = " ".join(line.split())
                if cleaned:
                    cleaned_lines.append(cleaned)
        
        return "\n\n".join(cleaned_lines)
