"""
Firecrawl client for web scraping functionality.
This module handles interactions with the Firecrawl API for extracting content
from web pages and processing the extracted data.
"""

import os
from typing import Dict, Any, Optional
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
        Extract content from a webpage using Firecrawl API.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            logger.info("Extracting content from URL: %s", url)
            
            # Make the request to scrape the URL with timeout
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                    'request_timeout': self.request_timeout  # Pass timeout in params
                }
            )
            
            return self._process_extracted_content(result.get('data', {}), url)
            
        except Exception as e:
            rprint(f"[red]Firecrawl API request failed for {url}: {str(e)}[/red]")
            logger.exception("Error extracting content from URL: %s", url)
            return {
                "title": "",
                "text": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
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
            "text": markdown_content,
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": "",  # Firecrawl doesn't provide this directly
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        
        # Clean and format the text if needed
        if processed["text"]:
            processed["text"] = self._clean_text(processed["text"])
        
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
