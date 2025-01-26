"""
Firecrawl client for web scraping functionality.
This module handles interactions with the Firecrawl API for extracting content
from web pages and processing the extracted data.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging

load_dotenv()

# Get API logger
api_logger = logging.getLogger('api.firecrawl')

class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        
        self.app = FirecrawlApp(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage using Firecrawl API.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        api_logger.info(f"Firecrawl API Request - URL: {url}")
        
        try:
            # Ensure URL has protocol
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                api_logger.debug(f"Added https:// protocol to URL: {url}")
            
            # Make the request to scrape the URL
            result = self.app.scrape_url(
                url,
                params={
                    'formats': ['markdown'],
                }
            )
            
            processed_result = self._process_extracted_content(result.get('data', {}), url)
            api_logger.debug(f"Processed content from {url}: {len(processed_result.get('text', ''))} chars")
            return processed_result
            
        except Exception as e:
            api_logger.error(f"Firecrawl API request failed for {url}: {str(e)}")
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
            api_logger.debug(f"Cleaned text for {original_url}: {len(processed['text'])} chars")
        else:
            api_logger.warning(f"No text content extracted from {original_url}")
        
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
