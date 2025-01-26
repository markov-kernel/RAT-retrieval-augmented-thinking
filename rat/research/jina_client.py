"""
Jina Reader client for web scraping functionality.
This module handles interactions with the Jina Reader API for extracting content
from web pages and processing the extracted data.
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from rich import print as rprint

load_dotenv()

class JinaClient:
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")
        
        self.base_url = "https://api.jina.ai/reader/crawl"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Respond-With": "markdown",  # Get markdown format
            "X-With-Generated-Alt": "true",  # Enable alt text generation
            "X-With-Images-Summary": "true",  # Include image summaries
            "X-With-Links-Summary": "true"  # Include link summaries
        }
        
    def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Extract content from a webpage using Jina Reader API.
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            payload = {
                "url": url,
                "engine": "readerlm-v2",  # Use the latest reader model
                "retainImages": "all",  # Keep all images
                "withGeneratedAlt": True,  # Generate alt text for images
                "withImagesSummary": True,  # Include image summaries
                "withLinksSummary": True,  # Include link summaries
                "noCache": False,  # Use caching for better performance
                "timeout": 60  # 60 second timeout
            }
            
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
            data = response.json()
            return self._process_extracted_content(data)
            
        except requests.exceptions.RequestException as e:
            rprint(f"[red]Jina Reader API request failed: {str(e)}[/red]")
            return {
                "title": "",
                "text": "",
                "metadata": {},
                "error": str(e)
            }
            
    def _process_extracted_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean the extracted content.
        
        Args:
            data: Raw API response data
            
        Returns:
            Processed and cleaned content
        """
        # Extract from the correct response structure
        content = data.get("data", "")  # Content is in the data field
        meta = data.get("meta", {})
        
        processed = {
            "title": meta.get("title", ""),
            "text": content,  # Main content is in data field
            "metadata": {
                "author": meta.get("author", ""),
                "published_date": meta.get("published_date", ""),
                "url": meta.get("url", ""),
                "domain": meta.get("domain", ""),
                "word_count": meta.get("word_count", 0),
                "language": meta.get("language", "")
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