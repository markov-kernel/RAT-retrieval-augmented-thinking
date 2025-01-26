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
from urllib.parse import quote

load_dotenv()

class JinaClient:
    def __init__(self):
        self.api_key = os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not found in environment variables")
        
        self.base_url = "https://r.jina.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json"
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
            # URL encode the target URL and append it to the base URL
            # The format should be: https://r.jina.ai/https://example.com
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            full_url = f"{self.base_url}/{url}"
            
            # Make the request
            response = requests.get(
                full_url,
                headers=self.headers,
                timeout=60  # 60 second timeout
            )
            response.raise_for_status()
            
            # Parse the response
            if response.headers.get('content-type', '').startswith('application/json'):
                data = response.json()
            else:
                # Extract text content
                text = response.text
                # Basic cleaning of HTML if present
                text = text.replace('<br>', '\n').replace('</p>', '\n\n')
                data = {"content": text}
                
            return self._process_extracted_content(data)
            
        except requests.exceptions.RequestException as e:
            rprint(f"[red]Jina Reader API request failed for {url}: {str(e)}[/red]")
            return {
                "title": "",
                "text": "",
                "metadata": {
                    "url": url,
                    "error": str(e)
                }
            }
            
    def _process_extracted_content(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean the extracted content.
        
        Args:
            data: Raw API response data
            
        Returns:
            Processed and cleaned content
        """
        # Extract content and metadata from response
        content = data.get("content", "")
        if isinstance(content, dict):
            # Handle structured content
            text = content.get("text", "")
            title = content.get("title", "")
            metadata = content.get("metadata", {})
        else:
            # Handle plain text content
            text = str(content)
            title = ""
            metadata = {}
        
        processed = {
            "title": title,
            "text": text,
            "metadata": {
                "url": metadata.get("url", ""),
                "author": metadata.get("author", ""),
                "published_date": metadata.get("published_date", ""),
                "domain": metadata.get("domain", ""),
                "word_count": len(text.split()) if text else 0,
                "language": metadata.get("language", "")
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