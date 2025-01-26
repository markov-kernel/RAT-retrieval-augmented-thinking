"""
Perplexity API client for web search functionality.
Uses the Perplexity API to perform intelligent web searches and extract relevant information.
"""

import os
import re
from openai import OpenAI
from typing import List, Dict, Any
from rich import print as rprint

class PerplexityClient:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("PERPLEXITY_API_KEY"),
            base_url="https://api.perplexity.ai"
        )
        
        self.model = "sonar-pro"
        self.system_message = (
            "You are a research assistant helping to find accurate and up-to-date information. "
            "When providing information, always cite your sources in the format [Source: URL]. "
            "Focus on finding specific, factual information and avoid speculation."
        )
        
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Perplexity API.
        
        Args:
            query: The search query
            
        Returns:
            Dict containing search results and extracted URLs
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_message},
                    {"role": "user", "content": query}
                ],
                temperature=0.7,
                stream=False
            )
            
            content = response.choices[0].message.content
            urls = self._extract_urls(content)
            
            return {
                "content": content,
                "urls": urls
            }
            
        except Exception as e:
            rprint(f"[red]Error in Perplexity search: {str(e)}[/red]")
            return {
                "content": "",
                "urls": []
            }
            
    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text, including those in citation format.
        
        Args:
            text: Text to extract URLs from
            
        Returns:
            List of extracted URLs
        """
        # Look for URLs in citation format [Source: URL]
        citation_pattern = r'\[Source: (https?://[^\]]+)\]'
        citation_urls = re.findall(citation_pattern, text)
        
        # Also look for raw URLs
        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)
        
        # Combine and deduplicate URLs
        all_urls = list(set(citation_urls + raw_urls))
        return all_urls

    def validate_url(self, url: str) -> bool:
        """
        Validate if a URL is accessible and safe to scrape.
        """
        import requests
        from urllib.parse import urlparse
        
        try:
            # Parse URL
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
                
            # Check if URL is accessible
            response = requests.head(url, timeout=5)
            return response.status_code == 200
            
        except Exception:
            return False 