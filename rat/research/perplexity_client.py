"""
Perplexity API client for web search functionality.
Now uses async OpenAI API calls.
"""

import os
import re
import json
import logging
import openai
from typing import List, Dict, Any
from rich import print as rprint
from dotenv import load_dotenv
import asyncio
from openai import OpenAI

load_dotenv()
api_logger = logging.getLogger('api.perplexity')


class PerplexityClient:
    def __init__(self):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        self.client = openai
        self.client.api_key = self.api_key
        self.client.api_base = "https://api.perplexity.ai"
        self.model = "sonar-pro"
        self.system_message = (
            "You are a research assistant helping to find accurate and up-to-date information. "
            "When providing information, always cite your sources in the format [Source: URL]. "
            "Focus on finding specific, factual information and avoid speculation."
        )

    async def search(self, query: str) -> Dict[str, Any]:
        """
        Perform an asynchronous web search using the Perplexity API via requests.
        """
        api_logger.info(f"Perplexity API Request - Query: {query}")
        payload = {
            "model": "sonar-reasoning",
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": query}
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1,
            "response_format": None
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = "https://api.perplexity.ai/chat/completions"

        try:
            response = await asyncio.to_thread(
                lambda: __import__('requests').post(url, json=payload, headers=headers)
            )
            if response.status_code != 200:
                api_logger.error(f"Perplexity API error: Error code: {response.status_code} - {response.text}")
                return {
                    "content": "",
                    "urls": [],
                    "query": query,
                    "metadata": {}
                }

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            urls = self._extract_urls(content)
            api_logger.debug(f"Response data: {json.dumps({'content': content, 'urls': urls}, indent=2)}")
            return {
                "content": content,
                "urls": urls,
                "query": query,
                "metadata": {
                    "model": "sonar",
                    "usage": data.get("usage", {})
                }
            }
        except Exception as e:
            api_logger.error(f"Perplexity API error: {str(e)}")
            return {
                "content": "",
                "urls": [],
                "query": query,
                "metadata": {}
            }

    def _extract_urls(self, text: str) -> List[str]:
        citation_pattern = r'$begin:math:display$Source: (https?://[^$end:math:display$]+)\]'
        citation_urls = re.findall(citation_pattern, text)
        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)
        all_urls = list(set(citation_urls + raw_urls))
        return all_urls

    async def validate_url(self, url: str) -> bool:
        import requests
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url)
            if not all([parsed.scheme, parsed.netloc]):
                return False
            return await asyncio.to_thread(lambda: requests.head(url, timeout=5).status_code == 200)
        except Exception:
            return False