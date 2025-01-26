"""
Perplexity API client for web search functionality.
Uses the Perplexity API to perform intelligent web searches and extract relevant information.
"""

import os
import re
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class PerplexityClient:
    """
    A simple client that calls Perplexity's /chat/completions endpoint
    using requests. See https://docs.perplexity.ai/api-reference/chat-completions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.api_key = os.getenv("PERPLEXITY_API_KEY")
        if not self.api_key:
            raise ValueError("PERPLEXITY_API_KEY not set in environment variables.")
        self.base_url = "https://api.perplexity.ai/chat/completions"

        # Configuration
        self.config = config or {}
        self.model = self.config.get("model", "sonar")
        self.request_timeout = self.config.get("request_timeout", 60)  # 60s default
        self.system_message = "You are a research assistant helping to find accurate, up-to-date information."
        
        logger.info(
            "PerplexityClient initialized with model=%s, timeout=%d",
            self.model,
            self.request_timeout
        )

    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a web search using the Perplexity /chat/completions API
        by passing a short system message and a user message in the
        'messages' array, along with any relevant parameters.
        """

        # Build the messages array with system + user roles
        messages = [
            {
                "role": "system",
                "content": self.system_message
            },
            {
                "role": "user",
                "content": query
            }
        ]

        # Prepare the JSON data to send
        # (feel free to tweak parameters as needed, or add more)
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "top_p": 0.9,
            # Example: filter the search domain, or remove it if you want all results
            # "search_domain_filter": ["perplexity.ai"],
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "month",
            "top_k": 0,
            "stream": False,
            # presence_penalty and frequency_penalty can be adjusted as needed
            "presence_penalty": 0,
            "frequency_penalty": 1
        }

        # Log what we're about to send (for debug)
        logger.info("Sending request to Perplexity with query='%s' and model=%s", query, self.model)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=self.request_timeout  # Add timeout to prevent hanging
            )
            if not response.ok:
                # Log and raise an error if the HTTP status is not 200
                logger.error(
                    "Perplexity API returned an error %d: %s",
                    response.status_code,
                    response.text
                )
                response.raise_for_status()

            # Parse response JSON
            resp_json = response.json()

            # The assistant's text is typically in resp_json["choices"][0]["message"]["content"]
            # We'll combine it in 'content' for consistency.
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                content = resp_json["choices"][0]["message"].get("content", "")
            else:
                content = ""

            # Extract citations or references, if present
            urls = self._extract_urls(content)
            # Some Perplexity responses also have a top-level "citations" array
            # to show references. We could merge them into `urls` if we like:
            citations = resp_json.get("citations", [])
            for citation_url in citations:
                if citation_url not in urls:
                    urls.append(citation_url)

            return {
                "content": content,
                "urls": urls
            }

        except requests.RequestException as e:
            logger.exception("Error in Perplexity search request:")
            return {
                "content": "",
                "urls": [],
                "error": str(e)
            }

    def _extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from any text. We also attempt to parse typical references like:
        [Source: https://example.com]
        """
        citation_pattern = r'\[Source:\s*(https?://[^\]]+)\]'
        citation_urls = re.findall(citation_pattern, text)

        url_pattern = r'https?://\S+'
        raw_urls = re.findall(url_pattern, text)

        all_urls = list(set(citation_urls + raw_urls))
        return all_urls