"""
Firecrawl client for web scraping functionality.
Now uses asyncio.to_thread to wrap blocking calls.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv
from rich import print as rprint
from firecrawl import FirecrawlApp
import logging
import asyncio

load_dotenv()

api_logger = logging.getLogger('api.firecrawl')


class FirecrawlClient:
    def __init__(self):
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY not found in environment variables")
        self.app = FirecrawlApp(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    async def extract_content(self, url: str) -> Dict[str, Any]:
        """
        Asynchronously extract content from a webpage.
        """
        api_logger.info(f"Firecrawl API Request - URL: {url}")
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                api_logger.debug(f"Added https:// protocol to URL: {url}")
            result = await asyncio.to_thread(self.app.scrape_url, url, params={'formats': ['markdown']})
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
        metadata = data.get("metadata", {})
        markdown_content = data.get("markdown", "")
        processed = {
            "title": metadata.get("title", metadata.get("ogTitle", "")),
            "text": markdown_content,
            "metadata": {
                "url": metadata.get("sourceURL", original_url),
                "author": metadata.get("author", ""),
                "published_date": "",
                "domain": metadata.get("ogSiteName", ""),
                "word_count": len(markdown_content.split()) if markdown_content else 0,
                "language": metadata.get("language", ""),
                "status_code": metadata.get("statusCode", 200)
            }
        }
        if processed["text"]:
            processed["text"] = self._clean_text(processed["text"])
            api_logger.debug(f"Cleaned text for {original_url}: {len(processed['text'])} chars")
        else:
            api_logger.warning(f"No text content extracted from {original_url}")
        return processed

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        lines = text.split("\n")
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith(("#", "-", "*", "1.", ">")):
                cleaned_lines.append(line)
            else:
                cleaned = " ".join(line.split())
                if cleaned:
                    cleaned_lines.append(cleaned)
        return "\n\n".join(cleaned_lines)