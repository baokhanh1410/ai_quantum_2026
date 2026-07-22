"""API Client for crawling SBV (State Bank of Vietnam) interbank interest rates."""

import logging
from typing import Any, Dict, Optional
from data_engine.api.base_client import BaseAPIClient
from core.config.settings import settings

logger = logging.getLogger("data_engine.api.sbv")

class SBVClient(BaseAPIClient):
    """Client for crawling interbank rates from SBV portal API."""

    def __init__(self):
        sbv_config = settings.apis.sbv_crawler
        headers = sbv_config.get("headers", {})
        # Headers might be a ConfigNode, let's extract dict
        if hasattr(headers, "to_dict"):
            headers_dict = headers.to_dict()
        else:
            headers_dict = dict(headers)
            
        super().__init__(default_headers=headers_dict)
        self.url = sbv_config.url
        self.default_params = sbv_config.get("default_params", {})
        if hasattr(self.default_params, "to_dict"):
            self.default_params = self.default_params.to_dict()
        else:
            self.default_params = dict(self.default_params)

    def fetch_rates(self, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
        """Fetches a page of rates records from SBV headless CMS api.

        Args:
            page: Page number to query.
            page_size: Number of records to return per page.

        Returns:
            The raw JSON response dictionary.
        """
        params = {
            "page": page,
            "pageSize": page_size,
            "sort": self.default_params.get("sort", "datePublished:desc")
        }
        logger.info(f"Fetching SBV rates from {self.url} with params={params}")
        response = self.request("GET", self.url, params=params)
        return response.json()
