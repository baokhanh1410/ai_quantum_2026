"""API Client for SJC Gold Price data (CSV history)."""

import logging
from data_engine.api.base_client import BaseAPIClient
from core.config.settings import settings

logger = logging.getLogger("data_engine.api.gold")

class GoldClient(BaseAPIClient):
    """Client for pulling historical SJC gold prices from SJC-price GitHub repository."""

    def __init__(self):
        gold_config = settings.apis.sjc_crawler
        super().__init__()
        self.url = gold_config.url
        self.method = gold_config.get("method", "GET")

    def fetch_gold_csv(self) -> str:
        """Fetches the raw SJC gold historical CSV data.

        Returns:
            The raw CSV content as a string.
        """
        logger.info(f"Fetching SJC Gold history CSV from {self.url}")
        response = self.request(self.method, self.url)
        return response.text
