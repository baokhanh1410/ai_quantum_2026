"""Base API Client with shared HTTP request session, timeout, and exception wrappers."""

import logging
import requests
from typing import Any, Dict, Optional
from data_engine.utils.retry import retry
from core.utils.exceptions import APIConnectionError

logger = logging.getLogger("data_engine.api.base_client")

class BaseAPIClient:
    """Base client class handling requests and connection errors."""

    def __init__(self, default_headers: Optional[Dict[str, str]] = None, timeout: int = 15):
        self.session = requests.Session()
        if default_headers:
            self.session.headers.update(default_headers)
        self.timeout = timeout

    @retry(max_retries=3, backoff_factor=2.0, exceptions=(requests.RequestException, APIConnectionError))
    def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> requests.Response:
        """Executes an HTTP request with automatic retry and exception wrapping.

        Args:
            method: HTTP verb (GET, POST, etc.)
            url: Target URL.
            params: Query parameters.
            data: Raw request body.
            json: JSON body.
            headers: Request-specific headers.
            timeout: Request timeout override.

        Returns:
            The HTTP response object.
        """
        req_timeout = timeout if timeout is not None else self.timeout
        try:
            logger.debug(f"Calling API: {method} {url} with params={params}")
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json,
                headers=headers,
                timeout=req_timeout
            )
            response.raise_for_status()
            return response
        except requests.Timeout as te:
            logger.error(f"API Request Timeout to {url}: {te}")
            raise APIConnectionError(f"API request timed out: {te}") from te
        except requests.HTTPError as he:
            logger.error(f"API HTTP error for {url}: {he.response.status_code} - {he.response.text}")
            raise APIConnectionError(f"HTTP {he.response.status_code}: {he}") from he
        except requests.RequestException as re:
            logger.error(f"API Request Exception to {url}: {re}")
            raise APIConnectionError(f"Network error calling API: {re}") from re
