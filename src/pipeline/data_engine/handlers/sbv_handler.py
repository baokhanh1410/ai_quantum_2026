"""Handler for formatting raw SBV interbank interest rates."""

import logging
import datetime
from typing import List, Dict, Any
from core.config.settings import settings

logger = logging.getLogger("data_engine.handlers.sbv")

class SBVHandler:
    """Formats SBV headless API JSON response into unified records."""

    def __init__(self):
        sbv_config = settings.apis.sbv_crawler
        self.maturity_mapping = sbv_config.get("maturity_mapping", {})
        if hasattr(self.maturity_mapping, "to_dict"):
            self.maturity_mapping = self.maturity_mapping.to_dict()
        else:
            self.maturity_mapping = dict(self.maturity_mapping)

    def _parse_date(self, date_str: str) -> datetime.datetime:
        """Parses SBV date and adjusts from UTC to GMT+7 (Vietnam Time)."""
        if not date_str:
            raise ValueError("Empty date string")
        try:
            if "T" in date_str:
                # Remove Z and parse ISO
                clean_str = date_str.replace("Z", "")
                # Clean millisecond parts if they exist
                if "." in clean_str:
                    clean_str = clean_str.split(".")[0]
                dt = datetime.datetime.strptime(clean_str, "%Y-%m-%dT%H:%M:%S")
                # Add 7 hours to convert to ICT timezone
                dt_vn = dt + datetime.timedelta(hours=7)
                return dt_vn.replace(hour=0, minute=0, second=0, microsecond=0)
            else:
                dt = datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
                return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        except Exception as e:
            logger.debug(f"Failed to parse date '{date_str}': {e}")
            # Try parsing simple YYYY-MM-DD
            try:
                return datetime.datetime.strptime(date_str[:10], "%Y-%m-%d")
            except Exception:
                raise ValueError(f"Unsupported date format: {date_str}")

    def format_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Transforms raw SBV JSON payload to unified database records.

        Args:
            data: Raw JSON dict from SBV client.

        Returns:
            List of OHLCV records with ticker names mapped to VNIBOR_*.
        """
        items = data.get("items", [])
        if not items:
            return []

        records = []
        for item in items:
            # Determine apply date
            date_val = None
            for field in item.get("contentFields", []):
                if field.get("name") == "ngayApDung":
                    date_val = field.get("contentFieldValue", {}).get("data")
                    break
            if not date_val:
                date_val = item.get("datePublished", "")

            try:
                timestamp = self._parse_date(date_val)
            except Exception as e:
                logger.warning(f"Skipping SBV item due to date parsing error: {e}")
                continue

            for field in item.get("contentFields", []):
                nested = field.get("nestedContentFields", [])
                if nested and any(nf.get("name") == "thoihan" for nf in nested):
                    term = None
                    rate = None
                    sales = None
                    for nf in nested:
                        name = nf.get("name")
                        val = nf.get("contentFieldValue", {}).get("data")
                        if name == "thoihan":
                            term = val
                        elif name == "laiSuatBQLienNganHang" and val is not None:
                            try:
                                rate = float(str(val).strip().replace(",", "."))
                            except ValueError:
                                rate = None
                        elif name == "dointhieu" or name == "doanhSo" and val is not None:
                            try:
                                sales = float(str(val).strip().replace(",", ""))
                            except ValueError:
                                sales = None
                    
                    if term in self.maturity_mapping and rate is not None:
                        ticker = self.maturity_mapping[term]
                        # Convert to standard format
                        # Lãi suất liên ngân hàng là single_value -> open=high=low=close=rate
                        # volume defaults to sales volume or 1.0 if not available
                        vol = sales if sales is not None else 1.0
                        records.append({
                            "timestamp": timestamp,
                            "open": rate,
                            "high": rate,
                            "low": rate,
                            "close": rate,
                            "volume": vol,
                            "timeframe": "1D",
                            "ticker_name": ticker
                        })

        return records
