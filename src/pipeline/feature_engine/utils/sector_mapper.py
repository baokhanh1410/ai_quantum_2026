"""Utility for mapping individual stock symbols to their corresponding Sector Indices.

This uses the configuration defined in config/sector_mapping.yaml and
the vnstock library (if available) to look up the sector of an unknown stock.
"""

import os
import yaml
import logging
from typing import Dict, Optional

logger = logging.getLogger("feature_engine.utils.sector_mapper")


class SectorMapper:
    """Maps stock symbols to HOSE sector indices (e.g. TCB -> VNFIN)."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            # Default to config/sector_mapping.yaml at project root
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
            config_path = os.path.join(root_dir, "config", "sector_mapping.yaml")
        
        self.config_path = config_path
        self.mapping: Dict[str, str] = {}
        self.fallback: str = "VN100"
        self._load_config()

    def _load_config(self) -> None:
        """Loads the YAML mapping rules."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Sector mapping config not found at {self.config_path}. Using empty mapping.")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                self.mapping = data.get("mapping", {})
                self.fallback = data.get("default_fallback", "VN100")
            logger.info(f"Loaded sector mapping with {len(self.mapping)} rules.")
        except Exception as e:
            logger.error(f"Failed to load sector mapping config: {e}")

    def get_sector_index(self, stock_symbol: str) -> str:
        """Resolves the HOSE sector index (e.g., VNFIN) for a given stock symbol.
        
        Args:
            stock_symbol: The 3-letter stock ticker (e.g., 'TCB').
            
        Returns:
            The string representing the sector index (e.g., 'VNFIN').
        """
        # Hardcode a few common ones to avoid network calls during fast inference
        # This acts as an L1 Cache. You can expand this or rely entirely on vnstock.
        _L1_CACHE = {
            "TCB": "VNFIN", "SSI": "VNFIN", "VND": "VNFIN", "MBB": "VNFIN", "VCB": "VNFIN",
            "VHM": "VNREAL", "VIC": "VNREAL", "NVL": "VNREAL", "KDH": "VNREAL",
            "HPG": "VNMAT", "HSG": "VNMAT", "NKG": "VNMAT", "DGC": "VNMAT",
            "FPT": "VNIT", "CMG": "VNIT",
            "MWG": "VNCOND", "PNJ": "VNCOND", "VNM": "VNCONS", "MSN": "VNCONS",
            "GAS": "VNENE", "PVD": "VNENE", "PVS": "VNENE",
            "POW": "VNUTI", "REE": "VNUTI",
            "VJC": "VNCOND", "HVN": "VNCOND",
            "VCG": "VNIND", "C4G": "VNIND", "HHV": "VNIND", "GMD": "VNIND"
        }
        
        stock_symbol = stock_symbol.upper().strip()
        if stock_symbol in _L1_CACHE:
            return _L1_CACHE[stock_symbol]

        # If not in cache, fallback to VNSTOCK API
        try:
            import vnstock
            import pandas as pd
            
            # Using try-except for vnstock as its API changes often
            # In vnstock legacy, company_overview sometimes crashes. 
            # We wrap it safely.
            try:
                df = vnstock.company_overview(stock_symbol)
                if isinstance(df, pd.DataFrame) and not df.empty and "industry" in df.columns:
                    icb_industry = df["industry"].iloc[0]
                    return self.mapping.get(icb_industry, self.fallback)
            except Exception as e:
                logger.debug(f"vnstock legacy API failed for {stock_symbol}: {e}")
                
            # If we are here, we couldn't resolve it via API.
            logger.warning(f"Could not resolve sector for {stock_symbol}. Defaulting to {self.fallback}.")
            return self.fallback

        except ImportError:
            logger.warning("vnstock is not installed. Returning fallback index.")
            return self.fallback
