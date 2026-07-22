import pandas as pd
import numpy as np
import logging
from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)

class StationarityTester:
    def __init__(self, significance_level=0.05):
        self.alpha = significance_level

    def run_adf_test(self, series: pd.Series):
        """
        Augmented Dickey-Fuller test. 
        Null Hypothesis (H0): The series has a unit root (is non-stationary).
        Alternate Hypothesis (H1): The series is stationary.
        """
        # Drop NaN values for statistical tests
        series = series.dropna()
        if len(series) < 30:
            return {"stationary": False, "p_value": None, "error": "Not enough data"}
            
        try:
            result = adfuller(series, autolag='AIC')
            p_value = result[1]
            return {
                "test_statistic": result[0],
                "p_value": p_value,
                "critical_values": result[4],
                "stationary": p_value < self.alpha
            }
        except Exception as e:
            return {"stationary": False, "p_value": None, "error": str(e)}

    def run_kpss_test(self, series: pd.Series):
        """
        Kwiatkowski-Phillips-Schmidt-Shin test.
        Null Hypothesis (H0): The series is stationary around a deterministic trend.
        Alternate Hypothesis (H1): The series has a unit root (is non-stationary).
        """
        series = series.dropna()
        if len(series) < 30:
            return {"stationary": False, "p_value": None, "error": "Not enough data"}

        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # nlags="auto" sets lags based on data size
                result = kpss(series, regression='c', nlags="auto")
            p_value = result[1]
            return {
                "test_statistic": result[0],
                "p_value": p_value,
                "critical_values": result[3],
                # Notice for KPSS: if p_value < alpha, we reject H0 (it is non-stationary)
                "stationary": p_value >= self.alpha 
            }
        except Exception as e:
            return {"stationary": False, "p_value": None, "error": str(e)}

    def analyze_features(self, df: pd.DataFrame, features: list, groupby_col: str = 'tic') -> pd.DataFrame:
        """
        Run ADF and KPSS tests on all specified features for each group (ticker).
        """
        if df.empty or groupby_col not in df.columns:
            logger.warning(f"DataFrame is empty or missing groupby column '{groupby_col}'")
            return pd.DataFrame()
            
        results = []
        for name, group in df.groupby(groupby_col):
            for feature in features:
                if feature not in group.columns:
                    continue
                
                series = group[feature]
                adf_res = self.run_adf_test(series)
                kpss_res = self.run_kpss_test(series)
                
                results.append({
                    "ticker": name,
                    "feature": feature,
                    "adf_stationary": adf_res.get("stationary"),
                    "adf_p_value": adf_res.get("p_value"),
                    "kpss_stationary": kpss_res.get("stationary"),
                    "kpss_p_value": kpss_res.get("p_value"),
                    "is_strictly_stationary": adf_res.get("stationary") and kpss_res.get("stationary")
                })
                
        return pd.DataFrame(results)
