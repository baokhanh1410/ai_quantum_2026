"""SHAP Explainer module for DRL Portfolio Models (Explainable AI / XAI).

Computes SHAP (SHapley Additive exPlanations) values for DRL Agents (PPO, A2C, DDPG),
enabling financial attribution of model decisions to technical indicators & macro features.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP library is not installed. Run 'pip install shap' to enable XAI feature explanations.")


class SHAPExplainer:
    """Explains DRL Agent trading actions using SHAP (SHapley Additive exPlanations)."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        target_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        n_background: int = 50
    ):
        """
        Args:
            model: Trained Stable-Baselines3 model (PPO, A2C, DDPG) or a policy object.
            feature_names: List of state feature names (e.g. ['RSI', 'ADX', 'ATR', 'YIELD_CURVE_SLOPE', ...]).
            target_names: List of action output names (e.g. ['CASH', 'VN30', 'SJC_SELL']).
            background_data: Background dataset matrix (N, state_dim) for KernelExplainer baseline.
            n_background: Number of samples to summarize background data for speed.
        """
        if not HAS_SHAP:
            raise RuntimeError("SHAP package is missing. Please install it using 'pip install shap'.")

        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names or [f"Action_{i}" for i in range(1, 10)]
        self.explainer = None
        self.background_data = background_data
        self.n_background = n_background

    def _predict_fn(self, obs_matrix: np.ndarray) -> np.ndarray:
        """Wrapper prediction function mapping observation matrix -> model action outputs."""
        actions = []
        for obs in obs_matrix:
            # Handle float32 conversion & batch dimension
            obs_tensor = np.array(obs, dtype=np.float32)
            act, _ = self.model.predict(obs_tensor, deterministic=True)
            actions.append(np.atleast_1d(act))
        return np.array(actions)

    def fit_explainer(self, X_sample: np.ndarray) -> Any:
        """Initializes KernelExplainer with background sample data."""
        if X_sample is None or len(X_sample) == 0:
            raise ValueError("X_sample cannot be empty.")

        # Subsample background data to speed up KernelExplainer calculation
        if len(X_sample) > self.n_background:
            bg_data = shap.sample(X_sample, self.n_background)
        else:
            bg_data = X_sample

        self.background_data = bg_data
        logger.info(f"Initializing SHAP KernelExplainer with background shape: {bg_data.shape}")
        self.explainer = shap.KernelExplainer(self._predict_fn, bg_data)
        return self.explainer

    def compute_shap_values(
        self,
        X_eval: np.ndarray,
        nsamples: int = 100
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """Computes SHAP values for evaluation state observations.

        Args:
            X_eval: Matrix of observation states (N, state_dim).
            nsamples: Number of Monte-Carlo perturbations per calculation (higher = more precise, lower = faster).

        Returns:
            SHAP values array or list of arrays (one per action dimension).
        """
        if self.explainer is None:
            self.fit_explainer(X_eval)

        logger.info(f"Computing SHAP values for {len(X_eval)} observations (nsamples={nsamples})...")
        shap_vals = self.explainer.shap_values(X_eval, nsamples=nsamples)
        return shap_vals

    def get_feature_importance_df(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        action_idx: int = 0
    ) -> pd.DataFrame:
        """Calculates mean absolute SHAP feature importance for a specified action output.

        Args:
            shap_values: Output from compute_shap_values().
            action_idx: Index of the action to analyze (e.g. 0 for asset 1 / cash).

        Returns:
            DataFrame sorted by importance with columns ['Feature', 'Mean_Abs_SHAP', 'Impact_Pct'].
        """
        if isinstance(shap_values, list):
            sv = shap_values[action_idx]
        else:
            sv = shap_values

        if sv.ndim == 3:
            # Shape (N, n_features, n_actions)
            sv = sv[:, :, action_idx]

        mean_abs_shap = np.mean(np.abs(sv), axis=0)

        # Truncate or pad feature names if state_dim doesn't match feature_names length exactly
        n_feats = len(mean_abs_shap)
        if len(self.feature_names) >= n_feats:
            names = self.feature_names[:n_feats]
        else:
            names = self.feature_names + [f"Feature_{i}" for i in range(len(self.feature_names), n_feats)]

        df_imp = pd.DataFrame({
            "Feature": names,
            "Mean_Abs_SHAP": mean_abs_shap
        })

        total_imp = df_imp["Mean_Abs_SHAP"].sum()
        df_imp["Impact_Pct"] = (df_imp["Mean_Abs_SHAP"] / (total_imp + 1e-8)) * 100.0
        df_imp = df_imp.sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        return df_imp

    def plot_summary_bar(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        X_eval: np.ndarray,
        action_idx: int = 0,
        max_display: int = 10,
        title: str = "SHAP Feature Importance (DRL Policy Decision)",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plots a bar chart of mean absolute SHAP values for top features.

        Returns:
            Matplotlib Figure object.
        """
        df_imp = self.get_feature_importance_df(shap_values, action_idx=action_idx)
        top_df = df_imp.head(max_display).iloc[::-1]  # Reverse for horizontal bar plot

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        bars = ax.barh(top_df["Feature"], top_df["Mean_Abs_SHAP"], color="#1f77b4", alpha=0.85)

        max_w = max(top_df["Mean_Abs_SHAP"]) if not top_df.empty else 0.01
        ax.set_xlim(0, max_w * 1.25 if max_w > 0 else 0.01)

        for bar, pct in zip(bars, top_df["Impact_Pct"]):
            width = bar.get_width()
            ax.text(width + max_w * 0.02, bar.get_y() + bar.get_height()/2, f"{pct:.1f}%",
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#333333')

        ax.set_xlabel("Mean |SHAP value| (Average Impact on Action Output)", fontsize=10)
        
        target_name = self.target_names[action_idx] if action_idx < len(self.target_names) else f"Action {action_idx}"
        ax.set_title(f"{title} - [{target_name}]", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved SHAP summary plot to {save_path}")

        return fig

    def plot_trade_waterfall(
        self,
        shap_values: Union[np.ndarray, List[np.ndarray]],
        X_eval: np.ndarray,
        sample_idx: int = 0,
        action_idx: int = 0,
        date_str: str = "N/A",
        max_display: int = 8,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plots a waterfall chart explaining a single trade decision at sample_idx date.

        Returns:
            Matplotlib Figure object.
        """
        if isinstance(shap_values, list):
            sv_single = shap_values[action_idx][sample_idx]
        else:
            if shap_values.ndim == 3:
                sv_single = shap_values[sample_idx, :, action_idx]
            else:
                sv_single = shap_values[sample_idx]

        x_single = X_eval[sample_idx]
        n_feats = len(sv_single)
        names = (self.feature_names[:n_feats] if len(self.feature_names) >= n_feats
                 else self.feature_names + [f"Feat_{i}" for i in range(len(self.feature_names), n_feats)])

        df_single = pd.DataFrame({
            "Feature": names,
            "SHAP_Value": sv_single,
            "Feature_Value": x_single[:n_feats],
            "Abs_SHAP": np.abs(sv_single)
        }).sort_values(by="Abs_SHAP", ascending=False).head(max_display)

        df_single = df_single.iloc[::-1]

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
        colors = ["#d62728" if val < 0 else "#2ca02c" for val in df_single["SHAP_Value"]]

        bars = ax.barh(df_single["Feature"], df_single["SHAP_Value"], color=colors, alpha=0.85)

        min_v = min(df_single["SHAP_Value"]) if not df_single.empty else -0.01
        max_v = max(df_single["SHAP_Value"]) if not df_single.empty else 0.01
        span = max(abs(min_v), abs(max_v))
        if span < 1e-6:
            span = 0.01

        # Symmetric / balanced margin padding
        padding = span * 0.35
        ax.set_xlim(min(min_v - padding, -padding), max(max_v + padding, padding))

        for bar, val, feat_v in zip(bars, df_single["SHAP_Value"], df_single["Feature_Value"]):
            if val < 0:
                tx = val + span * 0.015
                ha = "left"
                txt_color = "white" if abs(val) > span * 0.3 else "#222222"
            else:
                tx = val + span * 0.015
                ha = "left"
                txt_color = "#222222"

            ax.text(tx, bar.get_y() + bar.get_height()/2, f"{val:+.3f} (Val: {feat_v:.2f})",
                    va='center', ha=ha, fontsize=8, fontweight='bold', color=txt_color)

        ax.axvline(0, color="#888888", linestyle="-", linewidth=0.8)
        ax.set_xlabel("SHAP Value (Contribution to Action Decision)", fontsize=10)

        target_name = self.target_names[action_idx] if action_idx < len(self.target_names) else f"Action {action_idx}"
        ax.set_title(f"Trade Explanation Waterfall ({date_str}) - Target: [{target_name}]", fontsize=12, fontweight="bold", pad=12)
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            logger.info(f"Saved SHAP waterfall plot to {save_path}")

        return fig
