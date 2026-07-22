from fastapi import APIRouter, BackgroundTasks, HTTPException
from typing import Dict, Any, Optional
import logging

from model_engine.data.data_service import DataQueryService
from model_engine.data.processor import DataProcessor
from model_engine.env.stock_trading_env import StockTradingEnv
from model_engine.models.drl_models import DRLEnsembleStrategy
from model_engine.analysis.metrics_analyzer import MetricsAnalyzer
from model_engine.analysis.visualization import ModelVisualizer
from core.config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model Engine"])


def run_ensemble_training():
    logger.info("Starting DRL Ensemble Training Pipeline...")
    try:                                                    
        # Load configs
        train_start = MODEL_CONFIG.get("train_start_date", "2018-01-01")
        train_end = MODEL_CONFIG.get("train_end_date", "2020-12-31")
        val_start = MODEL_CONFIG.get("val_start_date", "2021-01-01")
        val_end = MODEL_CONFIG.get("val_end_date", "2021-12-31")
        features = MODEL_CONFIG.get("features", [])
        
        # Query Data
        query_svc = DataQueryService()
        train_raw = query_svc.fetch_data(train_start, train_end)
        val_raw = query_svc.fetch_data(val_start, val_end)
        
        # Process Data
        train_data = DataProcessor(train_raw, features).clean_data()
        val_data = DataProcessor(val_raw, features).clean_data()
        
        if train_data.empty or val_data.empty:
            logger.error("Insufficient data for training or validation.")
            return
            
        # Env Config
        env_kwargs = {
            "features": features,
            "initial_balance": MODEL_CONFIG.get("initial_balance", 1_000_000_000),
            "transaction_cost_pct": MODEL_CONFIG.get("transaction_cost_pct", 0.001),
            "turbulence_threshold": MODEL_CONFIG.get("turbulence_threshold", 150)
        }
        
        # Train Ensemble
        strategy = DRLEnsembleStrategy(StockTradingEnv, env_kwargs, train_data, val_data)
        best_agent_name, best_agent = strategy.train_and_select()
        
        logger.info(f"Training completed. Best agent: {best_agent_name}")
        
    except Exception as e:
        logger.error(f"Error during training pipeline: {e}")


@router.post("/train")
def trigger_training(background_tasks: BackgroundTasks):
    """
    Triggers the DRL Ensemble training pipeline in the background.
    """
    background_tasks.add_task(run_ensemble_training)
    return {"message": "DRL Ensemble training started in the background."}


@router.post("/evaluate")
def trigger_evaluation(start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, Any]:
    """Runs evaluation of the DRL strategy and generates performance charts and metrics summary.

    Args:
        start_date: Optional evaluation start date (YYYY-MM-DD)
        end_date: Optional evaluation end date (YYYY-MM-DD)

    Returns:
        Summary JSON with status, financial metrics, and paths to saved PNG plots.
    """
    try:
        val_start = start_date or MODEL_CONFIG.get("val_start_date", "2021-01-01")
        val_end = end_date or MODEL_CONFIG.get("val_end_date", "2021-12-31")
        train_start = MODEL_CONFIG.get("train_start_date", "2018-01-01")
        train_end = MODEL_CONFIG.get("train_end_date", "2020-12-31")
        features = MODEL_CONFIG.get("features", [])

        query_svc = DataQueryService()
        train_raw = query_svc.fetch_data(train_start, train_end)
        val_raw = query_svc.fetch_data(val_start, val_end)

        train_data = DataProcessor(train_raw, features).clean_data()
        val_data = DataProcessor(val_raw, features).clean_data()

        if train_data.empty or val_data.empty:
            raise HTTPException(status_code=400, detail="Insufficient data for evaluation.")

        env_kwargs = {
            "features": features,
            "initial_balance": MODEL_CONFIG.get("initial_balance", 1_000_000_000),
            "transaction_cost_pct": MODEL_CONFIG.get("transaction_cost_pct", 0.001),
            "turbulence_threshold": MODEL_CONFIG.get("turbulence_threshold", 150)
        }

        strategy = DRLEnsembleStrategy(StockTradingEnv, env_kwargs, train_data, val_data)
        best_name, best_model = strategy.train_and_select()

        # Evaluate and get trajectory DataFrames
        df_account, df_actions, df_shares = strategy.evaluate_and_get_trajectory(best_model, val_data)

        # Visualizer & Analyzer
        visualizer = ModelVisualizer()
        dashboard_path, metrics = visualizer.plot_summary_dashboard(
            df_agent=df_account,
            df_actions=df_actions,
            benchmark_name="VN30",
            save_name="eval_summary_dashboard.png"
        )
        nav_plot_path = visualizer.plot_nav_vs_benchmark(df_account, save_name="eval_nav_curve.png")
        underwater_path = visualizer.plot_underwater(df_account, save_name="eval_underwater.png")
        action_dist_path = visualizer.plot_action_distribution(df_actions, save_name="eval_action_dist.png")

        return {
            "status": "success",
            "best_agent": best_name,
            "evaluation_period": {"start": val_start, "end": val_end},
            "metrics": metrics,
            "plots": {
                "summary_dashboard": dashboard_path,
                "nav_vs_benchmark": nav_plot_path,
                "underwater_plot": underwater_path,
                "action_distribution": action_dist_path,
            }
        }

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

