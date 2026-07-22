import logging
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.config.settings import MODEL_CONFIG

logger = logging.getLogger(__name__)

class DRLEnsembleStrategy:
    def __init__(self, env_train_class, env_kwargs, train_data, val_data):
        self.env_train_class = env_train_class
        self.env_kwargs = env_kwargs
        self.train_data = train_data
        self.val_data = val_data
        self.config = MODEL_CONFIG.get("algorithms", {})
        
    def _make_env(self, data, is_eval: bool = False):
        kwargs = self.env_kwargs.copy()
        kwargs['df'] = data
        if is_eval:
            kwargs['random_start'] = False
            kwargs['episode_length'] = None
        else:
            if 'random_start' not in kwargs:
                kwargs['random_start'] = MODEL_CONFIG.get("random_start", True)
            if 'episode_length' not in kwargs:
                kwargs['episode_length'] = MODEL_CONFIG.get("episode_length", 60)
        env = self.env_train_class(**kwargs)
        return DummyVecEnv([lambda: env])

    def train_a2c(self):
        env_train = self._make_env(self.train_data, is_eval=False)
        logger.info("Training A2C...")
        model = A2C('MlpPolicy', env_train, 
                    learning_rate=self.config.get("a2c", {}).get("learning_rate", 0.0007),
                    n_steps=self.config.get("a2c", {}).get("n_steps", 5),
                    ent_coef=self.config.get("a2c", {}).get("ent_coef", 0.01),
                    verbose=0)
        model.learn(total_timesteps=10000) # Arbitrary timesteps for POC
        return model

    def train_ppo(self):
        env_train = self._make_env(self.train_data, is_eval=False)
        logger.info("Training PPO...")
        model = PPO('MlpPolicy', env_train, 
                    learning_rate=self.config.get("ppo", {}).get("learning_rate", 0.00025),
                    n_steps=self.config.get("ppo", {}).get("n_steps", 2048),
                    batch_size=self.config.get("ppo", {}).get("batch_size", 64),
                    ent_coef=self.config.get("ppo", {}).get("ent_coef", 0.01),
                    verbose=0)
        model.learn(total_timesteps=10000)
        return model

    def train_ddpg(self):
        env_train = self._make_env(self.train_data, is_eval=False)
        logger.info("Training DDPG...")
        model = DDPG('MlpPolicy', env_train, 
                     learning_rate=self.config.get("ddpg", {}).get("learning_rate", 0.001),
                     batch_size=self.config.get("ddpg", {}).get("batch_size", 128),
                     buffer_size=self.config.get("ddpg", {}).get("buffer_size", 50000),
                     verbose=0)
        model.learn(total_timesteps=10000)
        return model

    def evaluate_and_get_trajectory(self, model, data):
        """Runs evaluation and returns DataFrames for account values and action history.

        Returns:
            df_account: DataFrame with 'date', 'account_value', 'daily_return'
            df_actions: DataFrame with 'date' and action value per ticker
            df_shares: DataFrame with 'date' and share count per ticker
        """
        env = self._make_env(data, is_eval=True)

        obs = env.reset()
        done = np.array([False])
        
        while not done[0]:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)
            
        unwrapped_env = env.envs[0]
        asset_memory = getattr(unwrapped_env, 'previous_asset_memory', unwrapped_env.asset_memory)
        action_memory = getattr(unwrapped_env, 'previous_action_memory', unwrapped_env.action_memory)
        date_memory = getattr(unwrapped_env, 'previous_date_memory', unwrapped_env.date_memory)
        share_memory = getattr(unwrapped_env, 'previous_share_memory', unwrapped_env.share_memory)
        tickers = unwrapped_env.tickers

        # Match lengths between dates and asset memory
        min_len_asset = min(len(date_memory), len(asset_memory))
        df_account = pd.DataFrame({
            'date': date_memory[:min_len_asset],
            'account_value': asset_memory[:min_len_asset]
        })
        df_account['daily_return'] = df_account['account_value'].pct_change(1).fillna(0)

        # Build actions DataFrame
        min_len_action = min(len(date_memory) - 1, len(action_memory))
        if min_len_action > 0 and len(action_memory) > 0:
            act_matrix = np.array(action_memory[:min_len_action])
            if act_matrix.shape[1] == len(tickers) + 1:
                act_cols = ['CASH'] + [f"{t}" for t in tickers]
            else:
                act_cols = [f"{t}" for t in tickers]
            df_actions = pd.DataFrame(act_matrix, columns=act_cols)
            df_actions.insert(0, 'date', date_memory[1:min_len_action+1])
        else:
            df_actions = pd.DataFrame({'date': date_memory})

        # Build shares DataFrame
        min_len_share = min(len(date_memory), len(share_memory))
        if min_len_share > 0 and len(share_memory) > 0:
            share_matrix = np.array(share_memory[:min_len_share])
            share_cols = [f"{t}" for t in tickers]
            df_shares = pd.DataFrame(share_matrix, columns=share_cols)
            df_shares.insert(0, 'date', date_memory[:min_len_share])
        else:
            df_shares = pd.DataFrame({'date': date_memory})

        return df_account, df_actions, df_shares

    def _evaluate_model(self, model, data):
        """Returns Sharpe Ratio on given data"""
        df_account, _, _ = self.evaluate_and_get_trajectory(model, data)
        returns = df_account['daily_return']
        if returns.std() != 0:
            sharpe = (252 ** 0.5) * returns.mean() / returns.std()
        else:
            sharpe = 0.0
        return sharpe

    def train_and_select(self):
        """
        Trains all 3 models and selects the best one based on validation Sharpe Ratio.
        """
        models = {
            "A2C": self.train_a2c(),
            "PPO": self.train_ppo(),
            "DDPG": self.train_ddpg()
        }
        
        best_sharpe = -float('inf')
        best_model_name = None
        best_model = None
        
        logger.info("Evaluating models on Validation Set...")
        for name, model in models.items():
            sharpe = self._evaluate_model(model, self.val_data)
            logger.info(f"{name} Validation Sharpe Ratio: {sharpe:.4f}")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_model_name = name
                best_model = model
                
        logger.info(f"Ensemble selected: {best_model_name} with Sharpe {best_sharpe:.4f}")
        return best_model_name, best_model
