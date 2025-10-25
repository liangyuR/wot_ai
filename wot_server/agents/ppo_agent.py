"""
PPO Agent for World of Tanks
"""

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from gymnasium import spaces
import numpy as np
from typing import Dict, Any
import yaml


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for World of Tanks
    
    Processes screen images and additional game state
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions
        n_input_channels = observation_space["screen"].shape[-1]
        
        # CNN for screen processing (similar to Nature DQN)
        self.cnn_ = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute CNN output size
        with torch.no_grad():
            sample_screen = torch.zeros(1, n_input_channels, 84, 84)
            cnn_output_size = self.cnn_(sample_screen).shape[1]
        
        # Minimap CNN
        self.minimap_cnn_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Compute minimap CNN output size
        with torch.no_grad():
            sample_minimap = torch.zeros(1, 3, 64, 64)
            minimap_output_size = self.minimap_cnn_(sample_minimap).shape[1]
        
        # Additional state processing (health, ammo)
        state_dim = 2  # health + ammo
        
        # Combine all features
        combined_size = cnn_output_size + minimap_output_size + state_dim
        
        self.linear_ = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            observations: Dictionary with 'screen', 'health', 'ammo', 'minimap'
            
        Returns:
            Feature tensor
        """
        # Process screen (B, H, W, C) -> (B, C, H, W)
        screen = observations["screen"].permute(0, 3, 1, 2).float() / 255.0
        cnn_features = self.cnn_(screen)
        
        # Process minimap
        minimap = observations["minimap"].permute(0, 3, 1, 2).float() / 255.0
        minimap_features = self.minimap_cnn_(minimap)
        
        # Concatenate additional state
        health = observations["health"].float()
        ammo = observations["ammo"].float()
        state = torch.cat([health, ammo], dim=1)
        
        # Combine all features
        combined = torch.cat([cnn_features, minimap_features, state], dim=1)
        
        return self.linear_(combined)


class WotPpoAgent:
    """
    PPO Agent for World of Tanks
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PPO agent
        
        Args:
            config: Configuration dictionary
        """
        self.config_ = config
        self.model_ = None
        
    def createModel(self, env):
        """
        Create PPO model
        
        Args:
            env: Gymnasium environment
        """
        ppo_config = self.config_["ppo"]
        network_config = self.config_["network"]
        training_config = self.config_["training"]
        
        policy_kwargs = dict(
            features_extractor_class=CustomCNN,
            features_extractor_kwargs=dict(
                features_dim=network_config["features_extractor"]["features_dim"]
            ),
            net_arch=dict(
                pi=network_config["policy_net"]["hidden_sizes"],
                vf=network_config["value_net"]["hidden_sizes"]
            )
        )
        
        self.model_ = PPO(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=ppo_config["learning_rate"],
            n_steps=ppo_config["n_steps"],
            batch_size=ppo_config["batch_size"],
            n_epochs=ppo_config["n_epochs"],
            gamma=ppo_config["gamma"],
            gae_lambda=ppo_config["gae_lambda"],
            clip_range=ppo_config["clip_range"],
            ent_coef=ppo_config["ent_coef"],
            vf_coef=ppo_config["vf_coef"],
            max_grad_norm=ppo_config["max_grad_norm"],
            policy_kwargs=policy_kwargs,
            tensorboard_log=training_config["tensorboard_log"],
            device=training_config["device"],
            verbose=self.config_["debug"]["verbose"]
        )
        
        return self.model_
        
    def train(self, env, total_timesteps: int = None):
        """
        Train the agent
        
        Args:
            env: Training environment
            total_timesteps: Total training timesteps
        """
        if self.model_ is None:
            self.createModel(env)
            
        if total_timesteps is None:
            total_timesteps = self.config_["training"]["total_timesteps"]
        
        training_config = self.config_["training"]
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=training_config["save_freq"],
            save_path=training_config["checkpoint_dir"],
            name_prefix="wot_ppo"
        )
        
        eval_callback = EvalCallback(
            env,
            best_model_save_path=training_config["best_model_dir"],
            log_path=training_config["tensorboard_log"],
            eval_freq=training_config["eval_freq"],
            n_eval_episodes=training_config["eval_episodes"],
            deterministic=True
        )
        
        callback = CallbackList([checkpoint_callback, eval_callback])
        
        # Train
        self.model_.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=training_config["log_interval"]
        )
        
    def predict(self, observation: Dict[str, np.ndarray], deterministic: bool = True):
        """
        Predict action given observation
        
        Args:
            observation: Environment observation
            deterministic: Use deterministic policy
            
        Returns:
            action, state
        """
        return self.model_.predict(observation, deterministic=deterministic)
        
    def save(self, path: str):
        """Save model"""
        self.model_.save(path)
        
    def load(self, path: str):
        """Load model"""
        self.model_ = PPO.load(path)


def LoadConfig(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

