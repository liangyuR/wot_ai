"""
Training script for PPO agent
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from agents.ppo_agent import WotPpoAgent, LoadConfig
from env.wot_env import MakeWotEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from loguru import logger


def main():
    """Main training function"""
    
    parser = argparse.ArgumentParser(description="Train World of Tanks PPO Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ppo_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = LoadConfig(args.config)
    
    # Create directories
    os.makedirs(config["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["training"]["best_model_dir"], exist_ok=True)
    os.makedirs(config["training"]["tensorboard_log"], exist_ok=True)
    
    # Create environment
    logger.info("Creating environment...")
    env = MakeWotEnv(config)
    env = DummyVecEnv([lambda: env])
    
    # Create agent
    logger.info("Creating PPO agent...")
    agent = WotPpoAgent(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        agent.load(args.resume)
    
    # Determine total timesteps
    total_timesteps = args.timesteps or config["training"]["total_timesteps"]
    
    # Start training
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    logger.info("=" * 80)
    logger.info("IMPORTANT: Make sure World of Tanks is running!")
    logger.info("The game window should be visible and in focus.")
    logger.info("=" * 80)
    
    try:
        agent.train(env, total_timesteps=total_timesteps)
        
        # Save final model
        final_model_path = os.path.join(
            config["training"]["best_model_dir"],
            "final_model.zip"
        )
        agent.save(final_model_path)
        logger.info(f"Training completed! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user!")
        
        # Save current model
        interrupted_model_path = os.path.join(
            config["training"]["checkpoint_dir"],
            "interrupted_model.zip"
        )
        agent.save(interrupted_model_path)
        logger.info(f"Model saved to {interrupted_model_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        env.close()


if __name__ == "__main__":
    main()

