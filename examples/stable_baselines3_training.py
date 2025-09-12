"""Stable-Baselines3 Training Example with PyNDS.

This example demonstrates how to train RL agents using Stable-Baselines3
with PyNDS environments. It includes training, evaluation, and model saving.

Perfect for training AI agents to play Nintendo DS games!
"""

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from pynds.gym_env import PyNDSGymEnv


def create_training_env(rom_path: str, **kwargs):
    """Create a training environment with monitoring."""
    env = PyNDSGymEnv(rom_path, **kwargs)
    env = Monitor(env)  # Add monitoring
    return env


def train_ppo_agent(rom_path: str, total_timesteps: int = 100000):
    """Train a PPO agent on a PyNDS environment."""
    print("🤖 Training PPO Agent on PyNDS")
    print("=" * 50)

    # Create environment
    env = create_training_env(
        rom_path=rom_path,
        action_type="discrete",
        observation_type="rgb",
        frame_skip=4,  # Skip frames for faster training
        max_episode_steps=1000,
        render_mode=None,
    )

    # Create PPO model
    model = PPO(
        "CnnPolicy",  # CNN policy for image observations
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./tensorboard_logs/",
    )

    print(f"Training for {total_timesteps} timesteps...")
    print(f"Environment: {env.observation_space} -> {env.action_space}")

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model_path = "pynds_ppo_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    return model


def train_dqn_agent(rom_path: str, total_timesteps: int = 100000):
    """Train a DQN agent on a PyNDS environment."""
    print("🤖 Training DQN Agent on PyNDS")
    print("=" * 50)

    # Create environment
    env = create_training_env(
        rom_path=rom_path,
        action_type="discrete",
        observation_type="grayscale",  # DQN works well with grayscale
        frame_skip=4,
        max_episode_steps=1000,
        render_mode=None,
    )

    # Create DQN model
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=1e-4,
        buffer_size=50000,
        learning_starts=10000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        tensorboard_log="./tensorboard_logs/",
    )

    print(f"Training for {total_timesteps} timesteps...")
    print(f"Environment: {env.observation_space} -> {env.action_space}")

    # Train the model
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model_path = "pynds_dqn_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")

    env.close()
    return model


def evaluate_agent(model, rom_path: str, n_episodes: int = 5):
    """Evaluate a trained agent."""
    print(f"🎯 Evaluating Agent for {n_episodes} episodes")
    print("=" * 50)

    # Create evaluation environment
    env = PyNDSGymEnv(
        rom_path=rom_path,
        action_type="discrete",
        observation_type="rgb",
        frame_skip=4,
        max_episode_steps=1000,
        render_mode="human",  # Show evaluation
    )

    episode_rewards = []
    episode_lengths = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(
            f"Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}"
        )

    env.close()

    # Print statistics
    print("\nEvaluation Results:")
    print(
        f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(
        f"  Mean length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}"
    )
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")

    return episode_rewards, episode_lengths


def train_with_evaluation(rom_path: str, total_timesteps: int = 50000):
    """Train with periodic evaluation."""
    print("🔄 Training with Evaluation")
    print("=" * 50)

    # Create training environment
    train_env = create_training_env(
        rom_path=rom_path,
        action_type="discrete",
        observation_type="rgb",
        frame_skip=4,
        max_episode_steps=1000,
        render_mode=None,
    )

    # Create evaluation environment
    eval_env = create_training_env(
        rom_path=rom_path,
        action_type="discrete",
        observation_type="rgb",
        frame_skip=4,
        max_episode_steps=1000,
        render_mode=None,
    )

    # Create PPO model
    model = PPO(
        "CnnPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        tensorboard_log="./tensorboard_logs/",
    )

    # Setup evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    # Train with evaluation
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Save final model
    model.save("pynds_ppo_final")
    print("Final model saved to pynds_ppo_final")

    train_env.close()
    eval_env.close()

    return model


def main():
    """Run the main training script."""
    rom_path = "game.nds"  # Replace with your ROM path

    print("PyNDS Stable-Baselines3 Training")
    print("=" * 60)

    try:
        # Train PPO agent
        ppo_model = train_ppo_agent(rom_path, total_timesteps=50000)

        # Evaluate PPO agent
        print("\n" + "=" * 60)
        evaluate_agent(ppo_model, rom_path, n_episodes=3)

        # Train DQN agent
        print("\n" + "=" * 60)
        dqn_model = train_dqn_agent(rom_path, total_timesteps=50000)

        # Evaluate DQN agent
        print("\n" + "=" * 60)
        evaluate_agent(dqn_model, rom_path, n_episodes=3)

        print("\nTraining completed successfully!")
        print("Check the saved models and tensorboard logs for results.")

    except FileNotFoundError:
        print("ROM file not found! Please update the rom_path variable.")
        print("   Make sure you have a valid .nds or .gba file.")
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("   Install with: pip install stable-baselines3[extra]")
    except Exception as e:
        print(f"Error: {e}")
        print("   Make sure PyNDS is properly installed and configured.")


if __name__ == "__main__":
    main()
