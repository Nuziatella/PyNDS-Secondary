"""Basic PyNDS Gym Environment Usage Example.

This example demonstrates the basic usage of PyNDS with the Gym interface,
showing how to create an environment, take random actions, and collect
observations and rewards.

Perfect for getting started with PyNDS in reinforcement learning!
"""

from pynds.gym_env import PyNDSGymEnv


def basic_gym_example():
    """Basic Gym environment usage example."""
    print("PyNDS Basic Gym Example")
    print("=" * 40)
    
    # Create environment
    env = PyNDSGymEnv(
        rom_path="game.nds",  # Replace with your ROM path
        action_type="discrete",
        observation_type="rgb",
        frame_skip=1,
        max_episode_steps=1000,
        render_mode="human"  # Set to None for headless mode
    )
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action meanings: {env.get_action_meanings()}")
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run episode
    total_reward = 0
    step = 0
    
    while step < 100:  # Run for 100 steps
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        print(f"Step {step}: Action={action} ({env.get_action_meanings()[action]}), "
              f"Reward={reward:.3f}, Total={total_reward:.3f}")
        
        # Check if episode ended
        if terminated or truncated:
            print(f"Episode ended: {'Terminated' if terminated else 'Truncated'}")
            break
    
    print("\nEpisode completed!")
    print(f"Total steps: {step}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final episode info: {env.get_episode_info()}")
    
    # Close environment
    env.close()
    print("Environment closed!")


def observation_analysis():
    """Analyze different observation types."""
    print("\n🔍 Observation Analysis")
    print("=" * 40)
    
    # Test different observation types
    observation_types = ["rgb", "grayscale", "raw"]
    
    for obs_type in observation_types:
        print(f"\nTesting observation type: {obs_type}")
        
        env = PyNDSGymEnv(
            rom_path="game.nds",
            observation_type=obs_type,
            render_mode=None  # Headless mode
        )
        
        obs, _ = env.reset()
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Min: {obs.min()}, Max: {obs.max()}")
        
        env.close()


def action_space_exploration():
    """Explore different action spaces."""
    print("\n🎯 Action Space Exploration")
    print("=" * 40)
    
    action_types = ["discrete", "multi_discrete", "continuous"]
    
    for action_type in action_types:
        print(f"\nTesting action type: {action_type}")
        
        env = PyNDSGymEnv(
            rom_path="game.nds",
            action_type=action_type,
            render_mode=None
        )
        
        print(f"  Action space: {env.action_space}")
        print(f"  Sample action: {env.action_space.sample()}")
        
        if hasattr(env, 'get_action_meanings'):
            print(f"  Action meanings: {env.get_action_meanings()}")
        
        env.close()


if __name__ == "__main__":
    # Run examples
    try:
        basic_gym_example()
        observation_analysis()
        action_space_exploration()
    except FileNotFoundError:
        print("ROM file not found! Please update the rom_path in the examples.")
        print("   Make sure you have a valid .nds or .gba file.")
    except Exception as e:
        print(f"Error: {e}")
        print("   Make sure PyNDS is properly installed and configured.")
