"""Custom Reward Functions for PyNDS Gym Environment.

This example demonstrates how to create custom reward functions for specific
games and training objectives. Perfect for fine-tuning your RL agents!

Examples include:
- Memory-based rewards (tracking specific game states)
- Progress-based rewards (measuring advancement)
- Survival rewards (staying alive)
- Exploration rewards (encouraging discovery)
"""

import numpy as np
from typing import Dict, Any
from pynds.gym_env import PyNDSGymEnv


class CustomPyNDSGymEnv(PyNDSGymEnv):
    """PyNDS Gym Environment with custom reward functions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Custom reward tracking
        self.reward_history = []
        self.last_memory_values = {}
        self.progress_tracker = 0
        self.survival_steps = 0
        self.exploration_map = set()
        
        # Game-specific parameters (customize these!)
        self.target_memory_addresses = {
            'score': 0x02000000,  # Example: score address
            'lives': 0x02000004,  # Example: lives address
            'level': 0x02000008,  # Example: level address
        }
    
    def _custom_reward(self, obs: np.ndarray) -> float:
        """Enhanced custom reward function."""
        total_reward = 0.0
        
        # 1. Memory-based reward
        memory_reward = self._memory_based_reward()
        total_reward += memory_reward
        
        # 2. Progress-based reward
        progress_reward = self._progress_based_reward()
        total_reward += progress_reward
        
        # 3. Survival reward
        survival_reward = self._survival_reward()
        total_reward += survival_reward
        
        # 4. Exploration reward
        exploration_reward = self._exploration_reward(obs)
        total_reward += exploration_reward
        
        # 5. Frame change reward
        frame_reward = self._frame_change_reward(obs)
        total_reward += frame_reward
        
        # Store reward components for analysis
        self.reward_history.append({
            'total': total_reward,
            'memory': memory_reward,
            'progress': progress_reward,
            'survival': survival_reward,
            'exploration': exploration_reward,
            'frame': frame_reward
        })
        
        return total_reward
    
    def _memory_based_reward(self) -> float:
        """Reward based on specific memory values."""
        reward = 0.0
        
        try:
            for name, address in self.target_memory_addresses.items():
                # Read memory value (this is game-specific!)
                if name == 'score':
                    # Example: reward for increasing score
                    current_score = self.pynds.memory.read_ram_u32(address)
                    if name in self.last_memory_values:
                        score_diff = current_score - self.last_memory_values[name]
                        if score_diff > 0:
                            reward += score_diff * 0.01  # Scale reward
                    self.last_memory_values[name] = current_score
                
                elif name == 'lives':
                    # Example: penalty for losing lives
                    current_lives = self.pynds.memory.read_ram_u8(address)
                    if name in self.last_memory_values:
                        lives_diff = current_lives - self.last_memory_values[name]
                        if lives_diff < 0:
                            reward += lives_diff * 10.0  # Penalty for losing lives
                    self.last_memory_values[name] = current_lives
                
                elif name == 'level':
                    # Example: reward for level progression
                    current_level = self.pynds.memory.read_ram_u8(address)
                    if name in self.last_memory_values:
                        level_diff = current_level - self.last_memory_values[name]
                        if level_diff > 0:
                            reward += level_diff * 100.0  # Big reward for level up!
                    self.last_memory_values[name] = current_level
                    
        except Exception:
            # If memory reading fails, continue without memory rewards
            pass
        
        return reward
    
    def _progress_based_reward(self) -> float:
        """Reward based on game progress."""
        reward = 0.0
        
        # Example: reward for advancing in the game
        # This could be based on level, area, or other progress indicators
        try:
            # Check if we've made progress (customize this logic!)
            current_progress = self._get_game_progress()
            if current_progress > self.progress_tracker:
                progress_diff = current_progress - self.progress_tracker
                reward += progress_diff * 50.0  # Reward for progress
                self.progress_tracker = current_progress
        except Exception:
            pass
        
        return reward
    
    def _survival_reward(self) -> float:
        """Reward for staying alive."""
        self.survival_steps += 1
        
        # Small positive reward for each step survived
        survival_reward = 0.1
        
        # Bonus for surviving longer
        if self.survival_steps % 100 == 0:
            survival_reward += 1.0
        
        return survival_reward
    
    def _exploration_reward(self, obs: np.ndarray) -> float:
        """Reward for exploring new areas."""
        reward = 0.0
        
        # Create a simple hash of the observation for exploration tracking
        obs_hash = hash(obs.tobytes())
        
        if obs_hash not in self.exploration_map:
            self.exploration_map.add(obs_hash)
            reward += 1.0  # Reward for seeing new states
        
        return reward
    
    def _frame_change_reward(self, obs: np.ndarray) -> float:
        """Reward based on frame changes (original logic)."""
        if self.last_frame is not None:
            frame_diff = np.mean(np.abs(obs.astype(float) - self.last_frame.astype(float)))
            reward = min(frame_diff / 100.0, 1.0)
        else:
            reward = 0.0
        
        self.last_frame = obs.copy()
        return reward
    
    def _get_game_progress(self) -> float:
        """Get current game progress (customize for your game!)."""
        # This is a placeholder - implement based on your specific game
        # Examples:
        # - Level number
        # - Area/zone progress
        # - Score milestones
        # - Items collected
        
        try:
            # Example: use level as progress indicator
            level = self.pynds.memory.read_ram_u8(0x02000008)
            return float(level)
        except Exception:
            return 0.0
    
    def get_reward_analysis(self) -> Dict[str, Any]:
        """Get analysis of reward components."""
        if not self.reward_history:
            return {}
        
        # Calculate statistics for each reward component
        analysis = {}
        for component in ['total', 'memory', 'progress', 'survival', 'exploration', 'frame']:
            values = [r[component] for r in self.reward_history]
            analysis[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'total': np.sum(values)
            }
        
        return analysis


def demo_custom_rewards():
    """Demonstrate custom reward functions."""
    print("🎯 Custom Reward Functions Demo")
    print("=" * 50)
    
    # Create environment with custom rewards
    env = CustomPyNDSGymEnv(
        rom_path="game.nds",  # Replace with your ROM
        action_type="discrete",
        observation_type="rgb",
        frame_skip=4,
        max_episode_steps=1000,
        render_mode=None
    )
    
    print("Running episode with custom rewards...")
    
    # Run episode
    obs, info = env.reset()
    total_reward = 0
    step = 0
    
    while step < 100:  # Run for 100 steps
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step += 1
        
        if step % 20 == 0:  # Print every 20 steps
            print(f"Step {step}: Total reward = {total_reward:.2f}")
        
        if terminated or truncated:
            break
    
    # Analyze rewards
    print(f"\nEpisode completed! Total reward: {total_reward:.2f}")
    print("\nReward Analysis:")
    print("-" * 30)
    
    analysis = env.get_reward_analysis()
    for component, stats in analysis.items():
        print(f"{component.capitalize():>12}: "
              f"Mean={stats['mean']:.3f}, "
              f"Total={stats['total']:.3f}")
    
    env.close()


def create_game_specific_reward(game_name: str):
    """Create game-specific reward function."""
    print(f"Creating reward function for {game_name}")
    print("=" * 50)
    
    if game_name.lower() == "pokemon":
        return create_pokemon_reward()
    elif game_name.lower() == "mario":
        return create_mario_reward()
    elif game_name.lower() == "zelda":
        return create_zelda_reward()
    else:
        print(f"Unknown game: {game_name}")
        return None


def create_pokemon_reward():
    """Pokemon-specific reward function."""
    def pokemon_reward(env, obs):
        reward = 0.0
        
        try:
            # Reward for gaining experience
            exp = env.pynds.memory.read_ram_u32(0x02000000)  # Example address
            if hasattr(env, 'last_exp'):
                exp_gain = exp - env.last_exp
                if exp_gain > 0:
                    reward += exp_gain * 0.1
            env.last_exp = exp
            
            # Reward for catching Pokemon
            pokemon_caught = env.pynds.memory.read_ram_u8(0x02000004)  # Example
            if hasattr(env, 'last_pokemon'):
                if pokemon_caught > env.last_pokemon:
                    reward += 50.0  # Big reward for catching!
                env.last_pokemon = pokemon_caught
            else:
                env.last_pokemon = pokemon_caught
                
        except Exception:
            pass
        
        return reward
    
    return pokemon_reward


def create_mario_reward():
    """Mario-specific reward function."""
    def mario_reward(env, obs):
        reward = 0.0
        
        try:
            # Reward for coins
            coins = env.pynds.memory.read_ram_u8(0x02000000)  # Example
            if hasattr(env, 'last_coins'):
                coin_gain = coins - env.last_coins
                if coin_gain > 0:
                    reward += coin_gain * 10.0
            env.last_coins = coins
            
            # Reward for score
            score = env.pynds.memory.read_ram_u32(0x02000004)  # Example
            if hasattr(env, 'last_score'):
                score_gain = score - env.last_score
                if score_gain > 0:
                    reward += score_gain * 0.01
            env.last_score = score
            
        except Exception:
            pass
        
        return reward
    
    return mario_reward


def create_zelda_reward():
    """Zelda-specific reward function."""
    def zelda_reward(env, obs):
        reward = 0.0
        
        try:
            # Reward for rupees
            rupees = env.pynds.memory.read_ram_u16(0x02000000)  # Example
            if hasattr(env, 'last_rupees'):
                rupee_gain = rupees - env.last_rupees
                if rupee_gain > 0:
                    reward += rupee_gain * 5.0
            env.last_rupees = rupees
            
            # Reward for health
            health = env.pynds.memory.read_ram_u8(0x02000004)  # Example
            if hasattr(env, 'last_health'):
                health_diff = health - env.last_health
                if health_diff < 0:
                    reward += health_diff * 20.0  # Penalty for losing health
            env.last_health = health
            
        except Exception:
            pass
        
        return reward
    
    return zelda_reward


if __name__ == "__main__":
    print("PyNDS Custom Reward Functions")
    print("=" * 60)
    
    try:
        # Demo custom rewards
        demo_custom_rewards()
        
        # Show game-specific examples
        print("\n" + "="*60)
        print("Game-specific reward functions available:")
        print("- Pokemon: Experience gain, Pokemon catching")
        print("- Mario: Coin collection, score increase")
        print("- Zelda: Rupee collection, health management")
        
    except FileNotFoundError:
        print("ROM file not found! Please update the rom_path.")
    except Exception as e:
        print(f"Error: {e}")
        print("   Make sure PyNDS is properly installed and configured.")
