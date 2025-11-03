import numpy as np
import logging
from typing import List, Dict
import torch as th
import json
from stable_baselines3.common.buffers import RolloutBuffer
from datetime import datetime

logger = logging.getLogger(__name__)

class ImitationTrainer:
    """Train PPO agent using imitation learning from HPA observations"""
    
    def __init__(self, ppo_agent, env):
        self.ppo_agent = ppo_agent
        self.env = env
    
    def load_hpa_data(self, filepath: str) -> List[Dict]:
        """Load HPA observation data"""
        with open(filepath, 'r') as f:
            observations = json.load(f)
        logger.info(f"Loaded {len(observations)} HPA observations from {filepath}")
        return observations
    
    def prepare_training_data(self, observations: List[Dict]):
        """Convert HPA observations to training format"""
        states = []
        actions = []
        rewards = []
        
        for obs in observations:
            state = obs['current_state']
            action_delta = obs['action_delta']
            reward = obs['reward']
            
            # Convert action_delta to discrete action
            # Map: {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
            action_map = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
            # Clip to valid range
            action_delta_clipped = np.clip(action_delta, -2, 2)
            action = action_map.get(action_delta_clipped, 2)  # Default to no-op
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
        
        logger.info(f"Prepared {len(states)} training samples")
        logger.info(f"Action distribution: {np.bincount(actions)}")
        logger.info(f"Mean reward: {np.mean(rewards):.2f}, Std: {np.std(rewards):.2f}")
        
        return states, actions, rewards
    
    def behavioral_cloning(self, states: List[Dict], actions: List[int], 
                          epochs: int = 50, batch_size: int = 32):
        """Perform behavioral cloning from HPA expert"""
        logger.info(f"Starting behavioral cloning for {epochs} epochs")
        
        policy = self.ppo_agent.model.policy
        optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
        criterion = th.nn.CrossEntropyLoss()
        
        # Convert to tensors
        n_samples = len(states)
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch
                batch_states = [states[i] for i in batch_indices]
                batch_actions = th.tensor([actions[i] for i in batch_indices], dtype=th.long)
                
                # Convert states to observations
                batch_obs = []
                for state in batch_states:
                    self.env.set_state(state)
                    obs = self.env._state_dict_to_obs(state)
                    batch_obs.append(obs)
                
                # Stack observations
                obs_tensor = {}
                for key in batch_obs[0].keys():
                    obs_tensor[key] = th.tensor(
                        np.array([obs[key] for obs in batch_obs]), 
                        dtype=th.float32
                    )
                
                # Forward pass
                optimizer.zero_grad()
                dist = policy.get_distribution(obs_tensor)
                action_logits = dist.distribution.logits
                
                # Calculate loss
                loss = criterion(action_logits, batch_actions)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Behavioral cloning complete")
    
    def finetune_with_rl(self, timesteps: int = 10000):
        """Fine-tune the pre-trained agent with RL"""
        logger.info(f"Starting RL fine-tuning for {timesteps} timesteps")
        self.ppo_agent.model.learn(total_timesteps=timesteps, reset_num_timesteps=False)
        logger.info("RL fine-tuning complete")
    
    def train_from_hpa_data(self, filepath: str, bc_epochs: int = 50, 
                           rl_timesteps: int = 10000):
        """Complete training pipeline from HPA data"""
        # Load data
        observations = self.load_hpa_data(filepath)
        
        # Prepare training data
        states, actions, rewards = self.prepare_training_data(observations)
        
        # Behavioral cloning
        self.behavioral_cloning(states, actions, epochs=bc_epochs)
        
        # Save after BC
        bc_model_path = f"{self.ppo_agent.model_path}_after_bc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.ppo_agent.model.save(bc_model_path)
        logger.info(f"Saved model after behavioral cloning to {bc_model_path}")
        
        # RL fine-tuning (optional, in simulation)
        if rl_timesteps > 0:
            self.finetune_with_rl(rl_timesteps)
        
        # Save final model
        self.ppo_agent.model.save(self.ppo_agent.model_path)
        logger.info(f"Saved final model to {self.ppo_agent.model_path}")
        
        return {
            "samples_used": len(states),
            "bc_epochs": bc_epochs,
            "rl_timesteps": rl_timesteps,
            "bc_model_path": bc_model_path,
            "final_model_path": self.ppo_agent.model_path
        }
