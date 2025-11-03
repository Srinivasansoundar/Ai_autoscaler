import asyncio
import logging
import json
from datetime import datetime
from typing import List, Dict, Any
import os

logger = logging.getLogger(__name__)

class HPAObserver:
    """Observes HPA behavior and collects real transitions"""
    
    def __init__(self, k8s_manager, data_dir="data/hpa_observations"):
        self.k8s_manager = k8s_manager
        self.data_dir = data_dir
        self.observations = []
        self.is_collecting = False
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
    
    async def collect_transition(self, state_provider_func) -> Dict[str, Any]:
        """Collect a single state transition"""
        try:
            # Get current state
            current_state = await state_provider_func()
            current_replicas = current_state['infra']['pods_ready']
            hpa_desired = current_state['infra']['hpa_desired_replicas']
            
            # Calculate HPA action
            hpa_action_delta = hpa_desired - current_replicas
            
            # Wait for next observation (30 seconds to allow stabilization)
            await asyncio.sleep(30)
            
            # Get next state after HPA acts
            next_state = await state_provider_func()
            
            # Calculate reward based on real outcomes
            reward = self._calculate_reward(current_state, next_state, hpa_action_delta)
            
            transition = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_state": current_state,
                "action_delta": hpa_action_delta,
                "next_state": next_state,
                "reward": reward,
                "current_replicas": current_replicas,
                "hpa_desired": hpa_desired
            }
            
            self.observations.append(transition)
            logger.info(f"Collected transition: replicas={current_replicas} -> {next_state['infra']['pods_ready']}, "
                       f"action={hpa_action_delta}, reward={reward:.2f}")
            
            return transition
            
        except Exception as e:
            logger.error(f"Failed to collect transition: {e}")
            return None
    
    def _calculate_reward(self, prev_state, curr_state, action_delta):
        """Calculate reward based on actual outcomes"""
        curr_workload = curr_state['workload']
        curr_infra = curr_state['infra']
        
        reward = 0.0
        
        # Primary: Latency SLA
        latency_ms = curr_workload['p95_latency_ms']
        if latency_ms < 800:
            reward += 5.0
        elif latency_ms < 1000:
            reward += 2.0
        elif latency_ms < 1500:
            reward -= 3.0
        else:
            reward -= 10.0
        
        # Secondary: Error rate
        if curr_workload['error_rate_pct'] < 1.0:
            reward += 1.0
        else:
            reward -= curr_workload['error_rate_pct'] * 2.0
        
        # Tertiary: Resource efficiency
        cpu = curr_infra['cpu_utilization_pct']
        if 60 <= cpu <= 75:
            reward += 2.0
        elif 50 <= cpu < 60 or 75 < cpu <= 80:
            reward += 0.5
        elif cpu < 40 and curr_infra['pods_ready'] > 1:
            reward -= 1.5
        elif cpu > 85:
            reward -= 2.0
        
        # Stability penalty
        if action_delta != 0:
            reward -= 0.3
        
        return float(reward)
    
    async def observe_hpa(self, state_provider_func, duration_minutes: int = 60):
        """Observe HPA behavior for specified duration"""
        logger.info(f"Starting HPA observation for {duration_minutes} minutes...")
        self.is_collecting = True
        self.observations = []
        
        num_samples = duration_minutes * 2  # Every 30 seconds
        
        for i in range(num_samples):
            if not self.is_collecting:
                break
                
            transition = await self.collect_transition(state_provider_func)
            
            if transition:
                logger.info(f"Progress: {i+1}/{num_samples} transitions collected")
        
        # Save observations
        self.save_observations()
        logger.info(f"HPA observation complete. Collected {len(self.observations)} transitions")
        
        return self.observations
    
    def save_observations(self, filename=None):
        """Save observations to file"""
        if not filename:
            filename = f"hpa_observations_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.observations, f, indent=2)
        
        logger.info(f"Saved {len(self.observations)} observations to {filepath}")
        return filepath
    
    def load_observations(self, filename):
        """Load observations from file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'r') as f:
            self.observations = json.load(f)
        
        logger.info(f"Loaded {len(self.observations)} observations from {filepath}")
        return self.observations
    
    def stop_collection(self):
        """Stop observation collection"""
        self.is_collecting = False
        logger.info("Stopping HPA observation")
