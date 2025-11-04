import json
import random
from datetime import datetime
import numpy as np

class SyntheticDatasetGenerator:
    """Generate synthetic HPA-like training data"""
    
    def __init__(self):
        self.scenarios = [
            self._low_load_scenario,
            self._medium_load_scenario,
            self._high_load_scenario,
            self._spike_scenario,
            self._declining_load_scenario
        ]
    
    def _low_load_scenario(self):
        """Low traffic, should scale down"""
        return {
            "rps": random.uniform(5, 20),
            "cpu_util": random.uniform(10, 30),
            "latency": random.uniform(5, 15),
            "pods": random.randint(3, 5),
            "hpa_action": -1
        }
    
    def _medium_load_scenario(self):
        """Medium traffic, maintain"""
        return {
            "rps": random.uniform(40, 80),
            "cpu_util": random.uniform(45, 55),
            "latency": random.uniform(10, 30),
            "pods": random.randint(2, 4),
            "hpa_action": 0
        }
    
    def _high_load_scenario(self):
        """High traffic, scale up"""
        return {
            "rps": random.uniform(100, 200),
            "cpu_util": random.uniform(70, 95),
            "latency": random.uniform(50, 200),
            "pods": random.randint(1, 3),
            "hpa_action": 2
        }
    
    def _spike_scenario(self):
        """Traffic spike, aggressive scale up"""
        return {
            "rps": random.uniform(200, 400),
            "cpu_util": random.uniform(85, 99),
            "latency": random.uniform(200, 1000),
            "pods": random.randint(1, 4),
            "hpa_action": 3
        }
    
    def _declining_load_scenario(self):
        """Load decreasing, gradual scale down"""
        return {
            "rps": random.uniform(20, 50),
            "cpu_util": random.uniform(25, 40),
            "latency": random.uniform(8, 20),
            "pods": random.randint(4, 7),
            "hpa_action": -1
        }
    
    def _build_state(self, scenario_data):
        """Build full state vector from scenario"""
        pods = int(scenario_data["pods"])  # Ensure Python int
        cpu_util = float(scenario_data["cpu_util"])  # Ensure Python float
        
        return {
            "workload": {
                "rps": float(scenario_data["rps"]),
                "p95_latency_ms": float(scenario_data["latency"]),
                "error_rate_pct": float(random.uniform(0, 2)),
                "queue_length": float(random.uniform(0, 10))
            },
            "infra": {
                "pods_ready": pods,
                "hpa_desired_replicas": pods,
                "cpu_utilization_pct": cpu_util,
                "mem_utilization_pct": float(cpu_util * 0.8),
                "node_count": 1,
                "pod_cpu_request_cores": 0.1,
                "pod_cpu_limit_cores": 0.5
            },
            "cost": {
                "cost_per_min_usd": 0.0,
                "spot_ratio": 0.0
            },
            "time": {
                "minute_of_day_sin": float(random.uniform(-1, 1)),
                "minute_of_day_cos": float(random.uniform(-1, 1)),
                "day_of_week": int(random.randint(0, 6))
            },
            "scaling": {
                "last_action_delta": 0,
                "steps_since_action": int(random.randint(0, 10))
            },
            "trend": {
                "cpu_slope": float(random.uniform(-5, 5)),
                "rps_slope": float(random.uniform(-10, 10)),
                "latency_slope": float(random.uniform(-20, 20))
            }
        }
    
    def _calculate_reward(self, state, action_delta):
        """Calculate reward based on state and action"""
        workload = state['workload']
        infra = state['infra']
        
        reward = 0.0
        
        # Latency reward
        latency = workload['p95_latency_ms']
        if latency < 50:
            reward += 5.0
        elif latency < 100:
            reward += 2.0
        elif latency > 500:
            reward -= 5.0
        
        # CPU efficiency
        cpu = infra['cpu_utilization_pct']
        if 50 <= cpu <= 70:
            reward += 3.0
        elif cpu > 85:
            reward -= 3.0
        elif cpu < 30 and infra['pods_ready'] > 1:
            reward -= 1.0
        
        # Error penalty
        if workload['error_rate_pct'] > 1:
            reward -= workload['error_rate_pct'] * 2
        
        # Stability
        if action_delta != 0:
            reward -= 0.3
        
        return float(reward)
    
    def generate_dataset(self, num_samples=1000, filename="synthetic_hpa_data.json", scenario_weights=None):
        """
        Generate dataset with custom scenario distribution
        
        scenario_weights: dict like {"low": 0.2, "medium": 0.4, "high": 0.3, "spike": 0.05, "declining": 0.05}
        """
        if scenario_weights is None:
            # Default balanced distribution
            scenario_weights = {
                "low": 0.2,
                "medium": 0.4,
                "high": 0.25,
                "spike": 0.1,
                "declining": 0.05
            }
        
        # Map scenarios
        scenario_map = {
            "low": self._low_load_scenario,
            "medium": self._medium_load_scenario,
            "high": self._high_load_scenario,
            "spike": self._spike_scenario,
            "declining": self._declining_load_scenario
        }
        
        observations = []
        
        for i in range(num_samples):
            # Pick scenario based on weights
            scenario_name = random.choices(
                list(scenario_weights.keys()),
                weights=list(scenario_weights.values()),
                k=1
            )[0]
            
            scenario_func = scenario_map[scenario_name]
            scenario = scenario_func()
            
            # Build current state
            current_state = self._build_state(scenario)
            action_delta = int(scenario["hpa_action"])  # Ensure Python int
            
            # Simulate next state after action
            next_scenario = scenario.copy()
            # FIX: Convert np.clip result to Python int
            next_scenario["pods"] = int(np.clip(
                scenario["pods"] + action_delta, 1, 10
            ))
            
            # Adjust metrics based on action
            if action_delta > 0:
                next_scenario["cpu_util"] = float(next_scenario["cpu_util"] * 0.7)
                next_scenario["latency"] = float(next_scenario["latency"] * 0.8)
            elif action_delta < 0:
                next_scenario["cpu_util"] = float(next_scenario["cpu_util"] * 1.3)
                next_scenario["latency"] = float(next_scenario["latency"] * 1.1)
            
            next_state = self._build_state(next_scenario)
            
            # Calculate reward
            reward = self._calculate_reward(next_state, action_delta)
            
            # Create observation
            observation = {
                "timestamp": datetime.utcnow().isoformat(),
                "current_state": current_state,
                "action_delta": action_delta,
                "next_state": next_state,
                "reward": float(reward),  # Ensure Python float
                "current_replicas": int(current_state['infra']['pods_ready']),
                "hpa_desired": int(next_state['infra']['hpa_desired_replicas'])
            }
            
            observations.append(observation)
        
        # Save to file
        import os
        os.makedirs("data/hpa_observations", exist_ok=True)
        
        filepath = f"data/hpa_observations/{filename}"
        with open(filepath, 'w') as f:
            json.dump(observations, f, indent=2)
        
        print(f"Generated {num_samples} observations → {filepath}")
        return filename
