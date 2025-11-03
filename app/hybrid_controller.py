import asyncio
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class HybridController:
    """Hybrid controller that gradually transitions from HPA to PPO"""
    
    def __init__(self, k8s_manager, ppo_agent, scaling_controller,monitor=None):
        self.k8s_manager = k8s_manager
        self.ppo_agent = ppo_agent
        self.scaling_controller = scaling_controller
        self.ppo_confidence = 0.0  # 0 = full HPA, 1 = full PPO
        self.confidence_increment = 0.05  # Increase confidence gradually
        self.is_running = False
        self.control_interval = 30
        self.monitor = monitor
        # Performance tracking
        self.hpa_performance = []
        self.ppo_performance = []
        self.recent_window = 20  # Track last 20 decisions
    
    def _evaluate_decision_quality(self, state_dict, action_taken: int) -> float:
        """Evaluate quality of a scaling decision"""
        workload = state_dict['workload']
        infra = state_dict['infra']
        
        score = 0.0
        
        # Good: Latency under target
        if workload['p95_latency_ms'] < 1000:
            score += 3.0
        elif workload['p95_latency_ms'] < 1500:
            score += 1.0
        else:
            score -= 3.0
        
        # Good: Low error rate
        if workload['error_rate_pct'] < 1.0:
            score += 2.0
        else:
            score -= 2.0
        
        # Good: Efficient CPU usage
        cpu = infra['cpu_utilization_pct']
        if 60 <= cpu <= 75:
            score += 2.0
        elif cpu > 85:
            score -= 2.0
        
        # Good: Stability (fewer actions)
        if action_taken == 0:
            score += 1.0
        
        return score
    
    async def hybrid_control_loop(self, state_provider_func):
        """Hybrid control loop with gradual PPO adoption"""
        logger.info(f"Starting hybrid HPA+PPO control loop")
        self.is_running = True
        
        while self.is_running:
            try:
                # Get current state
                state_dict = await state_provider_func()
                current_replicas = state_dict['infra']['pods_ready']
                hpa_desired = state_dict['infra']['hpa_desired_replicas']
                
                # Get HPA decision
                hpa_action_delta = hpa_desired - current_replicas
                
                # Get PPO decision
                ppo_new_replicas, ppo_action_delta = self.ppo_agent.predict_action(state_dict)
                
                # Blend decisions based on confidence
                if np.random.random() < self.ppo_confidence:
                    chosen_action_delta = ppo_action_delta
                    decision_source = "PPO"
                else:
                    chosen_action_delta = hpa_action_delta
                    decision_source = "HPA"
                
                # LOG THE DECISION - ADD THIS BLOCK
                if self.monitor:
                    self.monitor.log_decision(
                        state=state_dict,
                        decision={
                            "source": decision_source,
                            "action_delta": chosen_action_delta,
                            "hpa_suggestion": hpa_action_delta,
                            "ppo_suggestion": ppo_action_delta,
                            "ppo_confidence": self.ppo_confidence,
                            "current_replicas": current_replicas,
                            "new_replicas": current_replicas + chosen_action_delta
                        }
                    )
                
                # Apply action
                if chosen_action_delta != 0:
                    new_replicas = current_replicas + chosen_action_delta
                    await self.scaling_controller.apply_scaling_action(new_replicas)
                
                # Wait and evaluate
                await asyncio.sleep(self.control_interval)
                
                # Get new state and evaluate decision quality
                new_state = await state_provider_func()
                decision_quality = self._evaluate_decision_quality(new_state, chosen_action_delta)
                
                # LOG THE OUTCOME - ADD THIS BLOCK
                if self.monitor:
                    self.monitor.log_decision(
                        state=state_dict,
                        decision={
                            "source": decision_source,
                            "action_delta": chosen_action_delta
                        },
                        outcome={
                            "quality_score": decision_quality,
                            "new_latency_p95": new_state['workload']['p95_latency_ms'],
                            "new_cpu_util": new_state['infra']['cpu_utilization_pct'],
                            "new_error_rate": new_state['workload']['error_rate_pct'],
                            "new_replicas": new_state['infra']['pods_ready']
                        }
                    )
                
                # Track performance (existing code continues...)
                if decision_source == "HPA":
                    self.hpa_performance.append(decision_quality)
                    if len(self.hpa_performance) > self.recent_window:
                        self.hpa_performance.pop(0)
                else:
                    self.ppo_performance.append(decision_quality)
                    if len(self.ppo_performance) > self.recent_window:
                        self.ppo_performance.pop(0)
                
                # Adjust confidence based on relative performance
                if len(self.ppo_performance) >= 10 and len(self.hpa_performance) >= 10:
                    ppo_avg = np.mean(self.ppo_performance)
                    hpa_avg = np.mean(self.hpa_performance)
                    
                    logger.info(f"Performance: PPO avg={ppo_avg:.2f}, HPA avg={hpa_avg:.2f}")
                    
                    # Increase PPO confidence if it's performing better
                    if ppo_avg > hpa_avg + 1.0:  # PPO significantly better
                        self.ppo_confidence = min(1.0, self.ppo_confidence + self.confidence_increment)
                        logger.info(f"Increasing PPO confidence to {self.ppo_confidence:.2f}")
                    elif ppo_avg < hpa_avg - 1.0:  # PPO significantly worse
                        self.ppo_confidence = max(0.0, self.ppo_confidence - self.confidence_increment)
                        logger.info(f"Decreasing PPO confidence to {self.ppo_confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Error in hybrid control loop: {e}")
                await asyncio.sleep(self.control_interval)
    
    def stop(self):
        """Stop hybrid control loop"""
        self.is_running = False
        logger.info("Hybrid control loop stopped")
    
    def get_status(self):
        """Get hybrid controller status"""
        return {
            "is_running": self.is_running,
            "ppo_confidence": self.ppo_confidence,
            "hpa_performance_avg": np.mean(self.hpa_performance) if self.hpa_performance else 0.0,
            "ppo_performance_avg": np.mean(self.ppo_performance) if self.ppo_performance else 0.0,
            "hpa_samples": len(self.hpa_performance),
            "ppo_samples": len(self.ppo_performance)
        }
