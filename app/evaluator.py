import asyncio
import logging
from typing import Dict, List
from datetime import datetime
import json
import os
import statistics

logger = logging.getLogger(__name__)

class PerformanceEvaluator:
    """Evaluate and compare autoscaling strategies"""
    
    def __init__(self):
        self.metrics_log = []
        self.evaluation_results = {}
        
    async def run_evaluation_scenario(
        self, 
        state_provider_func,
        duration_minutes: int = 30,
        sample_interval: int = 30,
        strategy_name: str = "unknown"
    ) -> Dict:
        """
        Run evaluation for a specific strategy
        
        Args:
            state_provider_func: Function to get current state
            duration_minutes: How long to run evaluation
            sample_interval: Seconds between samples
            strategy_name: "hpa", "hybrid", or "ppo"
        """
        logger.info(f"Starting {strategy_name} evaluation for {duration_minutes} minutes")
        
        metrics = []
        num_samples = (duration_minutes * 60) // sample_interval
        
        for i in range(num_samples):
            try:
                # Collect current state
                state = await state_provider_func()
                
                # Extract key metrics
                sample = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "strategy": strategy_name,
                    "latency_p95": state['workload']['p95_latency_ms'],
                    "rps": state['workload']['rps'],
                    "error_rate": state['workload']['error_rate_pct'],
                    "cpu_util": state['infra']['cpu_utilization_pct'],
                    "mem_util": state['infra']['mem_utilization_pct'],
                    "pods": state['infra']['pods_ready'],
                    "cpu_slope": state['trend']['cpu_slope'],
                    "latency_slope": state['trend']['latency_slope']
                }
                
                metrics.append(sample)
                logger.info(f"{strategy_name} sample {i+1}/{num_samples}: "
                          f"latency={sample['latency_p95']:.1f}ms, "
                          f"cpu={sample['cpu_util']:.1f}%, "
                          f"pods={sample['pods']}")
                
                await asyncio.sleep(sample_interval)
                
            except Exception as e:
                logger.error(f"Error collecting sample: {e}")
        
        # Calculate statistics
        results = self._calculate_statistics(metrics, strategy_name)
        
        # Save results
        self._save_results(results, strategy_name)
        
        return results
    
    def _calculate_statistics(self, metrics: List[Dict], strategy: str) -> Dict:
        """Calculate performance statistics"""
        if not metrics:
            return {}
        
        latencies = [m['latency_p95'] for m in metrics]
        cpu_utils = [m['cpu_util'] for m in metrics]
        error_rates = [m['error_rate'] for m in metrics]
        pods_counts = [m['pods'] for m in metrics]
        
        # Calculate SLA compliance (latency < 1000ms)
        sla_violations = sum(1 for lat in latencies if lat > 1000)
        sla_compliance_pct = ((len(latencies) - sla_violations) / len(latencies)) * 100
        
        # Calculate pod efficiency (fewer pods = better efficiency)
        avg_pods = statistics.mean(pods_counts)
        
        # Calculate stability (fewer changes = more stable)
        pod_changes = sum(1 for i in range(1, len(pods_counts)) 
                         if pods_counts[i] != pods_counts[i-1])
        stability_score = 100 - (pod_changes / len(pods_counts) * 100)
        
        results = {
            "strategy": strategy,
            "samples": len(metrics),
            "latency": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "p95": sorted(latencies)[int(len(latencies) * 0.95)],
                "max": max(latencies),
                "min": min(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            "cpu_utilization": {
                "mean": statistics.mean(cpu_utils),
                "median": statistics.median(cpu_utils),
                "max": max(cpu_utils),
                "min": min(cpu_utils)
            },
            "error_rate": {
                "mean": statistics.mean(error_rates),
                "max": max(error_rates),
                "total_errors_pct": sum(error_rates) / len(error_rates)
            },
            "resource_efficiency": {
                "avg_pods": avg_pods,
                "max_pods": max(pods_counts),
                "min_pods": min(pods_counts),
                "pod_changes": pod_changes,
                "stability_score": stability_score
            },
            "sla": {
                "compliance_pct": sla_compliance_pct,
                "violations": sla_violations,
                "total_samples": len(latencies)
            },
            "overall_score": self._calculate_overall_score(
                sla_compliance_pct, 
                statistics.mean(latencies),
                avg_pods,
                stability_score
            )
        }
        
        return results
    
    def _calculate_overall_score(
        self, 
        sla_compliance: float, 
        avg_latency: float,
        avg_pods: float,
        stability: float
    ) -> float:
        """
        Calculate overall performance score (0-100)
        
        Weights:
        - SLA compliance: 40%
        - Latency: 30%
        - Resource efficiency: 20%
        - Stability: 10%
        """
        # SLA score
        sla_score = sla_compliance
        
        # Latency score (inverse - lower is better)
        # Perfect: <100ms = 100, Bad: >1000ms = 0
        latency_score = max(0, 100 - (avg_latency / 10))
        
        # Efficiency score (inverse - fewer pods is better)
        # Ideal: 2 pods = 100, Wasteful: 10 pods = 0
        efficiency_score = max(0, 100 - ((avg_pods - 1) * 11))
        
        # Stability score (already 0-100)
        
        overall = (
            sla_score * 0.4 +
            latency_score * 0.3 +
            efficiency_score * 0.2 +
            stability * 0.1
        )
        
        return round(overall, 2)
    
    def _save_results(self, results: Dict, strategy: str):
        """Save evaluation results"""
        os.makedirs("data/evaluations", exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"data/evaluations/{strategy}_eval_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved {strategy} evaluation results to {filename}")
    
    def compare_strategies(self, strategy_files: List[str]) -> Dict:
        """Compare multiple evaluation results"""
        comparisons = []
        
        for filepath in strategy_files:
            with open(filepath, 'r') as f:
                results = json.load(f)
                comparisons.append(results)
        
        # Generate comparison report
        comparison_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "strategies": {}
        }
        
        for result in comparisons:
            strategy = result['strategy']
            comparison_report['strategies'][strategy] = {
                "overall_score": result['overall_score'],
                "avg_latency": result['latency']['mean'],
                "sla_compliance": result['sla']['compliance_pct'],
                "avg_pods": result['resource_efficiency']['avg_pods'],
                "stability": result['resource_efficiency']['stability_score']
            }
        
        # Determine winner
        best_strategy = max(
            comparison_report['strategies'].items(),
            key=lambda x: x[1]['overall_score']
        )
        
        comparison_report['winner'] = {
            "strategy": best_strategy[0],
            "score": best_strategy[1]['overall_score']
        }
        
        # Calculate improvements
        if 'hpa' in comparison_report['strategies']:
            baseline = comparison_report['strategies']['hpa']
            
            for strategy, metrics in comparison_report['strategies'].items():
                if strategy != 'hpa':
                    metrics['improvement_vs_hpa'] = {
                        "latency_reduction_pct": round(
                            ((baseline['avg_latency'] - metrics['avg_latency']) / baseline['avg_latency']) * 100, 2
                        ),
                        "sla_improvement_pct": round(
                            metrics['sla_compliance'] - baseline['sla_compliance'], 2
                        ),
                        "resource_savings_pct": round(
                            ((baseline['avg_pods'] - metrics['avg_pods']) / baseline['avg_pods']) * 100, 2
                        )
                    }
        
        # Save comparison
        filename = f"data/evaluations/comparison_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        return comparison_report
