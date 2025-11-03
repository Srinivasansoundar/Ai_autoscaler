import logging
from datetime import datetime
from typing import Dict, List
import json
import os

logger = logging.getLogger(__name__)

class DecisionMonitor:
    """Monitor and log scaling decisions for analysis"""
    
    def __init__(self, log_dir="logs/decisions"):
        self.log_dir = log_dir
        self.decisions = []
        os.makedirs(log_dir, exist_ok=True)
    
    def log_decision(self, state: Dict, decision: Dict, outcome: Dict = None):
        """Log a scaling decision with context"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "state": {
                "latency_p95": state['workload']['p95_latency_ms'],
                "rps": state['workload']['rps'],
                "error_rate": state['workload']['error_rate_pct'],
                "cpu_util": state['infra']['cpu_utilization_pct'],
                "replicas": state['infra']['pods_ready'],
                "hpa_desired": state['infra']['hpa_desired_replicas']
            },
            "decision": decision,
            "outcome": outcome
        }
        
        self.decisions.append(entry)
        
        # Log to file
        self._write_to_file(entry)
        
        return entry
    
    def _write_to_file(self, entry):
        """Write entry to daily log file"""
        date_str = datetime.utcnow().strftime('%Y%m%d')
        filepath = os.path.join(self.log_dir, f"decisions_{date_str}.jsonl")
        
        with open(filepath, 'a') as f:
            f.write(json.dumps(entry) + '\n')
    
    def get_recent_decisions(self, n: int = 50) -> List[Dict]:
        """Get recent decisions"""
        return self.decisions[-n:]
    
    def analyze_performance(self, window: int = 100) -> Dict:
        """Analyze recent performance"""
        recent = self.decisions[-window:]
        
        if not recent:
            return {}
        
        latencies = [d['state']['latency_p95'] for d in recent]
        cpus = [d['state']['cpu_util'] for d in recent]
        error_rates = [d['state']['error_rate'] for d in recent]
        
        return {
            "window_size": len(recent),
            "avg_latency_p95": sum(latencies) / len(latencies),
            "max_latency_p95": max(latencies),
            "avg_cpu_util": sum(cpus) / len(cpus),
            "avg_error_rate": sum(error_rates) / len(error_rates),
            "sla_violations": sum(1 for l in latencies if l > 1000),
            "sla_compliance_pct": (1 - sum(1 for l in latencies if l > 1000) / len(latencies)) * 100
        }
