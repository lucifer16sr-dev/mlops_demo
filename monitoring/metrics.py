
import time
from typing import Optional
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import CONTENT_TYPE_LATEST
import logging

logger = logging.getLogger(__name__)


class MetricsCollector:
       
    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            'mlops_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'mlops_request_duration_seconds',
            'Request duration in seconds',
            ['method', 'endpoint'],
            buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Model inference metrics
        self.inference_count = Counter(
            'mlops_inference_total',
            'Total number of model inferences',
            ['model_name', 'status']
        )
        
        self.inference_duration = Histogram(
            'mlops_inference_duration_seconds',
            'Model inference duration in seconds',
            ['model_name'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
        )
        
        # Batch metrics
        self.batch_size = Histogram(
            'mlops_batch_size',
            'Batch size distribution',
            ['model_name'],
            buckets=[1, 2, 4, 8, 16, 32, 64, 100]
        )
        
        # Error metrics
        self.error_count = Counter(
            'mlops_errors_total',
            'Total number of errors',
            ['error_type', 'endpoint']
        )
        
        # Active requests gauge
        self.active_requests = Gauge(
            'mlops_active_requests',
            'Number of active requests being processed'
        )
        
        # Model health gauge
        self.model_health = Gauge(
            'mlops_model_health',
            'Model health status (1=healthy, 0=unhealthy)',
            ['model_name']
        )
        
        logger.info("MetricsCollector initialized")
    
    def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        self.request_count.labels(method=method, endpoint=endpoint, status_code=status_code).inc()
        self.request_duration.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_inference(self, model_name: str, duration: float, status: str = "success"):
        self.inference_count.labels(model_name=model_name, status=status).inc()
        self.inference_duration.labels(model_name=model_name).observe(duration)
    
    def record_batch(self, model_name: str, batch_size: int):
        self.batch_size.labels(model_name=model_name).observe(batch_size)
    
    def record_error(self, error_type: str, endpoint: str):
        self.error_count.labels(error_type=error_type, endpoint=endpoint).inc()
    
    def set_active_requests(self, count: int):
        self.active_requests.set(count)
    
    def set_model_health(self, model_name: str, is_healthy: bool):
        self.model_health.labels(model_name=model_name).set(1 if is_healthy else 0)
    
    def get_metrics(self):
        return generate_latest(REGISTRY)
    
    def get_content_type(self):
        return CONTENT_TYPE_LATEST


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector():
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector