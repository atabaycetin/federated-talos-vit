"""Prometheus metrics. Each app instance owns its registry so tests don't collide."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)


class ServiceMetrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("talos_requests", "Prediction requests", ["task", "status"], registry=r)
        self.request_latency = Histogram(
            "talos_request_latency_seconds",
            "End-to-end /predict latency",
            ["task"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.inference_latency = Histogram(
            "talos_inference_latency_seconds",
            "Model forward pass latency",
            ["task"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.confidence = Histogram(
            "talos_prediction_confidence",
            "Top-1 softmax probability",
            ["task"],
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99),
            registry=r,
        )
        self.predicted_class = Counter(
            "talos_predicted_class", "Top-1 predictions", ["task", "label"], registry=r
        )
        self.drift_psi = Gauge(
            "talos_drift_psi", "PSI of live vs. training inputs", ["task", "feature"], registry=r
        )
        self.drift_alert = Gauge(
            "talos_drift_alert", "1 if any feature's PSI exceeds the alert level", ["task"], registry=r
        )
        self.task_swap = Histogram(
            "talos_task_swap_seconds",
            "Time to swap the active task vector",
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.model_info = Gauge(
            "talos_model_info", "Loaded models", ["task", "version", "method", "backend"], registry=r
        )
