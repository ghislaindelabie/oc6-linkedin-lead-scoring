"""Monitoring package: drift detection, dashboard utilities, profiling, and ONNX optimization."""
from linkedin_lead_scoring.monitoring.drift import DriftDetector
from linkedin_lead_scoring.monitoring import dashboard_utils
from linkedin_lead_scoring.monitoring import profiler

try:
    from linkedin_lead_scoring.monitoring import onnx_optimizer
except ImportError:
    onnx_optimizer = None  # onnx/onnxruntime not installed

__all__ = ["DriftDetector", "dashboard_utils", "profiler", "onnx_optimizer"]
