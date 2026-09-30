"""Event-level matching and metric computation."""

from .evaluator import EvaluationConfig, evaluate_events, summarize_timeliness

__all__ = ["EvaluationConfig", "evaluate_events", "summarize_timeliness"]
