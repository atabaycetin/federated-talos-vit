"""Thin MLflow wrapper so every stage logs to the same run, and tests can switch it off."""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path

from talos_mlops.config import PipelineConfig


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, f"{key}."))
        else:
            out[key] = v
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class Tracker:
    def __init__(self, cfg: PipelineConfig):
        self.enabled = cfg.mlflow.enabled
        self.cfg = cfg
        if self.enabled:
            import mlflow

            self.mlflow = mlflow
            mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
            mlflow.set_experiment(cfg.mlflow.experiment)

    @contextmanager
    def run(self, run_id: str | None = None):
        if not self.enabled:
            yield None
            return
        with self.mlflow.start_run(run_id=run_id, run_name=None if run_id else self.cfg.run_name) as run:
            if run_id is None:
                self.mlflow.log_params(_flatten(self.cfg.model_dump()))
                self.mlflow.set_tags({"task": self.cfg.task, "git_commit": _git_commit() or "unknown"})
            yield run.info.run_id

    def metrics(self, values: dict[str, float], step: int | None = None) -> None:
        if self.enabled:
            self.mlflow.log_metrics(values, step=step)

    def artifact(self, path: Path, subdir: str | None = None) -> None:
        if self.enabled:
            self.mlflow.log_artifact(str(path), artifact_path=subdir)

    def register_onnx(self, onnx_path: Path, metadata: dict) -> str | None:
        """Log the served ONNX model and register a new version in the model registry."""
        if not self.enabled:
            return None
        import onnx

        info = self.mlflow.onnx.log_model(
            onnx.load(str(onnx_path)),
            name="model",
            registered_model_name=self.cfg.registered_model_name,
            metadata=metadata,
        )
        return str(info.registered_model_version)
