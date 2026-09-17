# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    Detection-only build: this project only needs the standard detect task.
    The class wraps the Ultralytics ``Model`` base with a detect-only task map.
    """

    def __init__(self, model: str | Path = "yolo11m.yaml", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        Args:
            model (str | Path): Model name or path to a YAML config or .pt weights file.
            task (str, optional): Task type; defaults to auto-detection ('detect').
            verbose (bool): Display model info on load.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes (detect only)."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
        }
