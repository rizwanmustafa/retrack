"""
AIModel: the application integration interface that wraps the multi-view backend (SRS 1, NFR-14).

The web, persistence and monitoring code depend only on this module. A backend exposes loadCheckpoint,
resetTracking and analyze (class diagram: Retrack), plus checkpoint inspection used by checkpoint administration.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional


class BackendUnavailable(Exception):
    """The inference environment (libraries, device) is not installed or not usable."""


class ModelError(Exception):
    """A fatal model or resource error during loading or inference (FR-24)."""


@dataclass
class CheckpointInfo:
    compatible: bool
    problems: List[str] = field(default_factory=list)
    model_name: str = ''
    version: str = ''
    details: Dict[str, str] = field(default_factory=dict)  # displayable configuration, no paths


@dataclass
class Observation:
    """One observed session-local track in one processed frame set."""
    track_id: int
    x_cm: float
    y_cm: float
    confidence: float


@dataclass
class AnalysisResult:
    observations: List[Observation]


class AIModel(ABC):
    backend_name = 'abstract'

    @abstractmethod
    def inspect_checkpoint(self, path) -> CheckpointInfo:
        """Checks a checkpoint against the camera count, scene configuration, backend and output contract."""

    @abstractmethod
    def load_checkpoint(self, path) -> None:
        """Loads a compatible checkpoint for inference. Raises BackendUnavailable or ModelError."""

    @abstractmethod
    def reset_tracking(self) -> None:
        """Starts a fresh tracking state; track IDs restart for the new session (FR-13, FR-19)."""

    @abstractmethod
    def analyze(self, frame_set, calibration) -> AnalysisResult:
        """
        Runs detection and temporal association on one validated FrameSet.
        frame_set.images: {cameraId: HxWx3 uint8 RGB array} in the configured order.
        calibration: the session's stored calibration document.
        Returns the tracks observed in this frame set, positions in world cm. Raises ModelError on failure.
        """

    @abstractmethod
    def describe_environment(self) -> Dict[str, str]:
        """Inference environment for reproducibility records (NFR-12)."""

    @property
    @abstractmethod
    def loaded_checkpoint(self) -> Optional[str]:
        """Path of the checkpoint currently loaded, or None."""

    # The operation names used by NFR-14 and the class diagram
    def loadCheckpoint(self, path):
        return self.load_checkpoint(path)

    def resetTracking(self):
        return self.reset_tracking()
