"""Runtime scene-status contract used as an event-logic guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SceneStatus:
    """Frame-level scene guard.

    M6 will provide a detector for these anomalies.  M4/M5 already consume
    this contract so timers can be frozen deterministically in tests and in a
    future runtime monitor.
    """

    stable: bool = True
    lighting_change: bool = False
    camera_shift: bool = False
    occlusion: bool = False
    reason: str | None = None

    @property
    def event_logic_enabled(self) -> bool:
        return self.stable and not self.lighting_change and not self.camera_shift

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable": self.stable,
            "lighting_change": self.lighting_change,
            "camera_shift": self.camera_shift,
            "occlusion": self.occlusion,
            "reason": self.reason,
        }


class SceneStatusProvider(Protocol):
    """Minimal provider protocol required by the runtime pipeline."""

    def update(self, frame: Any) -> SceneStatus: ...

    def reset(self) -> None: ...


class StableSceneStatusProvider:
    """No-op provider used until the M6 scene monitor is enabled."""

    def update(self, frame: Any) -> SceneStatus:
        del frame
        return SceneStatus()

    def reset(self) -> None:
        return None

