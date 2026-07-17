from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(slots=True, kw_only=True)
class InteractionControl:
    """Base class for controls that alter animation state."""

    id: str
    type: str
    label: str
    target_property: str
    description: str | None = None

    def to_spec(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True, kw_only=True)
class SliderControl(InteractionControl):
    """Numeric parameter control, such as speed or sensor radius."""

    min_value: float = 0
    max_value: float = 1
    default_value: float = 0
    step: float = 1
    type: Literal["slider"] = "slider"


@dataclass(slots=True, kw_only=True)
class ToggleControl(InteractionControl):
    """Boolean state control, such as showing rays or pausing collisions."""

    default_value: bool = False
    type: Literal["toggle"] = "toggle"


@dataclass(slots=True, kw_only=True)
class ButtonControl(InteractionControl):
    """Command control, such as reset, replay, or single-step."""

    action: str = ""
    type: Literal["button"] = "button"


@dataclass(slots=True, kw_only=True)
class DragTarget:
    """Pointer interaction definition for draggable scene elements."""

    element_id: str
    bounds: dict[str, float] = field(default_factory=dict)
    snap_to_grid: bool = False

    def to_spec(self) -> dict[str, Any]:
        return asdict(self)


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
