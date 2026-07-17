from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from animate_agent.interaction.controls import InteractionControl


class SpecSerializable(Protocol):
    """Object that can be exported to the frontend animation schema."""

    def to_spec(self) -> dict[str, Any]:
        """Return a JSON-serializable animation spec."""


@dataclass(slots=True, kw_only=True)
class AnimatedElement:
    """Base class for visible objects in a teaching animation."""

    id: str
    type: str
    x: float
    y: float
    label: str | None = None
    style: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True, kw_only=True)
class ManualObject(AnimatedElement):
    """Generic object extracted from a manual when no domain-specific class exists."""

    type: Literal["manual_object"] = "manual_object"
    description: str | None = None


@dataclass(slots=True, kw_only=True)
class RobotCar(AnimatedElement):
    """Reusable 2D robot car primitive for robotics manuals."""

    type: Literal["robot_car"] = "robot_car"
    width: float = 62
    height: float = 40
    heading: float = 0
    speed: float = 1.0
    show_heading: bool = True
    show_trail: bool = True


@dataclass(slots=True, kw_only=True)
class LidarSensor(AnimatedElement):
    """Lidar fan and ray-cast visualization tied to another element."""

    x: float = 0
    y: float = 0
    owner_id: str = ""
    type: Literal["lidar_sensor"] = "lidar_sensor"
    radius: float = 150
    fov_degrees: float = 180
    ray_count: int = 13
    scan_interval_ms: int = 80


@dataclass(slots=True, kw_only=True)
class Obstacle(AnimatedElement):
    """Obstacle primitive that can be detected or dragged in demos."""

    type: Literal["obstacle"] = "obstacle"
    radius: float = 32
    draggable: bool = True


@dataclass(slots=True, kw_only=True)
class Circle(AnimatedElement):
    """Generic circle, useful for thresholds, ranges, and focus areas."""

    type: Literal["circle"] = "circle"
    radius: float = 40
    dashed: bool = False
    fill_opacity: float = 0.1


@dataclass(slots=True, kw_only=True)
class Label(AnimatedElement):
    """Text label anchored in the scene."""

    type: Literal["label"] = "label"
    text: str = ""
    anchor_element_id: str | None = None


@dataclass(slots=True, kw_only=True)
class RosNode(AnimatedElement):
    """ROS node shown in a topology or runtime animation."""

    type: Literal["ros_node"] = "ros_node"
    package: str | None = None
    executable: str | None = None
    role: Literal["publisher", "subscriber", "processor", "sensor", "actuator"] = "processor"
    source_ref: str | None = None


@dataclass(slots=True, kw_only=True)
class RosTopic(AnimatedElement):
    """ROS topic edge between publisher and subscriber nodes."""

    type: Literal["ros_topic"] = "ros_topic"
    name: str
    from_node_id: str
    to_node_id: str
    message_type: str
    source_ref: str | None = None


@dataclass(slots=True, kw_only=True)
class MessagePacket(AnimatedElement):
    """Message packet that travels along a topic edge."""

    type: Literal["message_packet"] = "message_packet"
    topic_id: str
    payload_label: str = ""
    state: Literal["queued", "in_flight", "received", "error"] = "queued"


@dataclass(slots=True, kw_only=True)
class TimelineStep:
    """One teachable moment in a generated animation."""

    id: str
    title: str
    narration: str
    focus_element_ids: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def to_spec(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True, kw_only=True)
class Scene:
    """Complete animation spec consumed by the frontend renderer."""

    title: str
    elements: list[SpecSerializable] = field(default_factory=list)
    timeline: list[TimelineStep] = field(default_factory=list)
    controls: list[InteractionControl] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "elements": [element.to_spec() for element in self.elements],
            "timeline": [step.to_spec() for step in self.timeline],
            "controls": [control.to_spec() for control in self.controls],
            "metadata": self.metadata,
        }


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}
