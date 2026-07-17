from animate_agent.animation import LidarSensor, Obstacle, RobotCar, Scene, TimelineStep
from animate_agent.interaction import SliderControl


def test_scene_serializes_reusable_robot_demo_objects() -> None:
    scene = Scene(
        title="小车避障",
        elements=[
            RobotCar(id="car", x=120, y=310, heading=0),
            LidarSensor(id="lidar", owner_id="car", radius=150, fov_degrees=180),
            Obstacle(id="obstacle-1", x=420, y=300, radius=34),
        ],
        timeline=[
            TimelineStep(
                id="scan",
                title="雷达扫描",
                narration="雷达向前方发射多条测距射线。",
                focus_element_ids=["lidar"],
            )
        ],
        controls=[
            SliderControl(
                id="safe-distance",
                label="安全距离",
                min_value=45,
                max_value=120,
                default_value=76,
                target_property="scene.safe_distance",
            )
        ],
    )

    payload = scene.to_spec()

    assert payload["title"] == "小车避障"
    assert payload["elements"][0]["type"] == "robot_car"
    assert payload["elements"][1]["type"] == "lidar_sensor"
    assert payload["controls"][0]["type"] == "slider"

