from animate_agent.animation.templates import (
    build_robot_obstacle_avoidance_scene,
    build_ros_pub_sub_scene,
)


def test_robot_obstacle_avoidance_template_exports_expected_spec() -> None:
    spec = build_robot_obstacle_avoidance_scene().to_spec()

    assert spec["metadata"]["template"] == "robot_obstacle_avoidance"
    assert len(spec["timeline"]) >= 4
    assert {element["type"] for element in spec["elements"]} >= {
        "robot_car",
        "lidar_sensor",
        "obstacle",
    }


def test_ros_pub_sub_template_exports_topology_objects() -> None:
    spec = build_ros_pub_sub_scene().to_spec()

    assert spec["metadata"]["template"] == "ros_pub_sub"
    assert {element["type"] for element in spec["elements"]} >= {
        "ros_node",
        "ros_topic",
        "message_packet",
        "robot_car",
    }
