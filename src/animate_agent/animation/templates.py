from __future__ import annotations

from animate_agent.animation.elements import (
    LidarSensor,
    MessagePacket,
    Obstacle,
    RobotCar,
    RosNode,
    RosTopic,
    Scene,
    TimelineStep,
)
from animate_agent.interaction.controls import ButtonControl, SliderControl


def build_robot_obstacle_avoidance_scene() -> Scene:
    """Create a reusable scene spec for robot obstacle-avoidance manuals."""

    return Scene(
        title="小车避障教学动画",
        elements=[
            RobotCar(id="car", x=120, y=310, label="小车"),
            LidarSensor(id="lidar", x=120, y=310, owner_id="car", label="2D 激光雷达"),
            Obstacle(id="obstacle-1", x=410, y=300, radius=34, label="障碍物 A"),
            Obstacle(id="obstacle-2", x=570, y=210, radius=27, label="障碍物 B"),
        ],
        timeline=[
            TimelineStep(
                id="scan",
                title="雷达扫描",
                narration="雷达向前方扇形区域发射多条测距射线。",
                focus_element_ids=["lidar"],
            ),
            TimelineStep(
                id="detect",
                title="命中障碍",
                narration="最近回波被标记为危险候选点。",
                focus_element_ids=["lidar", "obstacle-1"],
            ),
            TimelineStep(
                id="decide",
                title="阈值判断",
                narration="最近距离低于安全距离时，控制器进入避障状态。",
                focus_element_ids=["car", "obstacle-1"],
            ),
            TimelineStep(
                id="avoid",
                title="转向绕行",
                narration="控制器比较左右空旷程度，选择更安全的一侧。",
                focus_element_ids=["car"],
            ),
        ],
        controls=[
            SliderControl(
                id="speed",
                label="车速",
                min_value=0.4,
                max_value=2.2,
                default_value=1.0,
                step=0.1,
                target_property="car.speed",
            ),
            SliderControl(
                id="lidar-radius",
                label="雷达半径",
                min_value=90,
                max_value=210,
                default_value=150,
                step=5,
                target_property="lidar.radius",
            ),
            SliderControl(
                id="safe-distance",
                label="安全距离",
                min_value=45,
                max_value=120,
                default_value=76,
                step=2,
                target_property="scene.safe_distance",
            ),
            ButtonControl(
                id="reset",
                label="重置",
                target_property="scene",
                action="reset_scene",
            ),
        ],
        metadata={"template": "robot_obstacle_avoidance", "renderer": "canvas_2d"},
    )


def build_ros_pub_sub_scene() -> Scene:
    """Create a reusable scene spec for ROS Publisher/Subscriber manuals."""

    return Scene(
        title="ROS Publisher / Topic / Subscriber 交互课件",
        elements=[
            RosNode(
                id="talker-node",
                x=150,
                y=230,
                label="Publisher",
                package="demo_nodes",
                executable="talker.py",
                role="publisher",
                source_ref="source.publisher",
            ),
            RosTopic(
                id="cmd-topic",
                x=390,
                y=230,
                name="/cmd_vel",
                label="/cmd_vel",
                from_node_id="talker-node",
                to_node_id="base-node",
                message_type="geometry_msgs/Twist",
                source_ref="source.publisher",
            ),
            MessagePacket(
                id="twist-message",
                x=220,
                y=230,
                topic_id="cmd-topic",
                payload_label="Twist(linear.x=0.4)",
            ),
            RosNode(
                id="base-node",
                x=650,
                y=230,
                label="Subscriber",
                package="robot_base",
                executable="base_controller.py",
                role="subscriber",
                source_ref="source.subscriber",
            ),
            RobotCar(id="robot", x=420, y=430, label="Robot Runtime"),
        ],
        timeline=[
            TimelineStep(
                id="graph",
                title="生成 ROS 知识图谱",
                narration="手册中的节点、Topic 和消息类型被抽取成可点击拓扑。",
                focus_element_ids=["talker-node", "cmd-topic", "base-node"],
            ),
            TimelineStep(
                id="publish",
                title="Publisher 发布消息",
                narration="发布者把 Twist 消息写入 /cmd_vel Topic。",
                focus_element_ids=["talker-node", "twist-message"],
            ),
            TimelineStep(
                id="receive",
                title="Subscriber 接收消息",
                narration="订阅者收到消息后触发回调，驱动机器人状态变化。",
                focus_element_ids=["base-node", "robot"],
            ),
            TimelineStep(
                id="inspect-source",
                title="源码关联",
                narration="点击 Topic 或源码片段，可以反向高亮对应 ROS 节点。",
                focus_element_ids=["cmd-topic", "talker-node", "base-node"],
            ),
        ],
        controls=[
            ButtonControl(
                id="next-shot",
                label="推进镜头",
                target_property="timeline",
                action="advance_timeline",
            )
        ],
        metadata={"template": "ros_pub_sub", "renderer": "canvas_2d"},
    )
