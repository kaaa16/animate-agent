# ROS Publisher 与 Subscriber 手册节选

ROS 中的节点通过 Topic 交换消息。发布者节点负责向指定 Topic 发布消息，订阅者节点监听同一个 Topic 并在收到消息时执行回调函数。

示例：

```python
pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
sub = rospy.Subscriber("/scan", LaserScan, scan_callback)
```

当控制节点发布 `/cmd_vel` 消息时，底盘节点接收速度指令并驱动机器人移动。当激光雷达节点发布 `/scan` 消息时，避障节点读取距离数组并判断是否需要减速或转向。

