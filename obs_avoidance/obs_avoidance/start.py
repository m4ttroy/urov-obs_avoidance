#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TwistStamped


class InitialPositionController(Node):
    def __init__(self):
        super().__init__('initial_position_controller')

        # Target position in the local frame
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = -5.0

        # Velocity approach parameters
        self.kp_xy = 0.9
        self.kp_z = 0.7
        self.max_speed_xy = 3.0
        self.max_speed_z = 1.0

        # Switch to position setpoints inside this 3D distance
        self.switch_distance = 1.0  # metres

        # Pose feedback
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.pose_received = False

        # Modes: "velocity" -> "position"
        self.mode = 'velocity'

        self.pose_subscriber    = self.create_subscription(PoseStamped,'/model/bluerov2_ping/pose',self.pose_callback,qos_profile_sensor_data)
        self.velocity_publisher = self.create_publisher(TwistStamped,'/mavros/setpoint_velocity/cmd_vel',30)
        self.position_publisher = self.create_publisher(PoseStamped,'/mavros/setpoint_position/local',1)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.get_logger().info(
            'Initial position controller started. '
            'Target = (0, 0, -5) m'
        )

    def pose_callback(self, msg: PoseStamped):
        self.current_x = msg.pose.position.x
        self.current_y = msg.pose.position.y
        self.current_z = msg.pose.position.z
        self.pose_received = True

    @staticmethod
    def clamp(value, limit):
        return max(-limit, min(limit, value))

    def control_loop(self):
        if not self.pose_received:
            self.get_logger().warning(
                'Waiting for pose feedback...',
                throttle_duration_sec=2.0
            )
            return

        ex = self.target_x - self.current_x
        ey = self.target_y - self.current_y
        ez = self.target_z - self.current_z

        distance = math.sqrt(ex**2 + ey**2 + ez**2)

        # Switch once when close enough
        if self.mode == 'velocity' and distance <= self.switch_distance:
            self.mode = 'position'
            self.get_logger().info(
                f'Switching to POSITION mode at distance '
                f'{distance:.2f} m'
            )

        if self.mode == 'velocity':
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.twist.linear.x = self.clamp(
                self.kp_xy * ex, self.max_speed_xy
            )
            msg.twist.linear.y = self.clamp(
                self.kp_xy * ey, self.max_speed_xy
            )
            msg.twist.linear.z = self.clamp(
                self.kp_z * ez, self.max_speed_z
            )

            self.velocity_publisher.publish(msg)

        else:
            msg = PoseStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'map'

            msg.pose.position.x = self.target_x
            msg.pose.position.y = self.target_y
            msg.pose.position.z = self.target_z

            # Hold the current yaw-neutral orientation
            msg.pose.orientation.x = 0.0
            msg.pose.orientation.y = 0.0
            msg.pose.orientation.z = 0.0
            msg.pose.orientation.w = 1.0

            self.position_publisher.publish(msg)

        self.get_logger().info(
            f'Mode={self.mode} | '
            f'Position=({self.current_x:.2f}, '
            f'{self.current_y:.2f}, {self.current_z:.2f}) | '
            f'Error=({ex:.2f}, {ey:.2f}, {ez:.2f}) | '
            f'Distance={distance:.2f} m',
            throttle_duration_sec=1.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = InitialPositionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()