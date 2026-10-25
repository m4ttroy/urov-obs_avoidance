#!/usr/bin/env python3

import math
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan, Imu
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from .estimator import UKF_Estimator

class PIDController:
    def __init__(self, kp, ki, kd, output_limit, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.integral = 0.0

    def compute(self, error, actual_velocity, dt):
        if dt <= 0.0:
            return 0.0
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        i_term = self.ki * self.integral
        d_term = -self.kd * actual_velocity
        output = p_term + i_term + d_term
        return max(-self.output_limit, min(self.output_limit, output))

    def reset(self):
        self.integral = 0.0


class ObstacleAvoidanceController:
    def __init__(self):
        # Physical parameters
        self.obstacle_diameter = 0.25  # meters
        self.rov_width_margin = 0.17   # meters (half-width)
        self.safety_clearance = 1.0   # meters
        self.total_bypass_offset = self.rov_width_margin + (self.obstacle_diameter / 2.0) + self.safety_clearance  # ~0.60m

        self.sensor_offset_x = 0.16
        self.sensor_offset_y = 0.00

        # Sonar beam configuration (5 beams spanning ~17 degrees)
        self.beam_angles = [-0.148, -0.074, 0.0, 0.074, 0.148]  # radians
        self.sonar_ranges = []
        
        # Avoidance activation parameters
        self.detection_range = 2.5  # meters
        self.full_offset_distance = 1.0
        self.sonar_wall_offset = 0.40

        self.k_brake = 0.2
        self.max_brake = 0.50

    def update_sonar(self, ranges, angle_min=None, angle_increment=None):
        if len(ranges) != 5:
            self.sonar_ranges = []
            self.beam_angles = []
            return

        def sanitize(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return float("inf")

            if not math.isfinite(value) or value <= 0.0:
                return float("inf")

            return value

        if angle_min is not None and angle_increment is not None:
            angles = [
                float(angle_min) + i * float(angle_increment)
                for i in range(5)
            ]
        else:
            angles = [0.148, 0.074, 0.0, -0.074, -0.148]

        beams = sorted(
            [(angles[i], sanitize(ranges[i])) for i in range(5)],
            key=lambda beam: beam[0],
            reverse=True,
        )

        self.beam_angles = [angle for angle, _ in beams]
        self.sonar_ranges = [distance for _, distance in beams]

    def compute_detection_error(self, x_rov, target_x):
        return abs(target_x - x_rov)
    
    def compute_nu_det(self, x_rov, y_rov, yaw, target_x, target_y, y_ref_nominal, allow_lateral=True, path_x_sign=1):
        result = {
            "u_det": 0.0,
            "y_ref": y_ref_nominal,
            "e_det": 0.0,
            "y_det": 0,
            "i_min": None,
            "obstacle_xy": None,
        }

        if len(self.sonar_ranges) != 5:
            return result

        ranges = self.sonar_ranges
        r_min = min(ranges)

        if not math.isfinite(r_min) or r_min > self.detection_range:
            return result

        i_min = ranges.index(r_min)
        theta_min = self.beam_angles[i_min]

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        sensor_x = x_rov + self.sensor_offset_x * cos_yaw - self.sensor_offset_y * sin_yaw
        sensor_y = y_rov + self.sensor_offset_x * sin_yaw + self.sensor_offset_y * cos_yaw

        obstacle_x = sensor_x + r_min * math.cos(yaw + theta_min)
        obstacle_y = sensor_y + r_min * math.sin(yaw + theta_min)

        # weight = (self.detection_range - r_min) / (self.detection_range - self.full_offset_distance)
        weight = 1
        weight = max(0.0, min(1.0, weight))

        if i_min <= 1:
            y_det = +1  # Obstacle left -> shift right
        elif i_min >= 3:
            y_det = -1  # Obstacle right -> shift left
        else:
            left_clearance = min(ranges[0:2])
            right_clearance = min(ranges[3:5])
            y_det = +1 if right_clearance >= left_clearance else -1

        if allow_lateral:
            path_x_sign = 1 if path_x_sign >= 0 else -1
            y_ref = y_ref_nominal - (path_x_sign * y_det * self.total_bypass_offset * weight)
        else:
            y_ref = y_ref_nominal

        r_center = ranges[2]

        if math.isfinite(r_center) and 0.0 < r_center <= self.detection_range:
            expected_range = max(0.0, (target_x - sensor_x) * cos_yaw + (target_y - sensor_y) * sin_yaw)
            e_det = expected_range - r_center + self.sonar_wall_offset

            u_det = -self.k_brake * max(0.0, e_det) * weight
            u_det = max(-self.max_brake, min(0.0, u_det))
        else:
            e_det = 0.0
            u_det = 0.0

        result.update({
            "u_det": u_det,
            "y_ref": y_ref,
            "e_det": e_det,
            "y_det": y_det,
            "i_min": i_min,
            "obstacle_xy": (obstacle_x, obstacle_y),
        })

        return result


class SurveyPathFollower(Node):
    def __init__(self):
        super().__init__('survey_path_follower')

        # ---------------- Parameters ----------------
        self.declare_parameter('pose_topic', '/model/bluerov2_ping/pose')
        self.declare_parameter('pose_odom', '/model/bluerov2_ping/odometry')
        self.declare_parameter('imu_topic', '/world/bluerov2_ping/model/bluerov2_ping/link/base_link/sensor/imu_sensor/imu')
        self.declare_parameter('sonar_topic', '/ping_link/sensor/sonar')
        self.declare_parameter('setpoint_topic', '/mavros/setpoint_raw/local')
        self.declare_parameter('control_rate', 50.0)
        self.declare_parameter('target_depth', -5.0)

        self.declare_parameter('start_x', 0.0)
        self.declare_parameter('start_y', 0.0)

        self.declare_parameter('max_speed', 1.00)
        self.declare_parameter('max_accel', 1.50)
        self.declare_parameter('acceptance_radius', 0.20)

        self.declare_parameter('yaw_tolerance_deg', 5.00)
        self.declare_parameter('yaw_hold_seconds', 2.0)
        self.declare_parameter('debug_period', 1.0)

        self.declare_parameter('disturbance_model', False)

        self.pose_topic     = self.get_parameter('pose_topic').value
        self.odom_topic     = self.get_parameter('pose_odom').value
        self.imu_topic      = self.get_parameter('imu_topic').value
        self.sonar_topic    = self.get_parameter('sonar_topic').value
        self.setpoint_topic = self.get_parameter('setpoint_topic').value

        self.control_rate = float(self.get_parameter('control_rate').value)
        self.target_depth = float(self.get_parameter('target_depth').value)
        self.start_x      = float(self.get_parameter('start_x').value)
        self.start_y      = float(self.get_parameter('start_y').value)

        self.max_speed         = float(self.get_parameter('max_speed').value)
        self.max_accel         = float(self.get_parameter('max_accel').value)
        self.acceptance_radius = float(self.get_parameter('acceptance_radius').value)

        self.yaw_tolerance     = math.radians(float(self.get_parameter('yaw_tolerance_deg').value))
        self.yaw_hold_seconds  = float(self.get_parameter('yaw_hold_seconds').value)
        self.debug_period      = float(self.get_parameter('debug_period').value)
        self.disturbance_model = bool(self.get_parameter('disturbance_model').value)

        # Instantiate PID controllers for position outer loop
        self.pid_x = PIDController(kp=0.7, ki=0.04, kd=0.13, output_limit=self.max_speed, integral_limit=1.0)
        self.pid_y = PIDController(kp=0.9, ki=0.1, kd=0.2, output_limit=self.max_speed, integral_limit=1.0)
        self.pid_z = PIDController(kp=0.5, ki=0.03, kd=0.02, output_limit=self.max_speed, integral_limit=1.0)

        # Instantiate Continuous Obstacle Avoidance Component
        self.obstacle_controller = ObstacleAvoidanceController()
        self.ukf                 = UKF_Estimator(logger=self.get_logger())
        self.detection_enabled   = False

        # ROS Interfaces
        self.sub_pose     = self.create_subscription(PoseStamped, self.pose_topic, self.pose_callback, qos_profile_sensor_data)
        self.sub_odom     = self.create_subscription(Odometry, self.odom_topic, self.odom_callback, qos_profile_sensor_data)
        self.sub_imu      = self.create_subscription(Imu, self.imu_topic, self.imu_callback, qos_profile_sensor_data)
        self.sub_sonar    = self.create_subscription(LaserScan, self.sonar_topic, self.sonar_callback, qos_profile_sensor_data)
        self.pub_setpoint = self.create_publisher(PositionTarget, self.setpoint_topic, 30)

        # ---------------- Analysis Publishers ----------------
        self.pub_path_gt        = self.create_publisher(Path, '/analysis/gt_path', 30)
        self.pub_path_ref       = self.create_publisher(Path, '/analysis/ref_path', 30)
        self.pub_depth          = self.create_publisher(Float32MultiArray, '/analysis/depth_tracking', 30)
        self.pub_velocity       = self.create_publisher(Float32MultiArray, '/analysis/velocity_tracking', 30)
        self.pub_error          = self.create_publisher(Float32MultiArray, '/analysis/errors', 30)
        self.pub_object_recon   = self.create_publisher(MarkerArray, '/analysis/object_reconstruction', 30)
        # -----------------------------------------------------

        self.timer        = self.create_timer(1.0 / self.control_rate, self.control_loop)

        # State Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.current_q = [0.0, 0.0, 0.0, 1.0]
        self.pose_received = False

        self.actual_vx = 0.0
        self.actual_vy = 0.0
        self.actual_vz = 0.0

        self.latest_imu = None
        self.imu_received = False

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0

        # Waypoints for 8x8m pool grid
        self.waypoints = [
            (self.start_x, self.start_y),
            (8.0, 0.0), (8.0, 2.0), (0.0, 2.0),
            (0.0, 4.0), (8.0, 4.0), (8.0, 6.0),
            (0.0, 6.0), (0.0, 8.0), (8.0, 8.0),
        ]

        # Path buffer for visualization / analysis
        self.gt_path_msg = Path()
        self.gt_path_msg.header.frame_id = 'map'
        self.ref_path_msg = Path()
        self.ref_path_msg.header.frame_id = 'map'
        for wp in self.waypoints:
            ps = PoseStamped()
            ps.header.frame_id = 'map'
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            ps.pose.position.z = float(self.target_depth)
            ps.pose.orientation.w = 1.0
            self.ref_path_msg.poses.append(ps)

        self.phase = 'transit'
        self.current_wp_idx = 0
        self.align_started = None
        self.align_yaw = 0.0

        self.last_control_time = time.monotonic()
        self.last_debug_time = time.monotonic()
        self.get_logger().info('Continuous Model-Based SurveyPathFollower Initialized with Analysis Publishers.')

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation

        self.current_q = [q.x, q.y, q.z, q.w]
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_x = p.x
        self.current_y = p.y
        self.current_z = p.z
        self.pose_received = True

        # Append Ground Truth Path for Analysis
        stamp = self.get_clock().now().to_msg()
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = stamp
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose = msg.pose
        self.gt_path_msg.header.stamp = stamp
        self.pub_path_gt.publish(self.gt_path_msg)

        # Publish Reference Path periodically
        self.ref_path_msg.header.stamp = stamp
        self.pub_path_ref.publish(self.ref_path_msg)

    def odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear
        self.actual_vx = v.x
        self.actual_vy = v.y
        self.actual_vz = v.z

    def sonar_callback(self, msg: LaserScan):
        self.obstacle_controller.update_sonar(
            list(msg.ranges),
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
        )

    def imu_callback(self, msg: Imu):
        self.latest_imu = [
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ]
        self.imu_received = True

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset_command_velocity(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()

    def limit_velocity_change(self, vx, vy, vz, dt):
        dvx = vx - self.cmd_vx
        dvy = vy - self.cmd_vy
        dvz = vz - self.cmd_vz
        delta = math.sqrt(dvx**2 + dvy**2 + dvz**2)
        max_delta = self.max_accel * max(dt, 0.0)

        if delta > max_delta and delta > 1e-9:
            scale = max_delta / delta
            dvx *= scale
            dvy *= scale
            dvz *= scale

        self.cmd_vx += dvx
        self.cmd_vy += dvy
        self.cmd_vz += dvz
        return self.cmd_vx, self.cmd_vy, self.cmd_vz

    def compute_nu_cmd(self, tx, ty, tz, dt):
        ex = tx - self.current_x
        ey = ty - self.current_y
        ez = tz - self.current_z

        vx_cmd = self.pid_x.compute(ex, self.actual_vx, dt)
        vy_cmd = self.pid_y.compute(ey, self.actual_vy, dt)
        vz_cmd = self.pid_z.compute(ez, self.actual_vz, dt)
        return [vx_cmd, vy_cmd, vz_cmd]

    def compute_nu_det(self, tx, ty, allow_lateral=False, path_x_sign=1):
        if not self.detection_enabled:
            return {
                "u_det": 0.0,
                "y_ref": ty,
                "e_det": 0.0,
                "y_det": 0,
                "i_min": None,
                "obstacle_xy": None,
            }

        return self.obstacle_controller.compute_nu_det(
            x_rov=self.current_x,
            y_rov=self.current_y,
            yaw=self.current_yaw,
            target_x=tx,
            target_y=ty,
            y_ref_nominal=ty,
            allow_lateral=allow_lateral,
            path_x_sign=path_x_sign,
        )

    def compute_nu_dist(self):
        if self.disturbance_model:
            return [0.2, 0.1, -0.2]
        else:
            tau_dist = self.ukf.ukf_x[self.ukf.idx_tau_dist].reshape(6)
            scale = 0.005
            return [
                tau_dist[0] * scale,
                tau_dist[1] * scale,
                tau_dist[2] * scale
            ]

    def compute_total_velocity(self, tx, ty, tz, dt, allow_lateral=False, path_x_sign=1):
        detection = self.compute_nu_det(tx, ty, allow_lateral, path_x_sign)
        effective_ty = detection["y_ref"]

        nu_cmd = self.compute_nu_cmd(tx, effective_ty, tz, dt)
        nu_dist = self.compute_nu_dist()

        vx_total = nu_cmd[0] + detection["u_det"] + nu_dist[0]
        vy_total = nu_cmd[1] + nu_dist[1]
        vz_total = nu_cmd[2] + nu_dist[2]

        vx, vy, vz = self.limit_velocity_change(vx_total, vy_total, vz_total, dt)
        distance = math.sqrt((tx - self.current_x) ** 2 + (effective_ty - self.current_y) ** 2 + (tz - self.current_z) ** 2)

        # ---------------- Publish Analysis Metrics ----------------
        # 1. Depth Tracking: [current_z, target_z]
        depth_msg = Float32MultiArray()
        depth_msg.data = [float(self.current_z), float(tz)]
        self.pub_depth.publish(depth_msg)

        # 2. Velocity Tracking: [cmd_vx, cmd_vy, cmd_vz, actual_vx, actual_vy, actual_vz]
        vel_msg = Float32MultiArray()
        vel_msg.data = [float(vx), float(vy), float(vz), float(self.actual_vx), float(self.actual_vy), float(self.actual_vz)]
        self.pub_velocity.publish(vel_msg)

        # 3. Errors: [Euclidean distance, Cross-track error, Absolute depth error]
        if self.phase == 'move' and self.current_wp_idx < len(self.waypoints) - 1:
            x0, y0 = self.waypoints[self.current_wp_idx]
            x1, y1 = self.waypoints[self.current_wp_idx + 1]
            num = abs((y1 - y0)*self.current_x - (x1 - x0)*self.current_y + x1*y0 - y1*x0)
            den = math.hypot(y1 - y0, x1 - x0)
            cross_track_err = num / den if den > 1e-6 else 0.0
        else:
            cross_track_err = 0.0

        err_msg = Float32MultiArray()
        err_msg.data = [float(distance), float(cross_track_err), float(abs(self.current_z - tz))]
        self.pub_error.publish(err_msg)

        # 4. Object Reconstruction Markers (Cylinder with known diameter = 0.25m)
        obs_xy = detection.get("obstacle_xy")
        if obs_xy is not None:
            marker_array = MarkerArray()
            marker = Marker()
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.header.frame_id = 'map'
            marker.ns = 'reconstructed_objects'
            marker.id = 0
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = float(obs_xy[0])
            marker.pose.position.y = float(obs_xy[1])
            marker.pose.position.z = float(tz / 2.0)
            marker.pose.orientation.w = 1.0
            marker.scale.x = float(self.obstacle_controller.obstacle_diameter)
            marker.scale.y = float(self.obstacle_controller.obstacle_diameter)
            marker.scale.z = abs(float(tz))
            marker.color.r = 1.0
            marker.color.g = 0.5
            marker.color.b = 0.0
            marker.color.a = 0.8
            marker_array.markers.append(marker)
            self.pub_object_recon.publish(marker_array)
        # ----------------------------------------------------------

        return vx, vy, vz, distance, effective_ty, detection

    def publish_setpoint(self, px, py, pz, vx, vy, vz, yaw):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = (
            PositionTarget.IGNORE_AFX
            | PositionTarget.IGNORE_AFY
            | PositionTarget.IGNORE_AFZ
            | PositionTarget.IGNORE_YAW_RATE
        )
        msg.position.x = float(px)
        msg.position.y = float(py)
        msg.position.z = float(pz)
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        msg.yaw = float(self.wrap_angle(yaw))
        msg.yaw_rate = 0.0
        self.pub_setpoint.publish(msg)

    def log_debug(self, tx, ty, tz, vx, vy, vz, distance, yaw_target):
        now = time.monotonic()
        if now - self.last_debug_time < self.debug_period:
            return
        self.last_debug_time = now

        min_sonar_val = min(self.obstacle_controller.sonar_ranges) if self.obstacle_controller.sonar_ranges else float('inf')
        self.get_logger().info(
            f'Phase={self.phase} | MinSonar={min_sonar_val:.2f}m | '
            f'GT=({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | '
            f'CmdVel=({vx:.2f}, {vy:.2f}, {vz:.2f}) | Dist={distance:.2f}m'
        )

    def begin_alignment(self, waypoint_index):
        if waypoint_index >= len(self.waypoints) - 1:
            self.phase = 'complete'
            self.reset_command_velocity()
            return
        x0, y0 = self.waypoints[waypoint_index]
        x1, y1 = self.waypoints[waypoint_index + 1]
        self.align_yaw = math.atan2(y1 - y0, x1 - x0)
        self.align_started = time.monotonic()
        self.phase = 'align'
        self.reset_command_velocity()
        self.get_logger().info(f'Reached waypoint {waypoint_index}. Aligning heading.')

    def control_loop(self):
        now = time.monotonic()
        dt = min(max(now - self.last_control_time, 0.0), 0.2)
        self.last_control_time = now

        if not self.pose_received:
            self.get_logger().warning(f'Waiting for pose topic {self.pose_topic}', throttle_duration_sec=5.0)
            return
        
        # UKF Estimator Step
        nu_cmd_vector = [self.cmd_vx, self.cmd_vy, self.cmd_vz, 0.0, 0.0, 0.0]
        x_pred, P_pred, sigma_pred = self.ukf.ukf_predict(self.ukf.ukf_x, self.ukf.ukf_P, dt, nu_cmd_vector)
        self.ukf.ukf_x = x_pred
        self.ukf.ukf_P = P_pred

        if self.imu_received and self.latest_imu is not None:
            x_up, P_up = self.ukf.ukf_update_imu(self.ukf.ukf_x, self.ukf.ukf_P, sigma_pred, self.latest_imu)
            self.ukf.ukf_x = x_up
            self.ukf.ukf_P = P_up

        z_pose = [self.current_x, self.current_y, self.current_z, self.current_q[0], self.current_q[1], self.current_q[2], self.current_q[3]]
        z_vel = [self.actual_vx, self.actual_vy, self.actual_vz, 0.0, 0.0, 0.0]
        x_up, P_up = self.ukf.ukf_update_gazebo_gt(self.ukf.ukf_x, self.ukf.ukf_P, sigma_pred, z_pose, z_vel)
        self.ukf.ukf_x = x_up
        self.ukf.ukf_P = P_up

        if self.phase == 'transit':
            tx, ty = self.waypoints[0]
            tz = self.target_depth
            vx, vy, vz, distance, effective_ty, detection = self.compute_total_velocity(tx, ty, tz, dt, allow_lateral=False)
            self.publish_setpoint(tx, effective_ty, tz, vx, vy, vz, self.current_yaw)
            self.log_debug(tx, ty, tz, vx, vy, vz, distance, self.current_yaw)
            if distance <= self.acceptance_radius:
                self.detection_enabled = True
                self.get_logger().info('[SAFETY] Start waypoint reached; continuous obstacle detection enabled.')
                self.begin_alignment(0)
            return

        if self.phase == 'align':
            idx = self.current_wp_idx
            tx, ty = self.waypoints[idx]
            tz = self.target_depth
            self.publish_setpoint(tx, ty, tz, 0.0, 0.0, 0.0, self.align_yaw)
            distance = math.sqrt((tx - self.current_x)**2 + (ty - self.current_y)**2 + (tz - self.current_z)**2)
            yaw_error = abs(self.wrap_angle(self.align_yaw - self.current_yaw))
            if (now - self.align_started) >= self.yaw_hold_seconds and yaw_error <= self.yaw_tolerance:
                self.phase = 'move'
                self.get_logger().info(f'Heading aligned. Starting leg {idx} -> {idx + 1}.')
            return

        if self.phase == 'move':
            idx = self.current_wp_idx
            tx, ty = self.waypoints[idx + 1]
            tz = self.target_depth
            x0, y0 = self.waypoints[idx]
            desired_yaw = math.atan2(ty - y0, tx - x0)

            remaining_x = self.obstacle_controller.compute_detection_error(x_rov=self.current_x, target_x=tx)
            previous_state = self.detection_enabled

            if (not self.detection_enabled and remaining_x >= 4.0):
                self.detection_enabled = True
            elif (self.detection_enabled and remaining_x <= 2.75):
                self.detection_enabled = False

            horizontal_leg = abs(ty - y0) < 1e-6 and abs(tx - x0) > 1e-6
            path_x_sign = 1 if tx > x0 else -1
    
            vx, vy, vz, distance, effective_ty, detection = self.compute_total_velocity(tx, ty, tz, dt, allow_lateral=horizontal_leg, path_x_sign=path_x_sign)
            i_min       = detection["i_min"]
            obstacle_xy = detection["obstacle_xy"]
            y_det       = detection["y_det"]

            if i_min is not None and obstacle_xy is not None:
                decision = "RIGHT" if y_det == 1 else "LEFT"
                self.get_logger().info(
                    f"[OBSTACLE DETECTED] Beam={i_min} | Range={min(self.obstacle_controller.sonar_ranges):.2f} m | "
                    f"Obstacle: x={obstacle_xy[0]:.2f}, y={obstacle_xy[1]:.2f} m | Bypass={decision} | y_ref={effective_ty:.2f} m"
                )
            
            self.publish_setpoint(tx, effective_ty, tz, vx, vy, vz, desired_yaw)
            waypoint_distance = math.sqrt((tx - self.current_x) ** 2 + (ty - self.current_y) ** 2 + (tz - self.current_z) ** 2)
            self.log_debug(tx, ty, tz, vx, vy, vz, waypoint_distance, desired_yaw)

            if waypoint_distance <= self.acceptance_radius:
                self.current_wp_idx += 1
                if self.current_wp_idx >= len(self.waypoints) - 1:
                    self.phase = 'complete'
                    self.reset_command_velocity()
                    self.get_logger().info('Survey complete.')
                else:
                    self.begin_alignment(self.current_wp_idx)
            return

        if self.phase == 'complete':
            tx, ty = self.waypoints[-1]
            tz = self.target_depth
            self.publish_setpoint(tx, ty, tz, 0.0, 0.0, 0.0, self.current_yaw)


def main(args=None):
    rclpy.init(args=args)
    node = SurveyPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()