#!/usr/bin/env python3

import math
import time
import casadi as ca
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import PositionTarget
from std_msgs.msg import Float32MultiArray, Header
from visualization_msgs.msg import Marker, MarkerArray

from .parameter import *


class NMPCController:
    def __init__(self, horizon=50, dt=0.04, max_speed=1.00, max_obstacles=0):
        self.N = int(horizon)
        self.dt = float(dt)
        max_speed = abs(float(max_speed))
        self.max_obstacles = int(max_obstacles)

        self.tau_u = 0.5
        self.tau_v = 0.5
        self.tau_w = 0.5

        self.opti = ca.Opti()
        self.X = self.opti.variable(7, self.N + 1)  # x, y, z, u_body, v_body, w_body, yaw
        self.U = self.opti.variable(3, self.N)      # commanded body-frame velocities

        self.p_init = self.opti.parameter(7, 1)
        self.p_target = self.opti.parameter(3, 1)
        self.p_obs = self.opti.parameter(2, self.max_obstacles)
        self.p_obs_active = self.opti.parameter(self.max_obstacles, 1)

        self._last_X = None
        self._last_U = None
        self.last_error = None
        self.last_solve_ms = 0.0
        self._setup_nmpc_problem()

    def _setup_nmpc_problem(self):
        opti, X, U = self.opti, self.X, self.U
        opti.subject_to(X[:, 0] == self.p_init)

        Q_pos = diag([5.0, 10.0, 15.0])
        R_ctrl = diag([0.30, 0.60, 0.20])
        R_smooth = diag([0.25, 0.30, 0.35])
        Q_terminal_pos = diag([40.0, 50.0, 24.0])
        Q_terminal_vel = diag([10.0, 10.0, 8.0])

        cost = 0.0
        for k in range(self.N):
            x_k, y_k, z_k = X[0, k], X[1, k], X[2, k]
            u_k, v_k, w_k = X[3, k], X[4, k], X[5, k]
            psi_k = X[6, k]
            u_cmd, v_cmd, w_cmd = U[0, k], U[1, k], U[2, k]

            pos_err = X[0:3, k] - self.p_target
            cost += ca.mtimes([pos_err.T, Q_pos, pos_err])
            ctrl = U[:, k]
            cost += ca.mtimes([ctrl.T, R_ctrl, ctrl])
            if k > 0:
                ctrl_rate = U[:, k] - U[:, k - 1]
                cost += ca.mtimes([ctrl_rate.T, R_smooth, ctrl_rate])

            safe_radius = 0.30
            for j in range(self.max_obstacles):
                obs_dist_sq = (x_k - self.p_obs[0, j])**2 + (y_k - self.p_obs[1, j])**2
                barrier_penalty = self.p_obs_active[j] * ca.if_else(
                    obs_dist_sq < safe_radius**2, 
                    250.0 * (safe_radius**2 - obs_dist_sq)**2, 
                    0.0
                )
                cost += barrier_penalty

            x_next = x_k + (u_k * ca.cos(psi_k) - v_k * ca.sin(psi_k)) * self.dt
            y_next = y_k + (u_k * ca.sin(psi_k) + v_k * ca.cos(psi_k)) * self.dt
            z_next = z_k + w_k * self.dt
            u_next = u_k + (self.dt / self.tau_u) * (u_cmd - u_k)
            v_next = v_k + (self.dt / self.tau_v) * (v_cmd - v_k)
            w_next = w_k + (self.dt / self.tau_w) * (w_cmd - w_k)
            psi_next = psi_k

            opti.subject_to(X[:, k + 1] == ca.vertcat(
                x_next, y_next, z_next, u_next, v_next, w_next, psi_next
            ))

        terminal_pos_err = X[0:3, self.N] - self.p_target
        terminal_vel = X[3:6, self.N]
        cost += ca.mtimes([terminal_pos_err.T, Q_terminal_pos, terminal_pos_err])
        cost += ca.mtimes([terminal_vel.T, Q_terminal_vel, terminal_vel])
        opti.minimize(cost)

        opti.subject_to(opti.bounded(-max_speed, U[0, :], max_speed))
        opti.subject_to(opti.bounded(-max_speed, U[1, :], max_speed))
        opti.subject_to(opti.bounded(-0.25, U[2, :], 0.25))

        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': 60,
            'ipopt.acceptable_tol': 1e-3,
        }
        opti.solver('ipopt', opts)

    def solve(self, init_state, target_pos, obs_positions, obs_active=None):
        self.opti.set_value(self.p_init, init_state)
        self.opti.set_value(self.p_target, target_pos)
        obs_array = np.zeros((2, self.max_obstacles), dtype=float)
        active_array = np.zeros((self.max_obstacles, 1), dtype=float)
        if obs_positions is not None:
            try:
                raw = list(obs_positions)
            except TypeError:
                raw = []
            if len(raw) == 2 and all(np.isscalar(v) for v in raw):
                raw = [raw]
            for j, point in enumerate(raw[:self.max_obstacles]):
                try:
                    ox, oy = float(point[0]), float(point[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if math.isfinite(ox) and math.isfinite(oy):
                    obs_array[:, j] = [ox, oy]
                    active_array[j, 0] = 1.0
        if obs_active is not None and not np.any(active_array):
            active_array[0, 0] = float(obs_active)
        self.opti.set_value(self.p_obs, obs_array)
        self.opti.set_value(self.p_obs_active, active_array)

        init_array = np.asarray(init_state, dtype=float).reshape(7)
        if self._last_X is not None and self._last_U is not None:
            x_guess = np.hstack((self._last_X[:, 1:], self._last_X[:, -1:]))
            x_guess[:, 0] = init_array
            u_guess = np.hstack((self._last_U[:, 1:], self._last_U[:, -1:]))
        else:
            x_guess = np.repeat(init_array.reshape(7, 1), self.N + 1, axis=1)
            u_guess = np.zeros((3, self.N), dtype=float)
        self.opti.set_initial(self.X, x_guess)
        self.opti.set_initial(self.U, u_guess)

        started = time.monotonic()
        self.last_error = None
        try:
            sol = self.opti.solve()
            x_solution = np.asarray(sol.value(self.X), dtype=float)
            u_solution = np.asarray(sol.value(self.U), dtype=float)
            self._last_X = x_solution
            self._last_U = u_solution
            first_u = [float(v) for v in u_solution[:, 0]]
            return first_u, True
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            self._last_X = None
            self._last_U = None
            return [0.0, 0.0, 0.0], False
        finally:
            self.last_solve_ms = (time.monotonic() - started) * 1000.0

    def warmup(self, init_state):
        _, ok = self.solve(init_state, list(init_state[:3]), [], 0.0)
        self._last_X = None
        self._last_U = None
        return ok


def diag(lst):
    return ca.diag(ca.vertcat(*lst))


class ObstacleAvoidanceController:
    def __init__(self, pool_bounds=None, wall_margin=0.50):
        self.barrier_radius = 0.30
        self.sonar_ranges = []
        self.cluster_gap = 2.5
        self.cluster_corridor = 1.45
        max_speed = 1.00
        self.braking_margin = 0.65
        self.braking_accel = 1.50
        self.pre_clearance_surge_cap = 0.22
        self.side_clearance_tolerance = 0.10
        self.endpoint_wall_tolerance = 0.50
        self.pool_bounds = tuple(map(float, pool_bounds)) if pool_bounds is not None else None
        self.wall_margin = float(wall_margin)
        self.last_detection_reason = 'no_scan'

        self.is_bypassing = False
        self.locked_obstacle_x = 0.0
        self.locked_obstacle_y = 0.0
        self.locked_path_ux = 1.0
        self.locked_path_uy = 0.0
        self.locked_normal_x = 0.0
        self.locked_normal_y = 1.0
        self.locked_bypass_sign = 1.0
        self.locked_bypass_offset = total_bypass_offset
        self.locked_obstacles = []
        self.pass_end_progress = 0.8
        self.pass_length = 0.8
        self.return_length = 0.9

    @staticmethod
    def _unique_points(points, merge_distance=0.25):
        out = []
        for point in points:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if not any(math.hypot(x - q[0], y - q[1]) < merge_distance for q in out):
                out.append((x, y))
        return out

    def update_sonar(self, ranges, angle_min=None, angle_increment=None):
        if len(ranges) != 5:
            self.sonar_ranges = []
            beam_angles = []
            return

        def sanitize(value):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return float('inf')
            if not math.isfinite(value) or value <= 0.0:
                return float('inf')
            return value

        if angle_min is not None and angle_increment is not None:
            angles = [float(angle_min) + i * float(angle_increment) for i in range(5)]
        else:
            angles = [0.148, 0.074, 0.0, -0.074, -0.148]
        beams = sorted(
            [(angles[i], sanitize(ranges[i])) for i in range(5)],
            key=lambda beam: beam[0], reverse=True,
        )
        beam_angles = [angle for angle, _ in beams]
        self.sonar_ranges = [distance for _, distance in beams]

    @staticmethod
    def _basis(path_dx, path_dy, yaw):
        norm = math.hypot(path_dx, path_dy)
        if norm < 1e-6:
            ux, uy = math.cos(yaw), math.sin(yaw)
        else:
            ux, uy = path_dx / norm, path_dy / norm
        return ux, uy, -uy, ux

    @staticmethod
    def _projection(point, line_x, line_y, ux, uy, nx, ny):
        dx, dy = point[0] - line_x, point[1] - line_y
        return dx * ux + dy * uy, dx * nx + dy * ny

    def _lane_is_inside_bounds(self, point):
        if self.pool_bounds is None:
            return True
        min_x, max_x, min_y, max_y = self.pool_bounds
        m = self.wall_margin
        return min_x + m <= point[0] <= max_x - m and min_y + m <= point[1] <= max_y - m

    def _choose_bypass(self, cluster, target_x, target_y, ux, uy, nx, ny, preferred_sign):
        candidate_data = []
        for sign in (preferred_sign, -preferred_sign):
            cross_values = [
                (point[0] - target_x) * nx + (point[1] - target_y) * ny
                for point in cluster
            ]
            required_offset = max(
                total_bypass_offset,
                max((sign * cross for cross in cross_values), default=0.0)
                + self.barrier_radius + 0.08,
            )
            min_wall_clearance = float('inf')
            inside = True
            for point in cluster:
                along = (point[0] - target_x) * ux + (point[1] - target_y) * uy
                lane_point = (
                    target_x + along * ux + sign * required_offset * nx,
                    target_y + along * uy + sign * required_offset * ny,
                )
                if not self._lane_is_inside_bounds(lane_point):
                    inside = False
                    break
                if self.pool_bounds is not None:
                    min_x, max_x, min_y, max_y = self.pool_bounds
                    wall_clearance = min(
                        lane_point[0] - min_x, max_x - lane_point[0],
                        lane_point[1] - min_y, max_y - lane_point[1],
                    )
                    min_wall_clearance = min(min_wall_clearance, wall_clearance)
            if inside:
                min_obj_clearance = min(
                    abs(sign * required_offset - cross) for cross in cross_values
                )
                candidate_data.append((sign, required_offset, min_wall_clearance, min_obj_clearance))

        if not candidate_data:
            self.last_detection_reason = 'no_fully_feasible_bypass_side'
            return None, None

        return max(candidate_data, key=lambda row: (row[2], row[3], row[0] == preferred_sign))[:2]

    def _scan_obstacle(self, x_rov, y_rov, yaw, i_min, r_min):
        if i_min is None or not math.isfinite(r_min):
            return None
        theta = beam_angles[i_min] if len(beam_angles) == 5 else 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        sx = x_rov + sensor_offset_x * c - sensor_offset_y * s
        sy = y_rov + sensor_offset_x * s + sensor_offset_y * c
        return (sx + r_min * math.cos(yaw + theta),
                sy + r_min * math.sin(yaw + theta))

    def compute_nu_det(self, x_rov, y_rov, yaw, target_x, target_y,
                       allow_lateral=True, path_dx=1.0, path_dy=0.0):
        ux, uy, nx, ny = self._basis(path_dx, path_dy, yaw)
        result = {
            'u_det': 0.0, 'surge_limit': max_speed, 'braking_active': False,
            'x_ref': target_x, 'y_ref': target_y, 'e_det': 0.0, 'y_det': 0,
            'i_min': None, 'obstacle_xy': (0.0, 0.0), 'obstacles_xy': [],
            'obs_active': 0.0, 'blocked': False, 'reason': 'no_relevant_obstacle',
        }

        ranges = self.sonar_ranges if len(self.sonar_ranges) == 5 else []
        finite_forward = [(i, ranges[i]) for i in (1, 2, 3)
                          if len(ranges) == 5 and math.isfinite(ranges[i])]
        if finite_forward:
            i_min, r_min = min(finite_forward, key=lambda item: item[1])
        else:
            i_min, r_min = None, float('inf')
        result['i_min'] = i_min

        c, s = math.cos(yaw), math.sin(yaw)
        sx = x_rov + sensor_offset_x * c - sensor_offset_y * s
        sy = y_rov + sensor_offset_x * s + sensor_offset_y * c
        expected_range = max(0.0, (target_x - sx) * c + (target_y - sy) * s)
        endpoint_wall_return = (
            math.isfinite(r_min) and 0.0 < expected_range <= detection_range + self.endpoint_wall_tolerance
            and r_min >= max(0.0, expected_range - self.endpoint_wall_tolerance)
        )
        sonar_point = None
        sonar_valid = (
            math.isfinite(r_min) and r_min <= detection_range
            and not endpoint_wall_return
            and (expected_range - r_min + sonar_wall_offset > 0.5)
        )
        if sonar_valid:
            sonar_point = self._scan_obstacle(x_rov, y_rov, yaw, i_min, r_min)

        sonar_added = False
        if sonar_point is not None:
            static_near = False
            if self.is_bypassing:
                distinct = all(math.hypot(sonar_point[0] - p[0], sonar_point[1] - p[1]) >= 0.50
                               for p in self.locked_obstacles)
                along_from_first = ((sonar_point[0] - self.locked_obstacle_x) * self.locked_path_ux
                                    + (sonar_point[1] - self.locked_obstacle_y) * self.locked_path_uy)
                _, cross = self._projection(sonar_point, target_x, target_y, ux, uy, nx, ny)
                if distinct and not static_near and along_from_first >= -0.3 and abs(cross) <= self.cluster_corridor:
                    self.locked_obstacles.append(sonar_point)
                    self.locked_obstacles = self._unique_points(self.locked_obstacles, 0.45)
                    self.pass_end_progress = max(self.pass_end_progress, along_from_first + self.pass_length)
                    sonar_added = True
            elif not static_near:
                sonar_added = True

        if not self.is_bypassing and allow_lateral:
            candidate_points = []
            if sonar_point is not None and sonar_added:
                candidate_points.append(sonar_point)
            candidate_points = self._unique_points(candidate_points, 0.45)
            candidate_points.sort(key=lambda p: (p[0] - x_rov) * ux + (p[1] - y_rov) * uy)
            if candidate_points:
                first = candidate_points[0]
                cluster = []
                for point in candidate_points:
                    s_rel = (point[0] - first[0]) * ux + (point[1] - first[1]) * uy
                    _, cross = self._projection(point, target_x, target_y, ux, uy, nx, ny)
                    if -0.35 <= s_rel <= self.cluster_gap and abs(cross) <= self.cluster_corridor:
                        cluster.append(point)
                cluster = self._unique_points(cluster, 0.45)
                if not cluster:
                    cluster = [first]

                _, first_cross = self._projection(first, target_x, target_y, ux, uy, nx, ny)
                if abs(first_cross) > 0.08:
                    preferred_sign = -1.0 if first_cross > 0.0 else 1.0
                elif i_min == 1:
                    preferred_sign = -1.0
                elif i_min == 3:
                    preferred_sign = 1.0
                else:
                    left_range = ranges[0] if len(ranges) == 5 else float('inf')
                    right_range = ranges[4] if len(ranges) == 5 else float('inf')
                    if math.isfinite(left_range) and not math.isfinite(right_range):
                        preferred_sign = -1.0
                    elif math.isfinite(right_range) and not math.isfinite(left_range):
                        preferred_sign = 1.0
                    elif math.isfinite(left_range) and math.isfinite(right_range):
                        preferred_sign = 1.0 if right_range > left_range else -1.0
                    else:
                        preferred_sign = 1.0

                side, offset = self._choose_bypass(
                    cluster, target_x, target_y, ux, uy, nx, ny, preferred_sign
                )
                if side is None or offset is None:
                    result.update({
                        'surge_limit': 0.0,
                        'braking_active': True,
                        'obstacle_xy': first,
                        'obstacles_xy': cluster[:8],
                        'obs_active': 1.0,
                        'blocked': True,
                        'reason': 'no_safe_bypass_lane_stop',
                    })
                    self.last_detection_reason = result['reason']
                    return result
                ordered = sorted(cluster, key=lambda p: (p[0] - first[0]) * ux + (p[1] - first[1]) * uy)
                self.locked_obstacle_x, self.locked_obstacle_y = ordered[0]
                self.locked_path_ux, self.locked_path_uy = ux, uy
                self.locked_normal_x, self.locked_normal_y = nx, ny
                self.locked_bypass_sign = side
                self.locked_bypass_offset = offset
                self.locked_obstacles = ordered
                self.pass_end_progress = max(
                    self.pass_length,
                    max((p[0] - self.locked_obstacle_x) * ux + (p[1] - self.locked_obstacle_y) * uy
                        for p in ordered) + self.pass_length,
                )
                self.is_bypassing = True
                result['reason'] = 'sonar_cluster_bypass_started'

        if self.is_bypassing:
            ux, uy = self.locked_path_ux, self.locked_path_uy
            nx, ny = self.locked_normal_x, self.locked_normal_y
            progress = ((x_rov - self.locked_obstacle_x) * ux
                        + (y_rov - self.locked_obstacle_y) * uy)
            if progress > self.pass_end_progress + self.return_length:
                self.is_bypassing = False
                self.locked_obstacles = []
                result['reason'] = 'bypass_complete'
                self.last_detection_reason = result['reason']
                return result
            weight = 1.0 if progress <= self.pass_end_progress else max(0.0, 1.0 - (progress - self.pass_end_progress) / self.return_length)

            side = self.locked_bypass_sign
            offset = self.locked_bypass_offset
            result['x_ref'] = target_x + side * nx * offset * weight
            result['y_ref'] = target_y + side * ny * offset * weight

            clearances = [
                side * ((x_rov - p[0]) * nx + (y_rov - p[1]) * ny)
                for p in self.locked_obstacles
            ]
            lateral_clearance = min(clearances) if clearances else 0.0
            required_clearance = max(0.0, offset - self.side_clearance_tolerance)

            ahead_gaps = [((p[0] - x_rov) * ux + (p[1] - y_rov) * uy) for p in self.locked_obstacles]
            positive_gaps = [gap for gap in ahead_gaps if gap > 0.0]
            nearest_gap = min(positive_gaps) if positive_gaps else float('inf')
            if lateral_clearance < required_clearance:
                surge_limit = self.pre_clearance_surge_cap
                if math.isfinite(nearest_gap):
                    available = max(0.0, nearest_gap - self.braking_margin)
                    range_speed_limit = math.sqrt(2.0 * self.braking_accel * available)
                    surge_limit = min(surge_limit, range_speed_limit)
                braking_active = True
            else:
                surge_limit = max_speed
                braking_active = False

            result.update({
                'surge_limit': surge_limit,
                'braking_active': braking_active,
                'e_det': max(0.0, detection_range - r_min) if math.isfinite(r_min) else 0.0,
                'y_det': int(-side),
                'obstacle_xy': (self.locked_obstacle_x, self.locked_obstacle_y),
                'obstacles_xy': self._unique_points(self.locked_obstacles, 0.20),
                'obs_active': 1.0,
                'reason': result['reason'] if result['reason'] in {'sonar_cluster_bypass_started'} else 'bypass_active',
            })
        else:
            result['reason'] = 'no_relevant_obstacle'

        self.last_detection_reason = result['reason']
        return result


class SurveyPathFollower(Node):
    def __init__(self):
        super().__init__('survey_path_follower_nmpc')

        dt_ctrl = 1.0 / control_rate
        self.nmpc = NMPCController(horizon=25, dt=dt_ctrl, max_speed=max_speed, max_obstacles=8)
        warmup_state = [0.0, 0.0, target_depth, 0.0, 0.0, 0.0, 0.0]
        if not self.nmpc.warmup(warmup_state):
            self.get_logger().warning(f'NMPC solver warm-up failed: {self.nmpc.last_error or "unknown error"}')

        self.obstacle_controller = ObstacleAvoidanceController()
        self.obstacle_controller.max_speed = max_speed
        self.detection_enabled = False

        # ---------------- Subscribers ----------------
        self.sub_pose  = self.create_subscription(PoseStamped, "/model/bluerov2_ping/pose", self.pose_callback, qos_profile_sensor_data)
        self.sub_odom  = self.create_subscription(Odometry, "/model/bluerov2_ping/odometry", self.odom_callback, qos_profile_sensor_data)
        self.sub_sonar = self.create_subscription(LaserScan, "/ping_link/sensor/sonar", self.sonar_callback, qos_profile_sensor_data)

        # ---------------- Publishers ----------------
        self.pub_setpoint     = self.create_publisher(PositionTarget, "/mavros/setpoint_raw/local", 30)
        self.pub_depth        = self.create_publisher(Float32MultiArray, '/analysis/depth_tracking', 30)
        self.pub_velocity     = self.create_publisher(Float32MultiArray, '/analysis/velocity_tracking', 30)
        self.pub_error        = self.create_publisher(Float32MultiArray, '/analysis/errors', 30)
        self.pub_object_recon = self.create_publisher(MarkerArray, '/analysis/object_reconstruction', 30)
        self.pub_solve_time   = self.create_publisher(Float32MultiArray, '/analysis/solve_time', 30)
        # -----------------------------------------------------

        self.timer = self.create_timer(dt_ctrl, self.control_loop)

        # State Variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.current_yaw = 0.0
        self.pose_received = False

        self.actual_vx = 0.0
        self.actual_vy = 0.0
        self.actual_vz = 0.0

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0

        self.phase = 'transit'
        self.current_wp_idx = 0
        self.align_started = None
        self.align_yaw = 0.0

        self.settle_target_idx = 0
        self.settle_return_phase = 'transit'
        self.settle_yaw = 0.0
        self.settled_since = None
        self.last_solver_warning_time = 0.0

        self.last_control_time = time.monotonic()
        self.last_debug_time = time.monotonic()
        self.last_bypass_log_time = 0.0
        self.get_logger().info('BlueROV2 Obstacle-avoidance N-MPC Control.')

    def pose_callback(self, msg: PoseStamped):
        p = msg.pose.position
        q = msg.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_x = p.x
        self.current_y = p.y
        self.current_z = p.z
        self.pose_received = True

    def odom_callback(self, msg: Odometry):
        v = msg.twist.twist.linear
        child_frame = (msg.child_frame_id or '').strip('/').lower()
        if child_frame in {'map', 'odom', 'world', 'world_enu'}:
            c = math.cos(self.current_yaw)
            s = math.sin(self.current_yaw)
            self.actual_vx = v.x * c + v.y * s
            self.actual_vy = -v.x * s + v.y * c
        else:
            self.actual_vx = v.x
            self.actual_vy = v.y
        self.actual_vz = v.z

    def sonar_callback(self, msg: LaserScan):
        self.obstacle_controller.update_sonar(
            list(msg.ranges),
            angle_min=msg.angle_min,
            angle_increment=msg.angle_increment,
        )

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def reset_command_velocity(self):
        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_vz = 0.0

    def limit_velocity_change(self, vx, vy, vz, dt, accel_limit=None):
        dvx = vx - self.cmd_vx
        dvy = vy - self.cmd_vy
        dvz = vz - self.cmd_vz
        a = max_accel if accel_limit is None else max(0.0, float(accel_limit))
        max_delta = a * max(dt, 0.0)
        dvx = max(-max_delta, min(max_delta, dvx))
        dvy = max(-max_delta, min(max_delta, dvy))
        dvz = max(-max_delta, min(max_delta, dvz))
        self.cmd_vx += dvx
        self.cmd_vy += dvy
        self.cmd_vz += dvz
        return self.cmd_vx, self.cmd_vy, self.cmd_vz

    def compute_total_velocity(self, tx, ty, tz, dt, allow_lateral=False, path_dx=1.0, path_dy=0.0):
        if self.detection_enabled:
            detection = self.obstacle_controller.compute_nu_det(
                x_rov=self.current_x, y_rov=self.current_y, yaw=self.current_yaw,
                target_x=tx, target_y=ty, allow_lateral=allow_lateral,
                path_dx=path_dx, path_dy=path_dy,
            )
        else:
            detection = {
                "u_det": 0.0, "surge_limit": max_speed, "braking_active": False,
                "x_ref": tx, "y_ref": ty, "e_det": 0.0, "y_det": 0, "i_min": None,
                "obstacle_xy": (0.0, 0.0), "obstacles_xy": [], "obs_active": 0.0,
                "reason": "detection_disabled",
            }

        effective_tx = detection["x_ref"]
        effective_ty = detection["y_ref"]
        init_state = [
            self.current_x, self.current_y, self.current_z,
            self.actual_vx, self.actual_vy, self.actual_vz, self.current_yaw,
        ]
        target_pos = [effective_tx, effective_ty, tz]
        obs_positions = detection.get("obstacles_xy", [])
        if not obs_positions and detection.get("obs_active", 0.0) > 0.0:
            obs_xy_val = detection.get("obstacle_xy")
            if obs_xy_val is not None:
                obs_positions = [obs_xy_val]

        u_opt, solved = self.nmpc.solve(init_state, target_pos, obs_positions, detection["obs_active"])
        detection["solver_ok"] = solved

        solve_time_msg = Float32MultiArray()
        solve_time_msg.data = [float(self.nmpc.last_solve_ms), 1.0 if solved else 0.0]
        self.pub_solve_time.publish(solve_time_msg)

        if detection.get("blocked", False):
            vx, vy, vz = self.limit_velocity_change(0.0, 0.0, 0.0, dt, accel_limit=emergency_decel)
        elif solved:
            u_body = min(u_opt[0] + detection["u_det"], detection["surge_limit"])
            v_body = u_opt[1]
            w_body = u_opt[2]
            c = math.cos(self.current_yaw)
            s = math.sin(self.current_yaw)
            vx_world = u_body * c - v_body * s
            vy_world = u_body * s + v_body * c
            vz_world = w_body
            horizontal_speed = math.hypot(vx_world, vy_world)
            if horizontal_speed > max_speed > 0.0:
                scale = max_speed / horizontal_speed
                vx_world *= scale
                vy_world *= scale
            accel_limit = emergency_decel if detection.get("braking_active", False) else max_accel
            vx, vy, vz = self.limit_velocity_change(vx_world, vy_world, vz_world, dt, accel_limit=accel_limit)
        else:
            vx, vy, vz = self.limit_velocity_change(0.0, 0.0, 0.0, dt)

        distance = math.sqrt(
            (effective_tx - self.current_x)**2 + (effective_ty - self.current_y)**2 + (tz - self.current_z)**2
        )

        depth_msg = Float32MultiArray()
        depth_msg.data = [float(self.current_z), float(tz)]
        self.pub_depth.publish(depth_msg)

        vel_msg = Float32MultiArray()
        vel_msg.data = [float(vx), float(vy), float(vz), float(self.actual_vx), float(self.actual_vy), float(self.actual_vz)]
        self.pub_velocity.publish(vel_msg)

        if self.phase == 'move' and self.current_wp_idx < len(waypoints) - 1:
            x0, y0 = waypoints[self.current_wp_idx]
            x1, y1 = waypoints[self.current_wp_idx + 1]
            num = abs((y1 - y0)*self.current_x - (x1 - x0)*self.current_y + x1*y0 - y1*x0)
            den = math.hypot(y1 - y0, x1 - x0)
            cross_track_err = num / den if den > 1e-6 else 0.0
        else:
            cross_track_err = 0.0

        err_msg = Float32MultiArray()
        err_msg.data = [float(distance), float(cross_track_err), float(abs(self.current_z - tz))]
        self.pub_error.publish(err_msg)

        if obs_positions:
            marker_array = MarkerArray()
            for idx, obs in enumerate(obs_positions):
                marker = Marker()
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.header.frame_id = 'map'
                marker.ns = 'reconstructed_objects'
                marker.id = idx
                marker.type = Marker.CYLINDER
                marker.action = Marker.ADD
                marker.pose.position.x = float(obs[0])
                marker.pose.position.y = float(obs[1])
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

    def _new_setpoint(self, yaw):
        msg = PositionTarget()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        msg.yaw = float(self.wrap_angle(yaw))
        msg.yaw_rate = 0.0
        return msg

    def publish_velocity_setpoint(self, vx, vy, vz, yaw):
        msg = self._new_setpoint(yaw)
        msg.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)
        self.pub_setpoint.publish(msg)

    def publish_position_hold(self, px, py, pz, yaw):
        msg = self._new_setpoint(yaw)
        msg.type_mask = (
            PositionTarget.IGNORE_VX | PositionTarget.IGNORE_VY | PositionTarget.IGNORE_VZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW_RATE
        )
        msg.position.x = float(px)
        msg.position.y = float(py)
        msg.position.z = float(pz)
        self.pub_setpoint.publish(msg)

    def begin_settling(self, waypoint_index, return_phase, yaw_target):
        self.settle_target_idx = int(waypoint_index)
        self.settle_return_phase = return_phase
        self.settle_yaw = float(yaw_target)
        self.settled_since = None
        self.phase = 'settle'

    def actual_speed(self):
        return math.sqrt(self.actual_vx**2 + self.actual_vy**2 + self.actual_vz**2)

    def distance_to_waypoint(self, waypoint_index):
        tx, ty = waypoints[waypoint_index]
        return math.sqrt((tx - self.current_x)**2 + (ty - self.current_y)**2 + (target_depth - self.current_z)**2)

    def log_debug(self, tx, ty, tz, vx, vy, vz, distance, yaw_target):
        now = time.monotonic()
        if now - self.last_debug_time < debug_period:
            return
        self.last_debug_time = now
        forward_ranges = self.obstacle_controller.sonar_ranges[1:4]
        min_sonar_val = min(forward_ranges) if forward_ranges else float('inf')
        solve_status = 'ok' if self.nmpc.last_error is None else 'FAIL'
        self.get_logger().info(
            f'Phase={self.phase} | FrontSonar={min_sonar_val:.2f}m | '
            f'Obs={self.obstacle_controller.last_detection_reason} | '
            f'GT=({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | '
            f'VelENU=({vx:.2f}, {vy:.2f}, {vz:.2f}) | Dist={distance:.2f}m | '
            f'Solve={solve_status} ({self.nmpc.last_solve_ms:.1f}ms)'
        )

    def begin_alignment(self, waypoint_index):
        if waypoint_index >= len(waypoints) - 1:
            self.phase = 'complete'
            self.reset_command_velocity()
            return
        x0, y0 = waypoints[waypoint_index]
        x1, y1 = waypoints[waypoint_index + 1]
        self.align_yaw = math.atan2(y1 - y0, x1 - x0)
        self.align_started = time.monotonic()
        self.phase = 'align'
        self.reset_command_velocity()

    def control_loop(self):
        now = time.monotonic()
        dt = min(max(now - self.last_control_time, 0.0), 0.2)
        self.last_control_time = now

        if not self.pose_received:
            return

        if self.phase == 'transit':
            tx, ty = waypoints[0]
            tz = target_depth
            vx, vy, vz, distance, effective_ty, detection = self.compute_total_velocity(tx, ty, tz, dt, allow_lateral=False)
            self.publish_velocity_setpoint(vx, vy, vz, self.current_yaw)
            self.log_debug(tx, effective_ty, tz, vx, vy, vz, distance, self.current_yaw)
            if distance <= acceptance_radius:
                self.begin_settling(0, 'transit', self.current_yaw)
            return

        if self.phase == 'align':
            idx = self.current_wp_idx
            tx, ty = waypoints[idx]
            tz = target_depth
            self.publish_position_hold(tx, ty, tz, self.align_yaw)
            yaw_error = abs(self.wrap_angle(self.align_yaw - self.current_yaw))
            if (now - self.align_started) >= yaw_hold_seconds and yaw_error <= yaw_tolerance:
                self.phase = 'move'
            return

        if self.phase == 'move':
            idx = self.current_wp_idx
            tx, ty = waypoints[idx + 1]
            tz = target_depth
            x0, y0 = waypoints[idx]
            desired_yaw = math.atan2(ty - y0, tx - x0)
            path_dx = tx - x0
            path_dy = ty - y0
            vx, vy, vz, distance, effective_ty, detection = self.compute_total_velocity(
                tx, ty, tz, dt, allow_lateral=True, path_dx=path_dx, path_dy=path_dy,
            )
            self.publish_velocity_setpoint(vx, vy, vz, desired_yaw)
            waypoint_distance = self.distance_to_waypoint(idx + 1)
            self.log_debug(tx, effective_ty, tz, vx, vy, vz, waypoint_distance, desired_yaw)
            if waypoint_distance <= acceptance_radius and not self.obstacle_controller.is_bypassing:
                self.begin_settling(idx + 1, 'move', desired_yaw)
            return

        if self.phase == 'settle':
            target_idx = self.settle_target_idx
            tx, ty = waypoints[target_idx]
            tz = target_depth
            distance = self.distance_to_waypoint(target_idx)
            speed = self.actual_speed()
            vx, vy, vz = self.limit_velocity_change(0.0, 0.0, 0.0, dt)
            self.publish_velocity_setpoint(vx, vy, vz, self.settle_yaw)
            self.log_debug(tx, ty, tz, vx, vy, vz, distance, self.settle_yaw)

            if distance > acceptance_radius * 1.25:
                self.phase = self.settle_return_phase
                self.settled_since = None
                return

            if distance <= acceptance_radius and speed <= settle_speed:
                if self.settled_since is None:
                    self.settled_since = now
                elif now - self.settled_since >= settle_duration:
                    if self.settle_return_phase == 'transit':
                        self.detection_enabled = True
                        self.begin_alignment(0)
                    else:
                        self.current_wp_idx = target_idx
                        if self.current_wp_idx >= len(waypoints) - 1:
                            self.phase = 'complete'
                            self.reset_command_velocity()
                        else:
                            self.begin_alignment(self.current_wp_idx)
                    return
            else:
                self.settled_since = None
            return

        if self.phase == 'complete':
            tx, ty = waypoints[-1]
            tz = target_depth
            self.publish_position_hold(tx, ty, tz, self.current_yaw)


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