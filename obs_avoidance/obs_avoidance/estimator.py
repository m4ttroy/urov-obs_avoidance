import numpy as np
from .parameter import *

class UKF_Estimator:
    def __init__(self, logger=None):
        # State dimension: p_N(3) + q_N(4) + v_B(3) + w_B(3) + b_a(3) + b_g(3) + tau_dist(6) = 25
        self.ukf_state = 25
        self.logger = logger
        
        self.m = 10.0
        self.Ixx = 0.099
        self.Iyy = 0.178
        self.Izz = 0.269

        self.Dlin = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.Dquad = np.diag([33.732, 54.160, 73.225, 3.992, 3.992, 3.992])

        M = np.diag([self.m, self.m, self.m, self.Ixx, self.Iyy, self.Izz])
        self.Minv = np.linalg.inv(M)

        # Effective velocity-to-force tracking gains for ArduSub black-box bridge (Kp)
        self.Kp_vel = np.diag([150.0, 150.0, 150.0, 50.0, 50.0, 80.0])

        # Measurement Noise Covariances for Gazebo Sensors
        self.R_imu = np.diag([
            0.01**2, 0.01**2, 0.01**2,  # Linear Acceleration (m/s^2)
            0.001**2, 0.001**2, 0.001**2 # Angular Velocity (rad/s)
        ])
        self.R_gt_pose = np.diag([1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]) # Position (m) & Orientation rotvec
        self.R_gt_vel  = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]) # Linear & Angular velocity

        # State Slices
        self.idx_pN       = slice(0, 3)
        self.idx_qN       = slice(3, 7)
        self.idx_vB       = slice(7, 10)
        self.idx_wB       = slice(10, 13)
        self.idx_ba       = slice(13, 16)
        self.idx_bg       = slice(16, 19)
        self.idx_tau_dist = slice(19, 25)

        # UKF Tuning Parameters (Alpha, Beta, Kappa)
        self.alpha = 0.5
        self.beta = 2.0
        self.kappa = 0.0

        self.lambda_ = self.alpha**2 * (self.ukf_state + self.kappa) - self.ukf_state
        self.n_sigma = 2 * self.ukf_state + 1

        self.Wm = np.full(self.n_sigma, 1.0 / (2.0 * (self.ukf_state + self.lambda_)))
        self.Wc = np.full(self.n_sigma, 1.0 / (2.0 * (self.ukf_state + self.lambda_)))
        self.Wm[0] = self.lambda_ / (self.ukf_state + self.lambda_)
        self.Wc[0] = self.Wm[0] + (1.0 - self.alpha**2 + self.beta)

        self.ukf_x = np.zeros((self.ukf_state, 1), dtype=np.float64)
        self.ukf_P = np.eye(self.ukf_state, dtype=np.float64) * 1e-3

    # =========================================================
    # Quaternion Mathematical Utilities (Retained & Proven)
    # =========================================================
    @staticmethod
    def _normalize_quaternion(q):
        q = np.asarray(q, dtype=np.float64).reshape(4)
        n = np.linalg.norm(q)
        if not np.isfinite(n) or n < 1e-12:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return q / n

    @staticmethod
    def _quat_multiply(q1, q2):
        x1, y1, z1, w1 = np.asarray(q1, dtype=np.float64).reshape(4)
        x2, y2, z2, w2 = np.asarray(q2, dtype=np.float64).reshape(4)
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ], dtype=np.float64)

    @staticmethod
    def _quat_from_rotvec(rotvec):
        rotvec = np.asarray(rotvec, dtype=np.float64).reshape(3)
        theta = np.linalg.norm(rotvec)
        if theta < 1e-12:
            q = np.array([0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2], 1.0], dtype=np.float64)
            return q / np.linalg.norm(q)
        axis = rotvec / theta
        half = 0.5 * theta
        s = np.sin(half)
        return np.array([axis[0]*s, axis[1]*s, axis[2]*s, np.cos(half)], dtype=np.float64)

    @staticmethod
    def _quat_to_rotmat(q):
        x, y, z, w = UKF_Estimator._normalize_quaternion(q)
        return np.array([
            [1.0 - 2.0*(y*y + z*z), 2.0*(x*y - z*w),       2.0*(x*z + y*w)],
            [2.0*(x*y + z*w),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z - x*w)],
            [2.0*(x*z - y*w),       2.0*(y*z + x*w),       1.0 - 2.0*(x*x + y*y)]
        ], dtype=np.float64)

    @classmethod
    def _quat_error_rotvec(cls, q_ref, q_meas):
        q_ref = cls._normalize_quaternion(q_ref)
        q_meas = cls._normalize_quaternion(q_meas)
        q_err = cls._quat_multiply(np.array([-q_ref[0], -q_ref[1], -q_ref[2], q_ref[3]]), q_meas)
        v = q_err[:3]
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return 2.0 * v
        angle = 2.0 * np.arctan2(v_norm, np.clip(q_err[3], -1.0, 1.0))
        return v * (angle / v_norm)

    # =========================================================
    def _log_info(self, text):
        if self.logger is not None:
            self.logger.info(text)

    def reset(self):
        self.ukf_x = np.zeros((self.ukf_state, 1), dtype=np.float64)
        self.ukf_x[self.idx_qN] = np.array([0.0, 0.0, 0.0, 1.0]).reshape(4, 1)
        self.ukf_P = np.eye(self.ukf_state, dtype=np.float64) * 1e-2

    def build_Q(self, dt):
        Q = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)
        Q[self.idx_pN, self.idx_pN]             = np.eye(3) * 1e-5
        Q[self.idx_qN, self.idx_qN]             = np.eye(4) * 1e-5 * dt
        Q[self.idx_vB, self.idx_vB]             = np.eye(3) * 1e-4
        Q[self.idx_wB, self.idx_wB]             = np.eye(3) * 1e-4
        Q[self.idx_ba, self.idx_ba]             = np.eye(3) * 1e-6 * dt
        Q[self.idx_bg, self.idx_bg]             = np.eye(3) * 1e-6 * dt
        Q[self.idx_tau_dist, self.idx_tau_dist] = np.eye(6) * 1e-3 * dt  # Disturbance random walk process noise
        return Q

    def pack_state(self, p_N, q_N, v_B, w_B, b_a, b_g, tau_dist):
        state = np.zeros(self.ukf_state, dtype=np.float64)
        state[self.idx_pN]       = np.asarray(p_N, dtype=np.float64).reshape(3)
        state[self.idx_qN]       = self._normalize_quaternion(q_N)
        state[self.idx_vB]       = np.asarray(v_B, dtype=np.float64).reshape(3)
        state[self.idx_wB]       = np.asarray(w_B, dtype=np.float64).reshape(3)
        state[self.idx_ba]       = np.asarray(b_a, dtype=np.float64).reshape(3)
        state[self.idx_bg]       = np.asarray(b_g, dtype=np.float64).reshape(3)
        state[self.idx_tau_dist] = np.asarray(tau_dist, dtype=np.float64).reshape(6)
        return state

    def unpack_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        p_N      = state[self.idx_pN].copy()
        q_N      = self._normalize_quaternion(state[self.idx_qN].copy())
        v_B      = state[self.idx_vB].copy()
        w_B      = state[self.idx_wB].copy()
        b_a      = state[self.idx_ba].copy()
        b_g      = state[self.idx_bg].copy()
        tau_dist = state[self.idx_tau_dist].copy()
        return p_N, q_N, v_B, w_B, b_a, b_g, tau_dist

    # =========================================================
    # Sigma Points & UKF Core Routines
    # =========================================================
    def generate_sigma_points(self, x, P):
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        P = (P + P.T) / 2.0
        scaled_P = (self.ukf_state + self.lambda_) * (P + np.eye(P.shape[0]) * 1e-6)
        try:
            S = np.linalg.cholesky(scaled_P)
        except np.linalg.LinAlgError:
            U, Sigma, _ = np.linalg.svd(scaled_P)
            S = U @ np.diag(np.sqrt(np.clip(Sigma, 0.0, None)))

        sigma = np.zeros((self.n_sigma, self.ukf_state), dtype=np.float64)
        sigma[0] = x
        for i in range(self.ukf_state):
            sigma[i + 1] = x + S[:, i]
            sigma[i + 1 + self.ukf_state] = x - S[:, i]
        return sigma

    def weighted_mean(self, sigma_points):
        return np.sum(self.Wm[:, None] * sigma_points, axis=0)

    def state_covariance(self, sigma_points, x_mean):
        P = np.zeros((self.ukf_state, self.ukf_state), dtype=np.float64)
        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            P += self.Wc[i] * np.outer(dx, dx)
        return P

    def measurement_covariance(self, z_sigma, z_mean, R):
        measurement_dim = z_sigma.shape[1]
        S = np.zeros((measurement_dim, measurement_dim), dtype=np.float64)
        for i in range(self.n_sigma):
            dz = z_sigma[i] - z_mean
            S += self.Wc[i] * np.outer(dz, dz)
        return S + R + 1e-10 * np.eye(measurement_dim)

    def cross_covariance(self, sigma_points, x_mean, z_sigma, z_mean):
        measurement_dim = z_sigma.shape[1]
        Pxz = np.zeros((self.ukf_state, measurement_dim), dtype=np.float64)
        for i in range(self.n_sigma):
            dx = sigma_points[i] - x_mean
            dz = z_sigma[i] - z_mean
            Pxz += self.Wc[i] * np.outer(dx, dz)
        return Pxz

    # =========================================================
    # Fossen Dynamics & Process Model
    # =========================================================
    def compute_damping(self, nu):
        nu = nu.flatten()
        linear = self.Dlin @ nu
        quadratic = self.Dquad @ (np.abs(nu) * nu)
        return (linear + quadratic).reshape(-1,1)

    def compute_coriolis(self, nu):
        u,v,w,p,q,r = nu.flatten()
        C = np.array([
            [0,0,0,   0,  self.m*w, -self.m*v],
            [0,0,0, -self.m*w,  0,   self.m*u],
            [0,0,0, self.m*v, -self.m*u,    0],

            [0,   self.m*w, -self.m*v,   0,    self.Izz*r,  -self.Iyy*q],
            [-self.m*w, 0,   self.m*u, -self.Izz*r,   0,     self.Ixx*p],
            [self.m*v, -self.m*u,  0,   self.Iyy*q, -self.Ixx*p,      0]])
        return C @ nu

    def compute_restoring(self):
        return np.zeros((6,1))

    def compute_gamma(self, nu):
        gamma = (self.compute_coriolis(nu) +
                self.compute_damping(nu) +
                self.compute_restoring())
        return np.asarray(gamma, dtype=np.float64).reshape(6)
    
    def fossen_acceleration(self, nu_B, tau_control, tau_dist):
        nu_B = np.asarray(nu_B, dtype=np.float64).reshape(6)
        tau_control = np.asarray(tau_control, dtype=np.float64).reshape(6)
        tau_dist = np.asarray(tau_dist, dtype=np.float64).reshape(6)

        # Compute Coriolis, Damping, and Restoring forces using geometry helper
        gamma = np.asarray(self.compute_gamma(nu_B.reshape(-1, 1)), dtype=np.float64).reshape(6)
        
        # Fossen Equation: M * nu_dot = tau_control + tau_dist + gamma(nu)
        nu_dot_B = self.Minv @ (tau_control + tau_dist - gamma)
        return nu_dot_B

    def ukf_process_model(self, state, dt, nu_cmd):
        p_N, q_N, v_B, w_B, b_a, b_g, tau_dist = self.unpack_state(state)

        R_NB = self._quat_to_rotmat(q_N)
        nu_B = np.concatenate([v_B, w_B])
        nu_cmd = np.asarray(nu_cmd, dtype=np.float64).reshape(6)

        # 1. Bridge ArduSub velocity setpoint to effective force using Kp
        tau_control = self.Kp_vel @ (nu_cmd - nu_B)

        # 2. Kinematics Propagation
        pN_next = p_N + dt * (R_NB @ v_B)
        dq = self._quat_from_rotvec(w_B * dt)
        qN_next = self._normalize_quaternion(self._quat_multiply(q_N, dq))

        # 3. Kinetics Propagation (Fossen Dynamics + Lumped Disturbance)
        nu_dot_B = self.fossen_acceleration(nu_B, tau_control, tau_dist)
        vB_next = v_B + dt * nu_dot_B[:3]
        wB_next = w_B + dt * nu_dot_B[3:]

        # 4. Biases and Disturbance Random Walk
        ba_next = b_a
        bg_next = b_g
        tau_dist_next = tau_dist 

        return self.pack_state(pN_next, qN_next, vB_next, wB_next, ba_next, bg_next, tau_dist_next)

    def ukf_predict(self, x, P, dt, nu_cmd):
        sigma = self.generate_sigma_points(x, P)
        sigma_pred = np.zeros_like(sigma)

        for i in range(self.n_sigma):
            sigma_pred[i] = self.ukf_process_model(sigma[i], dt, nu_cmd)

        x_pred = self.weighted_mean(sigma_pred)
        P_pred = self.state_covariance(sigma_pred, x_pred) + self.build_Q(dt)
        x_pred[self.idx_qN] = self._normalize_quaternion(x_pred[self.idx_qN])

        return x_pred, P_pred, sigma_pred

    # =========================================================
    # Measurement Models (Gazebo IMU & Ground Truth)
    # =========================================================
    def imu_measurement_model(self, state):
        p_N, q_N, v_B, w_B, b_a, b_g, tau_dist = self.unpack_state(state)
        R_NB = self._quat_to_rotmat(q_N)
        g_N = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
        g_B = R_NB.T @ g_N

        nu_B = np.concatenate([v_B, w_B])
        # Approximate kinematic acceleration sensed by IMU body frame
        a_dot = self.fossen_acceleration(nu_B, np.zeros(6), tau_dist)[:3]
        
        a_meas_pred = a_dot + np.cross(w_B, v_B) - g_B + b_a
        w_meas_pred = w_B + b_g

        return np.concatenate([a_meas_pred, w_meas_pred])

    def gt_pose_measurement_model(self, state, q_ref):
        p_N, q_N, _, _, _, _, _ = self.unpack_state(state)
        # Return position and orientation error as rotvec for smooth UKF update
        q_err_rotvec = self._quat_error_rotvec(q_ref, q_N)
        return np.concatenate([p_N, q_err_rotvec])

    def gt_velocity_measurement_model(self, state):
        _, _, v_B, w_B, _, _, _ = self.unpack_state(state)
        return np.concatenate([v_B, w_B])

    # =========================================================
    # UKF Update Steps
    # =========================================================
    def ukf_update_imu(self, x_pred, P_pred, sigma_pred, z_imu):
        z_sigma = np.array([self.imu_measurement_model(s) for s in sigma_pred])
        z_mean = self.weighted_mean(z_sigma)
        
        S = self.measurement_covariance(z_sigma, z_mean, self.R_imu)
        Pxz = self.cross_covariance(sigma_pred, x_pred, z_sigma, z_mean)
        
        K = np.linalg.solve(S, Pxz.T).T
        innovation = np.asarray(z_imu, dtype=np.float64).reshape(6) - z_mean
        
        x_update = x_pred + K @ innovation
        x_update[self.idx_qN] = self._normalize_quaternion(x_update[self.idx_qN])
        P_update = P_pred - K @ S @ K.T
        return x_update, P_update

    def ukf_update_gazebo_gt(self, x_pred, P_pred, sigma_pred, z_pose, z_vel):
        # 1. Direct Ground Truth Injection for Kinematics (Eliminates dead-reckoning drift)
        x_update = x_pred.copy()
        
        # Directly set position and orientation from Gazebo GT to prevent integration drift
        x_update[self.idx_pN] = np.asarray(z_pose[:3], dtype=np.float64).reshape(3)
        x_update[self.idx_qN] = self._normalize_quaternion(z_pose[3:])
        
        # Directly set linear and angular velocity from GT (or let velocity update via UKF)
        x_update[self.idx_vB] = np.asarray(z_vel[:3], dtype=np.float64).reshape(3)
        x_update[self.idx_wB] = np.asarray(z_vel[3:], dtype=np.float64).reshape(3)

        # Keep covariance bounded and positive definite
        P_update = (P_pred + P_pred.T) / 2.0
        P_update = np.nan_to_num(P_update, nan=1e-3, posinf=1.0, neginf=-1.0)
        P_update += 1e-5 * np.eye(self.ukf_state)

        return x_update, P_update