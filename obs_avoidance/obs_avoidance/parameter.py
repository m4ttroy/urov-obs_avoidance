import numpy as np

# Clearence Properties
obstacle_diameter = 0.25
rov_width_margin = 0.17
safety_clearance = 0.50
total_bypass_offset = rov_width_margin + (obstacle_diameter / 2.0) + safety_clearance

# BlueROV2 ping-sonar properties
beam_angles = [-0.148, -0.074, 0.0, 0.074, 0.148] # Sonar beam configuration (5 beams spanning ~17 degrees)
sensor_offset_x = 0.16
sensor_offset_y = 0.00

# Detection and wall model
detection_range = 2.5
sonar_wall_offset = 0.40

# Survey Path Waypoints
waypoints = [
    (0.0, 0.0), (8.0, 0.0), 
    (8.0, 2.0), (0.0, 2.0),
    (0.0, 4.0), (8.0, 4.0), 
    (8.0, 6.0), (0.0, 6.0), 
    (0.0, 8.0), (8.0, 8.0),
    ]


# Controller Propertiess
control_rate      = 50.00
target_depth      = -5.00
max_speed         = 1.00
max_accel         = 1.50
acceptance_radius = 0.20

yaw_tolerance     = 5.00
yaw_hold_seconds  = 2.0
debug_period      = 1.00

emergency_decel   = 1.50
settle_speed      = 0.08
settle_duration   = 0.050