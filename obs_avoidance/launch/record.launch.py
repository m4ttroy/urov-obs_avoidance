import os
from datetime import datetime

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    home_dir = os.path.expanduser('~')
    
    timestamp = datetime.now().strftime('%Y_%m_%d-%H_%M_%S')
    bag_output_path = os.path.join(home_dir, f'rosbag_{timestamp}')


    record_process = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'record',
            '-s', 'mcap',
            '-o', bag_output_path,
            '--max-cache-size', '100000000',
            '/model/bluerov2_ping/odometry',
            '/model/bluerov2_ping/pose',
            '/analysis/gt_path',
            '/analysis/ref_path',
            '/analysis/depth_tracking',
            '/analysis/velocity_tracking',
            '/analysis/errors',
            '/analysis/object_reconstruction',
            '/analysis/solve_time',
        ], output='screen'
    )

    return LaunchDescription([
        record_process
    ])