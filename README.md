# BlueROV2 Model for Obstacle-avoidance in Gazebo Harmonic

This is an obstacle-avoidance control experiment using gazebo harmonic simulation. The simulation uses BlueROV2 with ping-sonar positioned in a forward-facing position to detect an obstacle. The environment use an 8 x 8 x 8 m underwater tank with cylinder objects placed at the U-ROV intersecting survey path. The U-ROV usses GCBB survey strategy to reach the final point.

The BlueROV2 model already include the BuoyancyPlugin, HydrodynamicsPlugin and ThrusterPlugin to better simulate the implementation.

<img width="370" height="370" alt="gz_sim" src="https://github.com/user-attachments/assets/ca449282-6850-4db1-bf4a-468157a2a734" /> 
<img width="370" height="370" alt="survey_route" src="https://github.com/user-attachments/assets/3fc55ea8-cdd9-4601-a935-6bf9d006253b" />



## Requirements

Please ensure that the following requirements have been met prior to installing the project

* [Gazebo Harmonic](https://gazebosim.org/docs/harmonic/install)
* [ardupilot_gazebo for Harmonic](https://github.com/ArduPilot/ardupilot_gazebo)
* [ArduSub and MAVProxy](https://ardupilot.org/dev/docs/building-setup-linux.html)
* [MAVROS](https://github.com/mavlink/mavros/blob/ros2/docs/installation.md)

## Running Gazebo

Configure the ~/.bashrc file prior to run the gazebo simulation, make sure to change "your_ws/src" to your instalation directory

~~~bash
export GZ_VERSION=harmonic
export GZ_SIM_RESOURCE_PATH=$HOME/your_ws/src/bluerov2-obs_avoidance/bluerov2_gz/models:$HOME/your_ws/src/bluerov2-obs_avoidance/bluerov2_gz/worlds:${GZ_SIM_RESOURCE_PATH}
export GZ_SIM_SYSTEM_PLUGIN_PATH=$HOME/your_ws/src/ardupilot_gazebo/build:$GZ_SIM_SYSTEM_PLUGIN_PATH
export PATH=$PATH:$HOME/rov_ws/src/ardupilot/Tools/autotest
~~~

To run the simulation of the underwater tank

~~~bash
gz sim -v4 -r 8x8_tank.world
~~~

Run the BlueROV2 SITL

~~~bash
source ~/venv-ardupilot/bin/activate
cd ~/your_ws/src/ardupilot/ArduSub
sim_vehicle.py -v ArduSub -f vectored --model JSON -L RATBeach --console
~~~

## GZ Topic to ROS2 Bridge 
~~~bash
ros2 run ros_gz_bridge parameter_bridge \
    '/model/bluerov2_ping/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose' \
    '/model/bluerov2_ping/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry' \
    '/ping_link/sensor/sonar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan' \
    '/world/bluerov2_ping/model/bluerov2_ping/link/base_link/sensor/imu_sensor/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
~~~
Use this command to bridge the Gazebo topics to ROS2 topics for topic subscriber.


Once Gazebo has been launched, you can directly send the control commands to the BlueROV2 model in Gazebo via mavproxy to arm and enter GUIDED mode, then open a new terminal and run the controller

~~~bash
ros2 run obs_avoid "control" 
~~~

Change the "control" to the controller matching your option
* `start` to place the U-ROV at initial point (0, 0, -5)
* `pid` to use PID controller
* `nmpc` to use N-MPC controller

N.B. for PID you can add extra argument --args -p enable_model:False to use UKF disturbance estimation or True to use known modeled disturbance


## References

* https://github.com/ardupilot/ardupilot_gazebo/wiki
* https://gazebosim.org/docs/harmonic/install
* https://ardupilot.org/dev/docs/building-setup-linux.html
* https://ardupilot.org/dev/docs/setting-up-sitl-on-linux.html
* https://ardupilot.org/mavproxy/docs/getting_started/download_and_installation.html
* https://www.ardusub.com/developers/rc-input-and-output.html
* https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
