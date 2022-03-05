# quad_sim_python
ROS2 packages to simulate (and control) a quadcopter. These packages were created to be used with a flying sensor in CARLA.


# How to use:
Easy way: I SHOULD CREATE AN IMAGE WITH THE PACKAGES ALREADY CLONED AND BUILT AND A SCRIPT TO LAUNCH BOTH CARLA AND ROS2!

1. Clone the repos with ROS2 packages:

```$ mkdir -p ~/carla-ros && cd ~/carla-ros```

```$ git clone --recurse-submodules https://github.com/ricardodeazambuja/carla-ros src/ros-bridge```

```$ git clone https://github.com/ricardodeazambuja/quad_sim_python src/quad_sim_python```

2. Start the CARLA simulator (headless):
https://github.com/ricardodeazambuja/carla-simulator-python
$ ./launch_simulator.sh

2. Get the script to launch my ROS2
https://github.com/ricardodeazambuja/ros2-playground

$ ./launch_ros2_desktop.sh -g -d ~/carla-ros --image carla-ros2-bridge:galactic

The script above will source ROS2.

$ colcon build --symlink-install

$ source install/setup.bash

Now I need a launch file to run the commands below and accept some useful arguments like town, objects_definition_file, init_pose, gains...
$ ros2 launch carla_ros_bridge carla_ros_bridge.launch.py passive:=False town:=Town10HD_Opt
$ ros2 launch carla_spawn_objects carla_spawn_objects.launch.py objects_definition_file:=src/ros-bridge/carla_spawn_objects/config/flying_sensor.json
$ ros2 run quad_sim_python quadsim --ros-args -p init_pose:=[0,0,2,0,0,0]
$ ros2 run quad_sim_python quadctrl --ros-args -p Px:=2



# Acknowledgments
Quadcopter simulator and controller adapted from https://github.com/bobzwik/Quad_Exploration
