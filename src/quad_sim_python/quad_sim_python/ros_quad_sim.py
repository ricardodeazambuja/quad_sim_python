
from threading import Lock
import numpy as np

from geometry_msgs.msg import Pose
from quad_sim_python_msgs.msg import QuadMotors, QuadWind, QuadState

import rclpy # https://docs.ros2.org/latest/api/rclpy/api/node.html
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from quad_sim_python.quad import Quadcopter
import quad_sim_python.utils as utils
from rclpy_param_helper import Dict2ROS2Params, ROS2Params2Dict

quad_params = {}
# Moments of inertia:
# (e.g. from Bifilar Pendulum experiment https://arc.aiaa.org/doi/abs/10.2514/6.2007-6822)
Ixx = 0.0123
Iyy = 0.0123
Izz = 0.0224
IB  = np.array([[Ixx, 0,   0  ],
                [0,   Iyy, 0  ],
                [0,   0,   Izz]]) # Inertial tensor (kg*m^2)

IRzz = 2.7e-5   # Rotor moment of inertia (kg*m^2)
quad_params["mB"]   = 1.2    # mass (kg)
quad_params["g"]    = 9.81   # gravity (m/s^2)
quad_params["dxm"]  = 0.16   # arm length (m) - between CG and front
quad_params["dym"]  = 0.16   # arm length (m) - between CG and right
quad_params["dzm"]  = 0.05   # motor height (m)
quad_params["IB"]   = IB
quad_params["IRzz"] = IRzz
quad_params["Cd"]         = 0.1      # https://en.wikipedia.org/wiki/Drag_coefficient
quad_params["kTh"]        = 1.076e-5 # thrust coeff (N/(rad/s)^2)  (1.18e-7 N/RPM^2)
quad_params["kTo"]        = 1.632e-7 # torque coeff (Nm/(rad/s)^2)  (1.79e-9 Nm/RPM^2)
quad_params["minThr"]     = 0.1*4    # Minimum total thrust
quad_params["maxThr"]     = 9.18*4   # Maximum total thrust
quad_params["minWmotor"]  = 75       # Minimum motor rotation speed (rad/s)
quad_params["maxWmotor"]  = 925      # Maximum motor rotation speed (rad/s)
quad_params["tau"]        = 0.015    # Value for second order system for Motor dynamics
quad_params["kp"]         = 1.0      # Value for second order system for Motor dynamics
quad_params["damp"]       = 1.0      # Value for second order system for Motor dynamics
quad_params["motorc1"]    = 8.49     # w (rad/s) = cmd*c1 + c0 (cmd in %)
quad_params["motorc0"]    = 74.7
# Select whether to use gyroscopic precession of the rotors in the quadcopter dynamics
# ---------------------------
# Set to False if rotor inertia isn't known (gyro precession has negigeable effect on drone dynamics)
quad_params["usePrecession"] = False

quad_params["Ts"] = 1/200 # state calculation time step (current ode settings run faster using a smaller value)
quad_params["Tp"] = 1/50 # period it publishes the current state
quad_params["orient"] = "ENU"
quad_params["target_frame"] = 'flying_sensor'
quad_params["map_frame"] = 'map'

class QuadSim(Node):
    def __init__(self):
        super().__init__('quadsim', 
                         allow_undeclared_parameters=True, # necessary for using set_parameters
                         automatically_declare_parameters_from_overrides=True) # allows command line parameters

        # Read ROS2 parameters the user may have set 
        # E.g. (https://docs.ros.org/en/foxy/How-To-Guides/Node-arguments.html):
        # --ros-args -p init_pose:=[0,0,0,0,0,0])
        # --ros-args --params-file params.yaml
        read_params = ROS2Params2Dict(self, 'quadsim', list(quad_params.keys()) + ["init_pose"])
        for k,v in read_params.items():
            # Update local parameters
            quad_params[k] = v
        
        # Update ROS2 parameters
        Dict2ROS2Params(self, quad_params) # the controller needs to read some parameters from here


        self.w_cmd_lock = Lock()
        self.wind_lock = Lock()
        self.sim_pub_lock = Lock()

        # pos[3], quat[4], rpy[3], vel[3], vel_dot[3], omega[3], omega_dot[3]
        self.curr_state = np.zeros(22, dtype='float32')

        self.wind = [0,0,0]
        self.prev_wind = [0,0,0]


        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        if "init_pose" not in quad_params:
            # Look up for the transformation between target_frame and map_frame frames
            try:
                now = rclpy.time.Time()
                trans = self.tf_buffer.lookup_transform(
                    quad_params["map_frame"],
                    quad_params["target_frame"],
                    now)

                init_pos = [trans.transform.translation.x, 
                            trans.transform.translation.y, 
                            trans.transform.translation.z]

                init_quat = [trans.rotation.translation.x,
                            trans.rotation.translation.y,
                            trans.rotation.translation.z,
                            trans.rotation.translation.w]
                init_rpy = utils.quatToYPR_ZYX(init_quat)[::-1]

                quad_params["init_pose"] = init_pos + init_rpy

                # Update ROS2 parameters
                Dict2ROS2Params(self, {"init_pose": init_pos+init_rpy}) # the controller needs to read some parameters from here

            except TransformException as ex:
                self.get_logger().error(f'Could not transform {quad_params["map_frame"]} to {quad_params["target_frame"]}: {ex}')
                self.get_logger().error('init_pose not available... using [0,0,0,0,0,0]')
                # Update ROS2 parameters
                Dict2ROS2Params(self, {"init_pose": [0,0,0,0,0,0]})


        self.start_sim()
        self.get_logger().info(f'Simulator started!')

        self.quadpos_pub = self.create_publisher(Pose, f'/carla/{quad_params["target_frame"]}/control/set_transform',1)
        self.quadstate_pub = self.create_publisher(QuadState, f'/quadsim/{quad_params["target_frame"]}/state',1)

        self.receive_w_cmd = self.create_subscription(
            QuadMotors,
            f'/quadsim/{quad_params["target_frame"]}/w_cmd',
            self.receive_w_cmd_cb,
            1)

        self.receive_wind = self.create_subscription(
            QuadWind,
            f'/quadsim/{quad_params["target_frame"]}/wind',
            self.receive_wind_cb,
            1)

    def start_sim(self):
        params = ROS2Params2Dict(self, 'quadsim', quad_params.keys())
        init_pose = np.array(params['init_pose']) # x0, y0, z0, phi0, theta0, psi0
        init_twist = np.array([0,0,0,0,0,0]) # xdot, ydot, zdot, p, q, r
        init_states = np.hstack((init_pose,init_twist))
        self.t = 0
        self.Ts = params['Ts']
        self.quad = Quadcopter(self.t, init_states, params=params.copy(), orient=params['orient'])
        self.w_cmd = [self.quad.params['w_hover']]*4
        new_params = {key: self.quad.params[key] for key in self.quad.params if key not in params}
        Dict2ROS2Params(self, new_params) # some parameters are created by the quad object

        self.sim_loop_timer = self.create_timer(self.Ts, self.on_sim_loop)
        self.sim_publish_timer = self.create_timer(params['Tp'], self.on_sim_publish)

    def receive_w_cmd_cb(self, msg):
        with self.w_cmd_lock:
            self.w_cmd = [msg.m1, 
                          msg.m2,
                          msg.m3,
                          msg.m4]
        self.get_logger().info(f'Received w_cmd: {self.w_cmd}')

    def receive_wind_cb(self, msg):
        with self.wind_lock:
            self.wind = [msg.vel_w, 
                         msg.head_w,
                         msg.elev_w]
        self.get_logger().info(f'Received wind: {self.wind}')

    def on_sim_loop(self):
        if self.wind_lock.acquire(blocking=False):
            self.prev_wind[:] = self.wind[:]
            self.wind_lock.release()

        with self.w_cmd_lock:
            self.quad.update(self.t, self.Ts, self.w_cmd, self.prev_wind)

        if self.sim_pub_lock.acquire(blocking=False):
            self.curr_state[0:3] = self.quad.pos[:]
            self.curr_state[3:7] = self.quad.quat[:]
            self.curr_state[7:10] = self.quad.euler[:]
            self.curr_state[10:13] = self.quad.vel[:]
            self.curr_state[13:16] = self.quad.vel_dot[:]
            self.curr_state[16:19] = self.quad.omega[:]
            self.curr_state[19:22] = self.quad.omega_dot[:]
            self.t += self.Ts
            self.sim_pub_lock.release()
        self.get_logger().info(f'Quad pos/vel: {self.quad.pos} / {self.quad.vel}')

    def on_sim_publish(self):
        msg1 = Pose()
        msg2 = QuadState()
        with self.sim_pub_lock:
            msg1.position.x = float(self.curr_state[0])
            msg1.position.y = float(self.curr_state[1])
            msg1.position.z = float(self.curr_state[2])
            msg1.orientation.x = float(self.curr_state[3])
            msg1.orientation.y = float(self.curr_state[4])
            msg1.orientation.z = float(self.curr_state[5])
            msg1.orientation.w = float(self.curr_state[6])

            msg2.t = self.t
            msg2.pos = self.curr_state[0:3][:]
            msg2.quat = self.curr_state[3:7][:]
            msg2.rpy = self.curr_state[7:10][:]
            msg2.vel = self.curr_state[10:13][:]
            msg2.vel_dot = self.curr_state[13:16][:]
            msg2.omega = self.curr_state[16:19][:]
            msg2.omega_dot = self.curr_state[19:22][:]
        self.quadpos_pub.publish(msg1)
        self.quadstate_pub.publish(msg2)
        self.get_logger().debug(f'Quad State: {self.curr_state}')

def main():
    print("Starting QuadSim...")
    rclpy.init()

    # # First node will wait for the TF to be available
    # temp_node = InitQuadSim()
    # try:
    #     rclpy.spin(temp_node)
    # except KeyboardInterrupt:
    #     pass
    
    # # Simulator will use the position previously acquired
    # quad_node = QuadSim(temp_node.target_frame, init_pos=temp_node.output)
    # temp_node.destroy_node()

    quad_node = QuadSim()
    try:
        rclpy.spin(quad_node)
    except KeyboardInterrupt:
        pass


    print("Shutting down QuadSim...")
    rclpy.shutdown()


if __name__ == '__main__':
   main()