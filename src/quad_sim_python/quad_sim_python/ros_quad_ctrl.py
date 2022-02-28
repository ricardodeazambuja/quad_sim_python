
from threading import Lock
from copy import copy
import numpy as np
from numpy.linalg import norm

from geometry_msgs.msg import Pose
from quad_sim_python_msgs.msg import QuadMotors, QuadState, QuadControlSetPoint

import rclpy # https://docs.ros2.org/latest/api/rclpy/api/node.html
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy_param_helper import Dict2ROS2Params, ROS2Params2Dict

from .ctrl import Controller
import quad_sim_python.utils as utils


# First I will create something using nodes instead of action server. 
# The controller will keep using the last setpoint received.

ctrl_params = {
            # Position P gains
            "Px"    : 2.0,
            "Py"    : 2.0,
            "Pz"    : 1.0,

            # Velocity P-D gains
            "Pxdot" : 5.0,
            "Dxdot" : 0.5,
            "Ixdot" : 5.0,

            "Pydot" : 5.0,
            "Dydot" : 0.5,
            "Iydot" : 5.0,

            "Pzdot" : 4.0,
            "Dzdot" : 0.5,
            "Izdot" : 5.0,

            # Attitude P gains
            "Pphi"   : 8.0,
            "Ptheta" : 8.0,
            "Ppsi"   : 1.5,

            # Rate P-D gains
            "Pp" : 1.5,
            "Dp" : 0.04,

            "Pq" : 1.5,
            "Dq" : 0.04 ,

            "Pr" : 1.0,
            "Dr" : 0.1,

            # Max Velocities (x,y,z) [m/s]
            "uMax" : 5.0,
            "vMax" : 5.0,
            "wMax" : 5.0,

            "saturateVel_separately" : True,

            # Max tilt [degrees]
            "tiltMax" : 50.0,

            # Max Rate [rad/s]
            "pMax" : 200.0,
            "qMax" : 200.0,
            "rMax" : 150.0,

             # Minimum velocity for yaw follow to kick in [m/s]
            "minTotalVel_YawFollow" : 0.1,

            "useIntegral" : True,    # Include integral gains in linear velocity control
            }

class QuadCtrl(Node):

    def __init__(self):
        super().__init__('quadctrl', 
                         allow_undeclared_parameters=True, 
                         automatically_declare_parameters_from_overrides=True)

        self.started = False
        self.quad_state = False
        self.t = None

        self.quadstate_lock = Lock()
        self.ctrl_sp_lock = Lock()

        self.curr_sp = QuadControlSetPoint()
        self.prev_sp = copy(self.curr_sp)

        # Read ROS2 parameters the user may have set 
        # E.g. (https://docs.ros.org/en/foxy/How-To-Guides/Node-arguments.html):
        # --ros-args -p Px:=5)
        # --ros-args --params-file params.yaml
        read_params = ROS2Params2Dict(self, 'quadctrl', ctrl_params.keys())
        for k,v in read_params.items():
            # Update local parameters
            ctrl_params[k] = v
        
        # Update ROS2 parameters
        Dict2ROS2Params(self, ctrl_params) # the controller needs to read some parameters from here

        quad_params_list = ['mB', 'g', 'IB', 'maxThr', 'minThr', 'orient', 'mixerFMinv', 'minWmotor', 'maxWmotor', 'target_frame']
        self.quad_params = ROS2Params2Dict(self, 'quadsim', quad_params_list)

        self.receive_control_sp = self.create_subscription(
            QuadControlSetPoint,
            f"/quadctrl/{self.quad_params['target_frame']}/ctrl_sp",
            self.receive_control_sp_cb,
            1)

        self.receive_quadstate = self.create_subscription(
            QuadState,
            f"/quadsim/{self.quad_params['target_frame']}/state",
            self.receive_quadstate_cb,
            1)

        self.w_cmd_pub = self.create_publisher(QuadMotors, f"/quadsim/{self.quad_params['target_frame']}/w_cmd",1)

    def start_ctrl(self):
        params = ROS2Params2Dict(self, 'quadctrl', ctrl_params.keys())
        self.ctrl = Controller(self.quad_params, orient=self.quad_params['orient'], params=params)

        
    def receive_control_sp_cb(self, msg):
        if not self.started:
            self.start_ctrl()
            self.started = True
            self.get_logger().info(f'Controller started!')

        with self.ctrl_sp_lock:
            self.curr_sp = msg

        self.get_logger().info(f'Received control setpoint: {self.curr_sp}')



    def receive_quadstate_cb(self, msg):
        self.quad_state = True
        if self.t != None:
            self.prev_t = self.t = msg.t
        else:
            self.prev_t = self.t
            self.t = msg.t
        self.get_logger().info(f'Received QuadState: {msg}')

        if self.started:
            if self.ctrl_sp_lock.acquire(blocking=False):
                self.prev_sp = self.curr_sp
                self.ctrl_sp_lock.release()

            self.ctrl.control((self.t-self.prev_t), self.prev_sp.ctrltype, self.prev_sp.yawtype, 
                               self.prev_sp.pos, self.prev_sp.vel, self.prev_sp.acc, self.prev_sp.thr, 
                               self.prev_sp.yaw, self.prev_sp.yawrate,
                               msg.pos, msg.vel, msg.vel_dot, msg.quat, msg.omega, msg.omega_dot, msg.rpy[2])

            # Mixer (generates motor speeds)
            # --------------------------- 
            thurst_drone_z = norm(self.ctrl.thrust_rep_sp) #max(0,np.dot(ctrl.thrust_rep_sp,ctrl.drone_z))
            moments_drone = 9.81*np.dot(self.quad_params["IB"], self.ctrl.rateCtrl)
            w_cmd = utils.mixerFM(thurst_drone_z, moments_drone, 
                                  self.quad_params["mixerFMinv"], self.quad_params["minWmotor"], self.quad_params["maxWmotor"])
            msg = QuadMotors()
            msg.m1 = int(w_cmd[0])
            msg.m2 = int(w_cmd[1])
            msg.m3 = int(w_cmd[2])
            msg.m4 = int(w_cmd[3])
            self.w_cmd_pub.publish(msg)


def main():
    print("Starting QuadCtrl...")
    rclpy.init()
    ctrl_node = QuadCtrl()
    try:
        rclpy.spin(ctrl_node)
    except KeyboardInterrupt:
        pass


    print("Shutting down QuadCtrl...")
    rclpy.shutdown()


if __name__ == '__main__':
   main()