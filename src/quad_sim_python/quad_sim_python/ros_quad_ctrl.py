
from threading import Lock
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
            "Py"    : 2.0,
            "Px"    : 2.0,
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
        super().__init__('quadctrl')

        self.started = False
        self.quad_state = False
        self.t = None

        self.quadstate_lock = Lock()

        # pos[3], quat[4], rpy[3], vel[3], vel_dot[3], omega[3], omega_dot[3]
        self.curr_state = np.zeros(22)
        self.curr_sp = QuadControlSetPoint()

        quad_params_list = ['mB', 'g', 'maxThr', 'minThr', 'orient', 'mixerFMinv', 'minWmotor', 'maxWmotor', 'target_frame', 'init_pos']
        self.quad_params = ROS2Params2Dict(self, 'quadsim', quad_params_list)

        self.receive_control_sp = self.create_subscription(
            QuadControlSetPoint,
            f"/quadctrl/{self.quad_params['target_frame']}/control/set_transform",
            self.receive_control_sp_cb,
            1)

        self.receive_quadstate = self.create_subscription(
            QuadState,
            f"/quadsim/{self.quad_params['target_frame']}/state",
            self.receive_quadstate_cb,
            1)

        self.w_cmd_pub = self.create_publisher(QuadMotors, f"/quadsim/{self.quad_params['target_frame']}/w_cmd",1)

    def start_ctrl(self):
        Dict2ROS2Params(self, ctrl_params)
        ctrl_params = ROS2Params2Dict(self, 'quadctrl', ctrl_params.keys())
        self.ctrl = Controller(self.quad_params, orient=self.quad_params['orient'], params=ctrl_params)

        
    def receive_control_sp_cb(self, msg):
        if not self.started:
            self.start_ctrl()
            self.started = True
            self.get_logger().info(f'Controller started!')

        with self.w_cmd_lock:
            self.curr_sp = msg

        self.get_logger().info(f'Received QuadState: {self.curr_state}')



    def receive_quadstate_cb(self, msg):
        self.quad_state = True
        if self.t != None:
            self.prev_t = self.t = msg.t
        else:
            self.prev_t = self.t
            self.t = msg.t

        self.curr_state[0:3][0] = msg.pos.x
        self.curr_state[0:3][1] = msg.pos.y
        self.curr_state[0:3][2] = msg.pos.z
        self.curr_state[3:7][0] = msg.quat.x
        self.curr_state[3:7][1] = msg.quat.y
        self.curr_state[3:7][2] = msg.quat.z
        self.curr_state[3:7][3] = msg.quat.w
        self.curr_state[7:10][0] = msg.rpy.x
        self.curr_state[7:10][1] = msg.rpy.y
        self.curr_state[7:10][2] = msg.rpy.z
        self.curr_state[10:13][0] = msg.vel.x
        self.curr_state[10:13][1] = msg.vel.y
        self.curr_state[10:13][2] = msg.vel.z
        self.curr_state[13:16][0] = msg.vel_dot.x
        self.curr_state[13:16][1] = msg.vel_dot.y
        self.curr_state[13:16][2] = msg.vel_dot.z
        self.curr_state[16:19][0] = msg.omega.x
        self.curr_state[16:19][1] = msg.omega.y
        self.curr_state[16:19][2] = msg.omega.z
        self.curr_state[19:22][0] = msg.omega_dot.x
        self.curr_state[19:22][1] = msg.omega_dot.y
        self.curr_state[19:22][2] = msg.omega_dot.z
        self.get_logger().info(f'Received QuadState: {self.curr_state}')

        if self.started:
            self.ctrl.control((self.t-self.prev_t), self.curr_sp.ctrltype, self.curr_sp.yawtype, 
                               [self.curr_sp.pos.x,self.curr_sp.pos.y,self.curr_sp.pos.z], [self.curr_sp.vel.x,self.curr_sp.vel.y,self.curr_sp.vel.z], 
                               [self.curr_sp.acc.x,self.curr_sp.acc.y,self.curr_sp.acc.z], [self.curr_sp.thr.x,self.curr_sp.thr.y,self.curr_sp.thr.z], 
                               self.curr_sp.yaw, self.curr_sp.yawrate,
                               self.curr_state[0:3], self.curr_state[10:13], self.curr_state[13:16], 
                               self.curr_state[3:7], self.curr_state[16:19], self.curr_state[19:22], self.curr_state[7:10][2])

            # Mixer (generates motor speeds)
            # --------------------------- 
            thurst_drone_z = norm(self.ctrl.thrust_rep_sp) #max(0,np.dot(ctrl.thrust_rep_sp,ctrl.drone_z))
            moments_drone = 9.81*np.dot(self.quad_params["IB"], self.ctrl.rateCtrl)
            w_cmd = utils.mixerFM(thurst_drone_z, moments_drone, 
                                  self.quad_params["mixerFMinv"], self.quad_params["minWmotor"], self.quad_params["maxWmotor"])
            msg = QuadMotors()
            msg.m1 = w_cmd[0]
            msg.m2 = w_cmd[1]
            msg.m3 = w_cmd[2]
            msg.m4 = w_cmd[3]
            self.w_cmd_pub.pub(msg)


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