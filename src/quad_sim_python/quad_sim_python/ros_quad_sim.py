
from threading import Lock
import numpy as np

from geometry_msgs.msg import Pose
from quad_sim_python_msgs.msg import QuadMotors, QuadWind, QuadState

import rclpy # https://docs.ros2.org/latest/api/rclpy/api/node.html
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from .quad import Quadcopter
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

quad_params["init_pos"] = [0,0,0]
quad_params["Ts"] = 0.005
quad_params["Tp"] = 1/50
quad_params["orient"] = "ENU"

class InitQuadSim(Node):

    def __init__(self):
        super().__init__('init_quadsim')
        # ros2 run quad_sim_python quadsim --ros-args -p target_frame:='flying_sensor'
        self.declare_parameter('target_frame', 'flying_sensor')
        self.target_frame = self.get_parameter(
            'target_frame').get_parameter_value().string_value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Call on_timer function every second
        self.timer = self.create_timer(0.1, self.on_timer)

    def on_timer(self):
        # Store frame names in variables that will be used to
        # compute transformations
        from_frame_rel = self.target_frame
        to_frame_rel = 'map'

        # Look up for the transformation between target_frame and turtle2 frames
        # and send velocity commands for turtle2 to reach target_frame
        try:
            now = rclpy.time.Time()
            trans = self.tf_buffer.lookup_transform(
                to_frame_rel,
                from_frame_rel,
                now)
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform {to_frame_rel} to {from_frame_rel}: {ex}')
            return

        self.output = [trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z]

        raise KeyboardInterrupt

class QuadSim(Node):
    def __init__(self, target_frame, init_pos=[0,0,0]):
        super().__init__('quadsim')
        self.w_cmd_lock = Lock()
        self.wind_lock = Lock()
        self.sim_pub_lock = Lock()

        # pos[3], quat[4], rpy[3], vel[3], vel_dot[3], omega[3], omega_dot[3]
        self.curr_state = np.zeros(22)

        self.wind = [0,0,0]
        self.prev_wind = [0,0,0]
        
        quad_params["init_pos"] = init_pos
        quad_params["target_frame"] = target_frame
        Dict2ROS2Params(self, quad_params)

        self.start_sim()
        self.w_cmd = [self.quad.params['w_hover']]*4
        self.get_logger().info(f'Simulator started!')

        self.quadpos_pub = self.create_publisher(Pose, f'/carla/{target_frame}/control/set_transform',1)
        self.quadstate_pub = self.create_publisher(QuadState, f'/quadsim/{target_frame}/state',1)

        self.receive_w_cmd = self.create_subscription(
            QuadMotors,
            f'/quadsim/{target_frame}/w_cmd',
            self.receive_w_cmd_cb,
            1)

        self.receive_wind = self.create_subscription(
            QuadWind,
            f'/quadsim/{target_frame}/wind',
            self.receive_wind_cb,
            1)

    def start_sim(self):
        params = ROS2Params2Dict(self, 'quadsim', quad_params.keys())
        init_pose = np.array([*params['init_pos'],0,0,0]) # x0, y0, z0, phi0, theta0, psi0
        init_twist = np.array([0,0,0,0,0,0]) # xdot, ydot, zdot, p, q, r
        init_states = np.hstack((init_pose,init_twist))
        self.t = 0
        self.Ts = params['Ts']
        orient = params['orient']
        self.quad = Quadcopter(self.t, init_states, params=params.copy(), orient=orient)
        new_params = {key: self.quad.params[key] for key in self.quad.params if key not in params}
        Dict2ROS2Params(self, new_params)

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
            self.sim_pub_lock.release()
        self.t += self.Ts
        self.get_logger().info(f'Quad pos/vel: {self.quad.pos} / {self.quad.vel}')

    def on_sim_publish(self):
        msg1 = Pose()
        msg2 = QuadState()
        with self.sim_pub_lock:
            msg1.position.x = self.curr_state[0]
            msg1.position.y = self.curr_state[1]
            msg1.position.z = self.curr_state[2]
            msg1.orientation.x = self.curr_state[3]
            msg1.orientation.y = self.curr_state[4]
            msg1.orientation.z = self.curr_state[5]
            msg1.orientation.w = self.curr_state[6]

            msg2.t = self.t
            msg2.pos.x = self.curr_state[0:3][0]
            msg2.pos.y = self.curr_state[0:3][1]
            msg2.pos.z = self.curr_state[0:3][2]
            msg2.quat.x = self.curr_state[3:7][0]
            msg2.quat.y = self.curr_state[3:7][1]
            msg2.quat.z = self.curr_state[3:7][2]
            msg2.quat.w = self.curr_state[3:7][3]
            msg2.rpy.x = self.curr_state[7:10][0]
            msg2.rpy.y = self.curr_state[7:10][1]
            msg2.rpy.z = self.curr_state[7:10][2]
            msg2.vel.x = self.curr_state[10:13][0]
            msg2.vel.y = self.curr_state[10:13][1]
            msg2.vel.z = self.curr_state[10:13][2]
            msg2.vel_dot.x = self.curr_state[13:16][0]
            msg2.vel_dot.y = self.curr_state[13:16][1]
            msg2.vel_dot.z = self.curr_state[13:16][2]
            msg2.omega.x = self.curr_state[16:19][0]
            msg2.omega.y = self.curr_state[16:19][1]
            msg2.omega.z = self.curr_state[16:19][2]
            msg2.omega_dot.x = self.curr_state[19:22][0]
            msg2.omega_dot.y = self.curr_state[19:22][1]
            msg2.omega_dot.z = self.curr_state[19:22][2]
        self.quadpos_pub.publish(msg1)
        self.quadstate_pub.publish(msg2)
        self.get_logger().info(f'Quad State: {msg2}')

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

    quad_node = QuadSim("flying_sensor")
    try:
        rclpy.spin(quad_node)
    except KeyboardInterrupt:
        pass


    print("Shutting down QuadSim...")
    rclpy.shutdown()


if __name__ == '__main__':
   main()