
from threading import Lock
import numpy as np

from geometry_msgs.msg import Pose
from quad_sim_python_msgs.msg import QuadMotors, QuadWind

import rclpy # https://docs.ros2.org/latest/api/rclpy/api/node.html
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener


from .quad import Quadcopter
from rclpy_param_helper import Dict2ROS2Params, ROS2Params2Dict

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



# The quad needs to publish its parameters (quad.params) to be read by control node!!!
class QuadSim(Node):
    def __init__(self, target_frame, init_pos=[0,0,0], Ts=0.005, Tp=1/50, orient="ENU"):
        super().__init__('quadsim')
        self.started = False # the simulation only starts after the first w_cmd is received
        self.w_cmd_lock = Lock()
        self.wind_lock = Lock()
        self.pose_lock = Lock()

        self.curr_pose = np.zeros(7)

        self.w_cmd = [0,0,0,0]
        self.wind = [0,0,0]
        self.prev_wind = [0,0,0]
        
        init_pose = np.array([*init_pos,0,0,0]) # x0, y0, z0, phi0, theta0, psi0
        init_twist = np.array([0,0,0,0,0,0]) # xdot, ydot, zdot, p, q, r
        init_states = np.hstack((init_pose,init_twist))
        self.t = 0
        self.Ts = Ts
        self.orient = orient
        self.quad = Quadcopter(self.t, init_states, orient=orient)
        Dict2ROS2Params(self, self.quad.params)
        params = ROS2Params2Dict(self, 'quadsim', self.quad.params.keys())
        print(params)
        raise

        self.quadpos_pub = self.create_publisher(Pose, f'/carla/{target_frame}/control/set_transform',1)

        self.receive_w_cmd = self.create_subscription(
            QuadMotors,
            '/quadsim/w_cmd',
            self.receive_w_cmd_cb,
            1)

        self.receive_wind = self.create_subscription(
            QuadWind,
            '/quadsim/wind',
            self.receive_wind_cb,
            1)

        self.sim_loop_timer = self.create_timer(Ts, self.on_sim_loop)
        self.sim_publish_timer = self.create_timer(Tp, self.on_sim_publish)

    def receive_w_cmd_cb(self, msg):
        with self.w_cmd_lock:
            self.started = True # the simulator will wait until it receives a command
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
        with self.w_cmd_lock:
            if self.started:
                self.quad.update(self.t, self.Ts, self.w_cmd, self.prev_wind)
        if self.pose_lock.acquire(blocking=False):
            self.curr_pose[:3] = self.quad.pos[:]
            self.curr_pose[3:] = self.quad.quat[:]
            self.pose_lock.release()
        self.t += self.Ts
        self.get_logger().info(f'Quad pos/vel: {self.quad.pos} / {self.quad.vel}')

    def on_sim_publish(self):
        msg = Pose()
        with self.pose_lock:
            msg.position.x = self.curr_pose[0]
            msg.position.y = self.curr_pose[1]
            msg.position.z = self.curr_pose[2]
            msg.orientation.x = self.curr_pose[3]
            msg.orientation.y = self.curr_pose[4]
            msg.orientation.z = self.curr_pose[5]
            msg.orientation.w = self.curr_pose[6]
        self.quadpos_pub.publish(msg)
        self.get_logger().info(f'Quad Pose: {msg}')

def main():
    print("Starting node...")
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

    quad_node = QuadSim("target_frame")
    try:
        rclpy.spin(quad_node)
    except KeyboardInterrupt:
        pass


    print("Shutting down QuadSim...")
    rclpy.shutdown()


if __name__ == '__main__':
   main()