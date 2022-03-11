# -*- coding: utf-8 -*-
"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

# Position and Velocity Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/PositionControl.cpp
# Desired Thrust to Desired Attitude based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_pos_control/Utility/ControlMath.cpp
# Attitude Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/AttitudeControl/AttitudeControl.cpp
# and https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/154099/eth-7387-01.pdf
# Rate Control based on https://github.com/PX4/Firmware/blob/master/src/modules/mc_att_control/mc_att_control_main.cpp

import numpy as np
from numpy import pi
from numpy import sin, cos, sqrt
from numpy.linalg import norm
import sys, os
curr_path = os.getcwd()
if os.path.basename(curr_path) not in sys.path:
    sys.path.append(os.path.dirname(os.getcwd()))

from scipy.spatial.transform import Rotation

rad2deg = 180.0/pi
deg2rad = pi/180.0

# Set PID Gains and Max Values
# ---------------------------

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

            # Max Rate [degrees/s]
            "pMax" : 200.0,
            "qMax" : 200.0,
            "rMax" : 150.0,

             # Minimum velocity for yaw follow to kick in [m/s]
            "minTotalVel_YawFollow" : 0.1,

            "useIntegral" : True    # Include integral gains in linear velocity control
            }

class Controller:
    
    def __init__(self, quad_params, orient="NED", params=ctrl_params):

        self.quad_params = quad_params

        self.saturateVel_separately = params["saturateVel_separately"]

        self.useIntegral = params["useIntegral"]

        self.minTotalVel_YawFollow = params["minTotalVel_YawFollow"]

        # Max tilt
        self.tiltMax = params["tiltMax"]*deg2rad

        # Max Rate
        self.pMax = params["pMax"]*deg2rad
        self.qMax = params["qMax"]*deg2rad
        self.rMax = params["rMax"]*deg2rad

        self.pos_P_gain = np.array([params["Px"], params["Py"], params["Pz"]])

        self.vel_P_gain = np.array([params["Pxdot"], params["Pydot"], params["Pzdot"]])
        self.vel_D_gain = np.array([params["Dxdot"], params["Dydot"], params["Dzdot"]])
        self.vel_I_gain = np.array([params["Ixdot"], params["Iydot"], params["Izdot"]])

        self.att_P_gain = np.array([params["Pphi"], params["Ptheta"], params["Ppsi"]]) # Pitch, Roll, Yaw

        self.rate_P_gain = np.array([params["Pp"], params["Pq"], params["Pr"]])
        self.rate_D_gain = np.array([params["Dp"], params["Dq"], params["Dr"]])

        self.velMax = np.array([params["uMax"], params["vMax"], params["wMax"]])
        self.velMaxAll = self.velMax.min()

        self.rateMax = np.array([self.pMax, self.qMax, self.rMax])

        self.thr_int = np.zeros(3)
        self.pos_sp        = np.zeros(3)
        self.vel_sp        = np.zeros(3)
        self.acc_sp        = np.zeros(3)
        self.thrust_sp     = np.zeros(3)
        self.thrust_rep_sp = np.zeros(3)
        self.yawFF         = 0.

        self.prev_heading_sp = None

        self.drone_z = np.zeros(3)

        if (orient == "NED"):
            self.z_mul = -1
            # Get thrust limits (Z)   
            # The Thrust limits are negated and swapped due to NED-frame
            self.maxThr_Z = -self.quad_params["minThr"]
            self.minThr_Z = -self.quad_params["maxThr"]
        else:
            self.z_mul = 1
            # Get thrust limits (Z)
            self.maxThr_Z = self.quad_params["maxThr"]
            self.minThr_Z = self.quad_params["minThr"] 

    
    def control(self, Ts, ctrlType, yawType, 
                      pos_sp, vel_sp, acc_sp, thrust_sp, yaw_sp, yawFF,
                      pos, vel, vel_dot, quat, omega, omega_dot, psi,
                      F_rep=np.zeros(3), pfVel=0, pfSatFor=0, pfFor=0):

        self.pos = pos
        self.vel = vel
        self.vel_dot = vel_dot
        self.quat = quat
        self.omega = omega
        self.omega_dot = omega_dot
        self.psi = psi

        # Desired State
        # ----------------
        self.pos_sp[:]    = pos_sp[:] 
        self.vel_sp[:]    = vel_sp[:] 
        self.acc_sp[:]    = acc_sp[:] 
        self.thrust_sp[:] = thrust_sp[:] 
        self.yaw_sp       = yaw_sp 
        self.yawFF        = yawFF

        if self.prev_heading_sp == None:
            self.prev_heading_sp = self.psi

        # Rotation Matrix of current state (Direct Cosine Matrix)
        self.quat_rot = Rotation.from_quat(self.quat[[1,2,3,0]])
        dcm = self.quat_rot.as_matrix()

        # Current thrust (drone) orientation
        self.drone_z = dcm[:,2]


        if not yawType:
            # Leave Yaw "loose"
            self.att_P_gain[2] = 0
        self.setYawWeight()

        # Select Controller
        # ---------------------------
        # The controller first calculates a thrust vector using the world frame.
        # Then it tries to align the drone to that vector.
        if (ctrlType == "xyz_vel"):
            self.saturateVel()
            self.z_vel_control(pfSatFor, F_rep, Ts)
            self.xy_vel_control(pfSatFor, F_rep, Ts)
            self.thrustToAttitude(pfFor, F_rep)
            self.attitude_control()
            self.rate_control()
        elif (ctrlType == "xy_vel_z_pos"):
            self.z_pos_control()
            self.saturateVel()
            self.z_vel_control(pfSatFor, F_rep, Ts)
            self.xy_vel_control(pfSatFor, F_rep, Ts)
            self.thrustToAttitude(pfFor, F_rep)
            self.attitude_control()
            self.rate_control()
        elif (ctrlType == "xyz_pos"):
            self.z_pos_control()
            self.xy_pos_control()
            self.saturateVel() # to give more authority to F_rep
            self.addFrepToVel(pfVel, F_rep)
            self.saturateVel()
            if (yawType == "follow"):
                # set Yaw setpoint and Yaw Rate Feed-Forward to follow the velocity setpoint
                self.yaw_follow(Ts)
            self.z_vel_control(pfSatFor, F_rep, Ts)
            self.xy_vel_control(pfSatFor, F_rep, Ts)
            self.thrustToAttitude(pfFor, F_rep)
            self.attitude_control()
            self.rate_control()


    def z_pos_control(self):
       
        # Z Position Control
        # --------------------------- 
        pos_z_error = self.pos_sp[2] - self.pos[2]
        self.vel_sp[2] += self.pos_P_gain[2]*pos_z_error
        
    
    def xy_pos_control(self):

        # XY Position Control
        # --------------------------- 
        pos_xy_error = (self.pos_sp[0:2] - self.pos[0:2])
        self.vel_sp[0:2] += self.pos_P_gain[0:2]*pos_xy_error
        
        
    def saturateVel(self):

        # Saturate Velocity Setpoint
        # --------------------------- 
        # Either saturate each velocity axis separately, or total velocity (preferred)
        if (self.saturateVel_separately):
            self.vel_sp = np.clip(self.vel_sp, -self.velMax, self.velMax)
        else:
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > self.velMaxAll):
                self.vel_sp = self.vel_sp/totalVel_sp*self.velMaxAll
    
    
    def addFrepToVel(self, pfVel, F_rep):

        # Increase F_rep proportional to velocity towards the obstacle
        # ...and F_rep magnitude.
        if pfVel:
            veldotf = np.dot(self.vel, F_rep)
            F_rep_mult = -veldotf if veldotf < 0 else 0

            # Add repulsive force "velocity" to velocity setpoint
            self.vel_sp += F_rep*F_rep_mult


    def yaw_follow(self, Ts):
        
        # Generate Yaw setpoint and FF
        # ---------------------------
            totalVel_sp = norm(self.vel_sp)
            if (totalVel_sp > self.minTotalVel_YawFollow):
                # Calculate desired Yaw
                self.yaw_sp = np.arctan2(self.vel_sp[1], self.vel_sp[0])
            
                # Dirty hack, detect when desEul[2] switches from -pi to pi (or vice-versa) and switch manualy prev_heading_sp 
                if (np.sign(self.yaw_sp) - np.sign(self.prev_heading_sp) and abs(self.yaw_sp-self.prev_heading_sp) >= 2*pi-0.1):
                    self.prev_heading_sp = self.prev_heading_sp + np.sign(self.yaw_sp)*2*pi
            
                # Angle between current vector with the next heading vector
                delta_psi = self.yaw_sp - self.prev_heading_sp
            
                # Set Yaw rate
                self.yawFF = delta_psi / Ts 

                # Prepare next iteration
                self.prev_heading_sp = self.yaw_sp


    def z_vel_control(self, pfSatFor, F_rep, Ts):
        
        # Z Velocity Control (Thrust in D-direction)
        # ---------------------------
        # Hover thrust (m*g) is sent as a Feed-Forward term, in order to 
        # allow hover when the position and velocity error are nul
        vel_z_error = self.vel_sp[2] - self.vel[2]

        thrust_z_sp = (self.vel_P_gain[2]*vel_z_error - self.vel_D_gain[2]*self.vel_dot[2] +
                       self.quad_params["mB"]*(self.acc_sp[2] + self.z_mul*self.quad_params['g']) +
                       self.thr_int[2])

        thrust_z_sp += pfSatFor*F_rep[2]
    

        # Apply Anti-Windup in D-direction
        stop_int_D = (thrust_z_sp >= self.maxThr_Z and vel_z_error >= 0.0) or (thrust_z_sp <= self.minThr_Z and vel_z_error <= 0.0)

        # Calculate integral part
        if not (stop_int_D):
            self.thr_int[2] += self.vel_I_gain[2]*vel_z_error*Ts * self.useIntegral
            # Limit thrust integral
            self.thr_int[2] = min(abs(self.thr_int[2]), self.quad_params["maxThr"])*np.sign(self.thr_int[2])

        # Saturate thrust setpoint in D-direction
        self.thrust_sp[2] = np.clip(thrust_z_sp, self.minThr_Z, self.maxThr_Z)

    
    def xy_vel_control(self, pfSatFor, F_rep, Ts):
        
        # XY Velocity Control (Thrust in NE-direction)
        # ---------------------------
        vel_xy_error = self.vel_sp[0:2] - self.vel[0:2]
        thrust_xy_sp = (self.vel_P_gain[0:2]*vel_xy_error - self.vel_D_gain[0:2]*self.vel_dot[0:2] + 
                        self.quad_params["mB"]*(self.acc_sp[0:2]) + self.thr_int[0:2])

        thrust_xy_sp += pfSatFor*F_rep[0:2]

        # Max allowed thrust in NE based on tilt and excess thrust
        thrust_max_xy_tilt = abs(self.thrust_sp[2])*np.tan(self.tiltMax)
        thrust_max_xy = sqrt(self.quad_params["maxThr"]**2 - self.thrust_sp[2]**2)
        thrust_max_xy = min(thrust_max_xy, thrust_max_xy_tilt)

        # Saturate thrust in NE-direction
        self.thrust_sp[0:2] = thrust_xy_sp
        if (np.dot(self.thrust_sp[0:2].T, self.thrust_sp[0:2]) > thrust_max_xy**2):
            mag = norm(self.thrust_sp[0:2])
            self.thrust_sp[0:2] = thrust_xy_sp/mag*thrust_max_xy
        
        # Use tracking Anti-Windup for NE-direction: during saturation, the integrator is used to unsaturate the output
        # see Anti-Reset Windup for PID controllers, L.Rundqwist, 1990 - https://doi.org/10.1016/S1474-6670(17)51865-0
        arw_gain = 2.0/self.vel_P_gain[0:2]
        vel_err_lim = vel_xy_error - (thrust_xy_sp - self.thrust_sp[0:2])*arw_gain
        self.thr_int[0:2] += self.vel_I_gain[0:2]*vel_err_lim*Ts * self.useIntegral
    

    def thrustToAttitude(self, pfFor, F_rep):
        # Create Full Desired Quaternion Based on Thrust Setpoint and Desired Yaw Angle
        # ---------------------------

        # Add potential field repulsive force to Thrust setpoint
        self.thrust_rep_sp = self.thrust_sp + pfFor*F_rep

        # Desired body_z axis direction
        body_z = self.z_mul*(self.thrust_rep_sp/norm(self.thrust_rep_sp))
        
        # Vector of desired Yaw direction in XY plane, rotated by pi/2 (fake body_y axis)
        y_C = np.array([-sin(self.yaw_sp), cos(self.yaw_sp), 0.0])
        
        # Desired body_x axis direction
        body_x = np.cross(y_C, body_z)
        
        # Desired body_y axis direction
        body_y = np.cross(body_z, body_x)

        # Desired rotation matrix
        R_sp = np.array([body_x, body_y, body_z]).T

        # Full desired quaternion (full because it considers the desired Yaw angle)
        self.qd_full = Rotation.from_matrix(R_sp).as_quat()[[3,0,1,2]]
        self.qd_full_rot = Rotation.from_matrix(R_sp)
        
        
    def attitude_control(self):

        # Desired thrust orientation
        drone_z_desired = self.z_mul*(self.thrust_rep_sp/norm(self.thrust_rep_sp))

        # Quaternion error between the 2 vectors
        qe_red = np.zeros(4)
        qe_red[0] = np.dot(self.drone_z, drone_z_desired) + sqrt(norm(self.drone_z)**2 * norm(drone_z_desired)**2)
        qe_red[1:4] = np.cross(self.drone_z, drone_z_desired)
        qe_red = (qe_red/norm(qe_red))
        qe_red_rot = Rotation.from_quat(qe_red[[1,2,3,0]])
        
        # Reduced desired quaternion (reduced because it doesn't consider the desired Yaw angle)
        qd_red_rot = qe_red_rot * self.quat_rot

        # Mixed desired quaternion (between reduced and full) and resulting desired quaternion qd
        q_mix = (qd_red_rot.inv() * self.qd_full_rot).as_quat()[[3,0,1,2]]
        q_mix = q_mix*np.sign(q_mix[0])
        q_mix[0] = np.clip(q_mix[0], -1.0, 1.0)
        q_mix[3] = np.clip(q_mix[3], -1.0, 1.0)

        temp_rot = Rotation.from_quat(np.array([0, 0, sin(self.yaw_w*np.arcsin(q_mix[3])), cos(self.yaw_w*np.arccos(q_mix[0]))])) 
        
        # Resulting error quaternion
        quat_inv_rot = self.quat_rot.inv()
        qe = (quat_inv_rot * (qd_red_rot * temp_rot)).as_quat()[[3,0,1,2]]

        # Create rate setpoint from quaternion error
        self.rate_sp = (2.0*np.sign(qe[0])*qe[1:4])*self.att_P_gain
        
        # Limit yawFF
        self.yawFF = np.clip(self.yawFF, -self.rateMax[2], self.rateMax[2])

        # Add Yaw rate feed-forward
        self.rate_sp += quat_inv_rot.as_matrix()[:,2]*self.yawFF

        # Limit rate setpoint
        self.rate_sp = np.clip(self.rate_sp, -self.rateMax, self.rateMax)


    def rate_control(self):
        
        # Rate Control
        # ---------------------------
        rate_error = self.rate_sp - self.omega
        self.rateCtrl = self.rate_P_gain*rate_error - self.rate_D_gain*self.omega_dot     # Be sure it is right sign for the D part
        

    def setYawWeight(self):
        
        # Calculate weight of the Yaw control gain
        roll_pitch_gain = 0.5*(self.att_P_gain[0] + self.att_P_gain[1])
        self.yaw_w = np.clip(self.att_P_gain[2]/roll_pitch_gain, 0.0, 1.0)

        self.att_P_gain[2] = roll_pitch_gain

    @staticmethod
    def mixerFM(thr, moment, mixerFMinv, minWmotor, maxWmotor):
        # Mixer (generates motor speeds)
        t = np.array([thr, moment[0], moment[1], moment[2]])
        w_cmd = np.sqrt(np.clip(np.dot(mixerFMinv, t), minWmotor**2, maxWmotor**2)) # clip before sqrt to avoid negative numbers
        return w_cmd

    def getMotorSpeeds(self):
        thurst_drone_z = norm(self.thrust_rep_sp)
        moments_drone = self.quad_params["g"]*np.dot(self.quad_params["IB"], self.rateCtrl)

        w_cmd = self.mixerFM(thurst_drone_z, moments_drone, 
                             self.quad_params["mixerFMinv"], 
                             self.quad_params["minWmotor"], 
                             self.quad_params["maxWmotor"])

        return w_cmd
