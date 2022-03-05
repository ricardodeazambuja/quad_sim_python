# -*- coding: utf-8 -*-
"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import sys, os
curr_path = os.getcwd()
if os.path.basename(curr_path) not in sys.path:
    sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from numpy import sin, cos, pi, sign
from scipy.integrate import ode
from scipy.spatial.transform import Rotation
from numpy.linalg import inv

deg2rad = pi/180.0


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

class Quadcopter:

    def __init__(self, Ti, init_states=[0,0,0,0,0,0,0,0,0,0,0,0], orient = "NED", params = quad_params):
        # init_states: x0, y0, z0, phi0, theta0, psi0, xdot, ydot, zdot, p, q, r

        if (orient == "NED"):
            self.z_mul = -1
        else:
            self.z_mul = 1

        
        # Quad Params
        # ---------------------------
        self.params = params
        self.params["mixerFM"]    = self.makeMixerFM() # Make mixer that calculated Thrust (F) and moments (M) as a function on motor speeds
        self.params["mixerFMinv"] = inv(self.params["mixerFM"])
        
        # Command for initial stable hover
        # ---------------------------
        ini_hover = self.init_cmd(self.params)
        self.params["FF"] = ini_hover[0]         # Feed-Forward Command for Hover
        self.params["w_hover"] = ini_hover[1]    # Motor Speed for Hover
        self.params["thr_hover"] = ini_hover[2]  # Motor Thrust for Hover  
        self.thr = np.ones(4)*ini_hover[2]
        self.tor = np.ones(4)*ini_hover[3]

        # Initial State
        # ---------------------------
        self.state = self.init_state(init_states)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13] # body angular velocities
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])
        self.vel_dot = np.zeros(3)
        self.omega_dot = np.zeros(3)
        self.acc = np.zeros(3)

        self.extended_state()
        self.forces()

        # Set Integrator - https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.ode.html
        # ---------------------------
        self.integrator = ode(self.state_dot).set_integrator('dopri5', first_step='0.00005', atol='10e-6', rtol='10e-6')
        self.integrator.set_initial_value(self.state, Ti)


    def init_cmd(self, params):
        mB = params["mB"]
        g = params["g"]
        kTh = params["kTh"]
        kTo = params["kTo"]
        c1 = params["motorc1"]
        c0 = params["motorc0"]
        
        # w = cmd*c1 + c0   and   m*g/4 = kTh*w^2   and   torque = kTo*w^2
        thr_hover = mB*g/4.0
        w_hover   = np.sqrt(thr_hover/kTh)
        tor_hover = kTo*w_hover*w_hover
        cmd_hover = (w_hover-c0)/c1

        return [cmd_hover, w_hover, thr_hover, tor_hover]


    def init_state(self, init_states):
        
        x0, y0, z0, phi0, theta0, psi0, xdot, ydot, zdot, p, q, r = init_states

        # Here it's used w x y z for quaternions
        quat = Rotation.from_euler('xyz', [phi0, theta0, psi0]).as_quat()[[3,0,1,2]]
        
        # z0 = -self.z_mul*z0
        # zdot = -self.z_mul*zdot

        s = np.zeros(21)
        s[0]  = x0       # x
        s[1]  = y0       # y
        s[2]  = z0       # z
        s[3]  = quat[0]  # q0
        s[4]  = quat[1]  # q1
        s[5]  = quat[2]  # q2
        s[6]  = quat[3]  # q3
        s[7]  = xdot
        s[8]  = ydot
        s[9]  = zdot
        s[10] = p # body angular velocities
        s[11] = q # body angular velocities
        s[12] = r # body angular velocities

        w_hover = self.params["w_hover"] # Hovering motor speed
        wdot_hover = 0.                  # Hovering motor acc

        s[13] = w_hover
        s[14] = wdot_hover
        s[15] = w_hover
        s[16] = wdot_hover
        s[17] = w_hover
        s[18] = wdot_hover
        s[19] = w_hover
        s[20] = wdot_hover
        
        return s

    def makeMixerFM(self):
        dxm = self.params["dxm"]
        dym = self.params["dym"]
        kTh = self.params["kTh"]
        kTo = self.params["kTo"] 

        # Motor 1 is front left, then clockwise numbering.
        # A mixer like this one allows to find the exact RPM of each motor 
        # given a desired thrust and desired moments.
        # Inspiration for this mixer (or coefficient matrix) and how it is used : 
        # https://link.springer.com/article/10.1007/s13369-017-2433-2
        mixerFM = np.array([[                kTh,                 kTh,                kTh,                kTh],
                            [            dym*kTh,            -dym*kTh,           -dym*kTh,            dym*kTh],
                            [-self.z_mul*dxm*kTh, -self.z_mul*dxm*kTh, self.z_mul*dxm*kTh, self.z_mul*dxm*kTh],
                            [     self.z_mul*kTo,     -self.z_mul*kTo,     self.z_mul*kTo,    -self.z_mul*kTo]])
        
        
        return mixerFM

    def extended_state(self):
        # Euler angles of current state
        self.euler = Rotation.from_quat(self.quat[[1,2,3,0]]).as_euler('xyz')
        self.theta = self.euler[1] # around X => Roll
        self.phi   = self.euler[2] # around Y => Pitch
        self.heading = self.psi = self.euler[0] # around Z => Yaw
        # https://en.wikipedia.org/wiki/Euler_angles#Tait%E2%80%93Bryan_angles

    
    def forces(self):
        
        # Rotor thrusts and torques
        self.thr = self.params["kTh"]*self.wMotor*self.wMotor
        self.tor = self.params["kTo"]*self.wMotor*self.wMotor

    def state_dot(self, t, state, cmd, wind):

        # Import Params
        # ---------------------------    
        mB   = self.params["mB"]
        g    = self.params["g"]
        dxm  = self.params["dxm"]
        dym  = self.params["dym"]
        IB   = self.params["IB"]
        IBxx = IB[0,0]
        IByy = IB[1,1]
        IBzz = IB[2,2]
        Cd   = self.params["Cd"]
        
        kTh  = self.params["kTh"]
        kTo  = self.params["kTo"]
        tau  = self.params["tau"]
        kp   = self.params["kp"]
        damp = self.params["damp"]
        minWmotor = self.params["minWmotor"]
        maxWmotor = self.params["maxWmotor"]

        IRzz = self.params["IRzz"]
        if (self.params["usePrecession"]):
            uP = 1
        else:
            uP = 0
    
        # Import State Vector
        # ---------------------------  
        x      = state[0]
        y      = state[1]
        z      = state[2]
        q0     = state[3]
        q1     = state[4]
        q2     = state[5]
        q3     = state[6]
        xdot   = state[7]
        ydot   = state[8]
        zdot   = state[9]
        p      = state[10]
        q      = state[11]
        r      = state[12]
        wM1    = state[13]
        wdotM1 = state[14]
        wM2    = state[15]
        wdotM2 = state[16]
        wM3    = state[17]
        wdotM3 = state[18]
        wM4    = state[19]
        wdotM4 = state[20]

        # Motor Dynamics and Rotor forces (Second Order System: https://apmonitor.com/pdc/index.php/Main/SecondOrderSystems)
        # ---------------------------
        wddotM1 = (-2.0*damp*tau*wdotM1 - wM1 + kp*cmd[0])/(tau**2)
        wddotM2 = (-2.0*damp*tau*wdotM2 - wM2 + kp*cmd[1])/(tau**2)
        wddotM3 = (-2.0*damp*tau*wdotM3 - wM3 + kp*cmd[2])/(tau**2)
        wddotM4 = (-2.0*damp*tau*wdotM4 - wM4 + kp*cmd[3])/(tau**2)
    
        wMotor = np.array([wM1, wM2, wM3, wM4])
        wMotor = np.clip(wMotor, minWmotor, maxWmotor)
        thrust = kTh*wMotor*wMotor
        torque = kTo*wMotor*wMotor
    
        ThrM1 = thrust[0]
        ThrM2 = thrust[1]
        ThrM3 = thrust[2]
        ThrM4 = thrust[3]
        TorM1 = torque[0]
        TorM2 = torque[1]
        TorM3 = torque[2]
        TorM4 = torque[3]

        # Wind Model
        # ---------------------------
        velW = wind[0] # [m/s]
        qW1  = wind[1] # Wind heading [degrees]
        qW2  = wind[2] # Wind elevation (positive = upwards wind in NED, positive = downwards wind in ENU) [degrees]
    
        # State Derivatives (from PyDy) This is already the analytically solved vector of MM*x = RHS
        # ---------------------------
        DynamicsDot = np.zeros(13)
        DynamicsDot[0] = xdot
        DynamicsDot[1] = ydot
        DynamicsDot[2] = zdot
        DynamicsDot[3] = -0.5*p*q1 - 0.5*q*q2 - 0.5*q3*r
        DynamicsDot[4] =  0.5*p*q0 - 0.5*q*q3 + 0.5*q2*r
        DynamicsDot[5] =  0.5*p*q3 + 0.5*q*q0 - 0.5*q1*r
        DynamicsDot[6] = -0.5*p*q2 + 0.5*q*q1 + 0.5*q0*r
        DynamicsDot[7] = (Cd*sign(velW*cos(qW1)*cos(qW2) - xdot)*(velW*cos(qW1)*cos(qW2) - xdot)**2 + self.z_mul*2*(q0*q2 + q1*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB
        DynamicsDot[8] = (Cd*sign(velW*sin(qW1)*cos(qW2) - ydot)*(velW*sin(qW1)*cos(qW2) - ydot)**2 - self.z_mul*2*(q0*q1 - q2*q3)*(ThrM1 + ThrM2 + ThrM3 + ThrM4))/mB
        DynamicsDot[9] = (-Cd*sign(velW*sin(qW2) + zdot)*(velW*sin(qW2) + zdot)**2 + self.z_mul*(ThrM1 + ThrM2 + ThrM3 + ThrM4)*(q0**2 - q1**2 - q2**2 + q3**2) - self.z_mul*g*mB)/mB
        # uP activates or deactivates the use of gyroscopic precession.
        # Set uP to False if rotor inertia is not known (gyro precession has negigeable effect on drone dynamics)
        DynamicsDot[10] = ((IByy - IBzz)*q*r + self.z_mul*uP*IRzz*(wM1 - wM2 + wM3 - wM4)*q + ( ThrM1 - ThrM2 - ThrM3 + ThrM4)*dym)/IBxx
        DynamicsDot[11] = ((IBzz - IBxx)*p*r - self.z_mul*uP*IRzz*(wM1 - wM2 + wM3 - wM4)*p + self.z_mul*(-ThrM1 - ThrM2 + ThrM3 + ThrM4)*dxm)/IByy
        DynamicsDot[12] = ((IBxx - IBzz)*p*q + self.z_mul*(TorM1 - TorM2 + TorM3 - TorM4))/IBzz    
    
        # State Derivative Vector
        # ---------------------------
        sdot     = np.zeros([21])
        sdot[0]  = DynamicsDot[0]
        sdot[1]  = DynamicsDot[1]
        sdot[2]  = DynamicsDot[2]
        sdot[3]  = DynamicsDot[3]
        sdot[4]  = DynamicsDot[4]
        sdot[5]  = DynamicsDot[5]
        sdot[6]  = DynamicsDot[6]
        sdot[7]  = DynamicsDot[7]
        sdot[8]  = DynamicsDot[8]
        sdot[9]  = DynamicsDot[9]
        sdot[10] = DynamicsDot[10]
        sdot[11] = DynamicsDot[11]
        sdot[12] = DynamicsDot[12]
        sdot[13] = wdotM1
        sdot[14] = wddotM1
        sdot[15] = wdotM2
        sdot[16] = wddotM2
        sdot[17] = wdotM3
        sdot[18] = wddotM3
        sdot[19] = wdotM4
        sdot[20] = wddotM4

        self.acc = sdot[7:10]

        return sdot

    def update(self, t, Ts, cmd, wind=[0,0,0]):
        # cmd: Motor rotation speed (rad/s)

        prev_vel   = self.vel
        prev_omega = self.omega

        self.integrator.set_f_params(cmd, wind)
        self.state = self.integrator.integrate(t, t+Ts)

        self.pos   = self.state[0:3]
        self.quat  = self.state[3:7]
        self.vel   = self.state[7:10]
        self.omega = self.state[10:13]
        self.wMotor = np.array([self.state[13], self.state[15], self.state[17], self.state[19]])

        self.vel_dot = (self.vel - prev_vel)/Ts
        self.omega_dot = (self.omega - prev_omega)/Ts

        self.extended_state()
        self.forces()
