# -*- coding: utf-8 -*-
"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
import time
from numpy.linalg import norm

from potentialField import PotField
from ctrl import Controller
from quad import Quadcopter
from utils.windModel import Wind
import utils

deg2rad = np.pi/180.0

sim_hz = []

def main():

    # Simulation Setup
    # --------------------------- 
    Ti = 0 # init time

    # Testing sample periods
    # Ts = 0.0025 # 985Hz
    Ts = 0.005 # 880Hz
    # Ts = 0.0075 # 632Hz
    # Ts = 0.01 # 595Hz
    # Ts = 0.02 # 389Hz 
    # the ode solver struggles to reach the min error 
    # when Ts is too big, therefore it takes more iterations
    Tf = 30 # max sim time
    
    # save the animation
    ifsave = 0

    # Choose trajectory settings
    # --------------------------- 
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]


    # Select Control Type             (0: xyz_pos,                  1: xy_vel_z_pos,            2: xyz_vel)
    ctrlType = ctrlOptions[0]


    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
    yawType = 4


    print(f"Control type: {ctrlType}")

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    init_pose = np.array([0,10,0,0,0,0]) # in NED
    init_twist = np.array([0,0,0,0,0,0]) # in NED
    init_states = np.hstack((init_pose,init_twist))

    potfld = PotField(pfType=1,importedData=np.zeros((0,3),dtype=float))
    quad = Quadcopter(Ti, init_states)


    # Initialize trajectory setpoint
    desPos = init_pose[:3]          # Desired position (x, y, z)
    desVel = np.array([0., 0., 0.]) # Desired velocity (xdot, ydot, zdot)
    desAcc = np.array([0., 0., 0.]) # Desired acceleration (xdotdot, ydotdot, zdotdot)
    desThr = np.array([0., 0., 0.]) # Desired thrust in N-E-D directions (or E-N-U, if selected)
    desEul = np.array([0., 0., 0.]) # Desired orientation in the world frame (phi, theta, psi)
    desPQR = np.array([0., 0., 0.]) # Desired angular velocity in the body frame (p, q, r)
    desYawRate = 30.0*np.pi/180     # Desired yaw speed
    sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)

    ctrl = Controller(quad.params, yawType)
    wind = Wind('None', 2.0, 90, -15)
    potfld.rep_force(quad.pos, desPos)


    # Generate First Commands
    # ---------------------------
    ctrl.control(ctrlType, yawType, 
                 desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate,
                 quad.pos, quad.vel, quad.vel_dot, quad.quat, quad.omega, quad.omega_dot, quad.psi, 
                 potfld, Ts)
    
    # Initialize Result Matrixes
    # ---------------------------

    t_all          = []
    s_all          = []
    pos_all        = []
    vel_all        = []
    quat_all       = []
    omega_all      = []
    euler_all      = []
    sDes_traj_all  = []
    sDes_calc_all  = []
    w_cmd_all      = []
    wMotor_all     = []
    thr_all        = []
    tor_all        = []
    potfld_all     = []
    fieldPointcloud = []

    t_all.append(Ti)
    s_all.append(quad.state)
    pos_all.append(quad.pos)
    vel_all.append(quad.vel)
    quat_all.append(quad.quat)
    omega_all.append(quad.omega)
    euler_all.append(quad.euler)
    sDes_traj_all.append(sDes)
    sDes_calc_all.append(ctrl.sDesCalc)
    w_cmd_all.append(ctrl.w_cmd)
    wMotor_all.append(quad.wMotor)
    thr_all.append(quad.thr)
    tor_all.append(quad.tor)
    potfld_all.append(potfld.F_rep)
    fieldPointcloud.append(potfld.fieldPointcloud)
    
    # rs = np.random.RandomState() # just add a seed for reproducibility ...
    # wall = rs.rand(2000,3)*5-2.5
    wall = np.empty((0,3)) # no wall
    # l = 10
    # wall[:,0] = wall[:,0]*l-l/2
    # wall[:,1] = 0 #wall[:,0]*2-1
    # wall[:,2] = -(wall[:,2]*l-l/2)

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    start_time = time.time()
    final_pos = False
    min_dist = 0.3
    stop_vel = 0.1
    while (round(t,3) < Tf) and not final_pos:
        t_ini = time.monotonic()

        desPos = [0,-10,0]
        desVel = np.array([0., 0., 0.])
        desAcc = np.array([0., 0., 0.])
        desThr = np.array([0., 0., 0.])
        desEul = np.array([0., 0., 0.])
        desPQR = np.array([0., 0., 0.])
        desYawRate = 0#30.0*np.pi/180
        sDes = np.hstack((desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate)).astype(float)

        potfld = PotField(pfType=1, importedData=wall, rangeRadius=10, fieldRadius=5, kF=1)
        potfld.rep_force(quad.pos, desPos)

        # Dynamics (using last timestep's commands)
        # ---------------------------    
        quad.update(t, Ts, ctrl.w_cmd, wind=None)
        t += Ts

        # Generate Commands (for next iteration)
        # ---------------------------
        ctrl.control(ctrlType, yawType, 
                     desPos, desVel, desAcc, desThr, desEul, desPQR, desYawRate,
                     quad.pos, quad.vel, quad.vel_dot, quad.quat, quad.omega, quad.omega_dot, quad.psi, 
                     potfld, Ts)

        
        # print("{:.3f}".format(t))
        t_all.append(t)
        s_all.append(quad.state)
        pos_all.append(quad.pos)
        vel_all.append(quad.vel)
        quat_all.append(quad.quat)
        omega_all.append(quad.omega)
        euler_all.append(quad.euler)
        sDes_traj_all.append(sDes)
        sDes_calc_all.append(ctrl.sDesCalc)
        w_cmd_all.append(ctrl.w_cmd)
        wMotor_all.append(quad.wMotor)
        thr_all.append(quad.thr)
        tor_all.append(quad.tor)
        potfld_all.append(potfld.F_rep)
        fieldPointcloud.append(potfld.fieldPointcloud)
        
        i += 1
        sim_hz.append(1/(time.monotonic()-t_ini))

        final_pos = (abs(sDes[:3]-quad.pos).sum() < min_dist) and (norm(quad.vel) < stop_vel)
    
    total_time = time.time() - start_time
    print(f"Simulated {t:.2f}s in {total_time:.2f}s or {t/total_time:.2}X - sim_hz [max,min,avg]: {max(sim_hz):.4f},{min(sim_hz):.4f},{sum(sim_hz)/len(sim_hz):.4f}")

    # View Results
    # ---------------------------

    t_all = np.asanyarray(t_all)
    s_all = np.asanyarray(s_all)
    pos_all = np.asanyarray(pos_all)
    vel_all = np.asanyarray(vel_all)
    quat_all = np.asanyarray(quat_all)
    omega_all = np.asanyarray(omega_all)
    euler_all = np.asanyarray(euler_all)
    sDes_traj_all = np.asanyarray(sDes_traj_all)
    sDes_calc_all = np.asanyarray(sDes_calc_all)
    w_cmd_all = np.asanyarray(w_cmd_all)
    wMotor_all = np.asanyarray(wMotor_all)
    thr_all = np.asanyarray(thr_all)
    tor_all = np.asanyarray(tor_all)
    potfld_all = np.asanyarray(potfld_all)
    fieldPointcloud = np.array(fieldPointcloud, dtype=object)

    # wall = np.empty((0,wall.shape[1])) # no wall
    ani = utils.sameAxisAnimation(t_all, np.asanyarray([desPos]), pos_all, quat_all, sDes_traj_all, Ts, quad.params, 1, yawType, ifsave, wall, potfld_all, fieldPointcloud)

if __name__ == "__main__":
    main()