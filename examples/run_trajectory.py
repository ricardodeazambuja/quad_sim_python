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

from quad_sim_python import Trajectory
from quad_sim_python import PotField
from quad_sim_python import Controller
from quad_sim_python import Quadcopter
import quad_sim_python.disp as disp

ORIENT = "ENU"

DEG2RAD = np.pi/180.0

sim_hz = []


def makeWaypoints(init_pose, wp, yaw, total_time=5):
    
    wp = np.vstack((init_pose[:3], wp)).astype(float)
    yaw = np.hstack((init_pose[-1], yaw)).astype(float)*DEG2RAD

    # For pos_waypoint_arrived_wait, this time will be the 
    # amount of time waiting
    t = np.linspace(0, total_time, wp.shape[0])
    dist = np.sum([((i-e)**2)**0.5 for i,e in zip(wp[:-1],wp[1:])])
    v_average = dist/total_time

    return t, wp, yaw, v_average
    

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
    Tf = 100 # max sim time
    
    # save the animation
    ifsave = 0

    # Choose trajectory settings
    # --------------------------- 
    ctrlOptions = ["xyz_pos", "xy_vel_z_pos", "xyz_vel"]
    trajSelect = [0]*3

    # Select Control Type             (0: xyz_pos,                  1: xy_vel_z_pos,            2: xyz_vel)
    ctrlType = ctrlOptions[0]

    # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
    #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
    #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
    #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
    #                                 12: pos_waypoint_arrived     13: pos_waypoint_arrived_wait
    trajSelect[0] = 12         

    # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow
    yawType = trajSelect[1] = "follow"           

    # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
    trajSelect[2] = 1           

    print("Control type: {}".format(ctrlType))

    # Initialize Quadcopter, Controller, Wind, Result Matrixes
    # ---------------------------
    init_pose = [10,10,10,0,0,0]
    init_twist = [0,0,10,0,0,0]
    init_states = np.hstack((init_pose,init_twist))

    wp = np.array([[2, 2, -1],
                   [-2, 3, -3],
                   [-2, -1, -3],
                   [3, -2, -1],
                   [-3, 2, -1]])

    yaw = np.array([10,
                    20, 
                   -90, 
                   120, 
                   45])
    desired_traj = makeWaypoints(init_pose, wp, yaw, total_time=20)

    potfld = PotField(pfType=1,importedData=np.zeros((0,3),dtype=float), kF=10)

    quad = Quadcopter(Ti, init_states, orient=ORIENT)
    traj = Trajectory(quad.psi, ctrlType, trajSelect, desired_traj, dist_consider_arrived=1)
    ctrl = Controller(quad.params, orient=ORIENT)


    # Trajectory for First Desired States
    # ---------------------------
    traj.desiredState(quad.pos, 0, Ts)        

    # Generate First Commands
    # ---------------------------
    desPos     = traj.sDes[0:3]
    desVel     = traj.sDes[3:6]
    desAcc     = traj.sDes[6:9]
    desThr     = traj.sDes[9:12]
    desEul     = traj.sDes[12:15]
    desPQR     = traj.sDes[15:18]
    desYawRate = traj.sDes[18]

    potfld.rep_force(quad.pos, desPos, quad.vel)

    ctrl.control(Ts, ctrlType, yawType, 
                 desPos, desVel, desAcc, desThr, desEul[2], desYawRate,
                 quad.pos, quad.vel, quad.vel_dot, quad.quat, quad.omega, quad.omega_dot, quad.psi, 
                 potfld.F_rep, potfld.pfVel, potfld.pfSatFor, potfld.pfFor)

    w_cmd = ctrl.getMotorSpeeds()

    # Initialize Result Matrixes
    # ---------------------------
    numTimeStep = int(Tf/Ts+1)

    t_all          = []
    s_all          = []
    pos_all        = []
    vel_all        = []
    quat_all       = []
    omega_all      = []
    euler_all      = []
    sDes_traj_all  = []
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
    sDes_traj_all.append(traj.sDes)
    w_cmd_all.append(w_cmd)
    wMotor_all.append(quad.wMotor)
    thr_all.append(quad.thr)
    tor_all.append(quad.tor)
    potfld_all.append(potfld.F_rep)
    fieldPointcloud.append(potfld.fieldPointcloud)

    rs = np.random.RandomState() # just add a seed for reproducibility ...
    wall = rs.rand(500,3)
    wall[:,0] = wall[:,0]*5-2.5
    wall[:,1] = 0
    wall[:,2] = -wall[:,2]*5

    # Run Simulation
    # ---------------------------
    t = Ti
    i = 1
    start_time = time.time()
    while (round(t,3) < Tf) and (i < numTimeStep) and not (all(traj.desPos == traj.wps[-1,:]) and sum(abs(traj.wps[-1,:]-quad.pos)) <= traj.dist_consider_arrived):
        t_ini = time.monotonic()

        # Dynamics (using last timestep's commands)
        # ---------------------------    
        wind = [0,0,0]# Wind('RANDOMSINE', 10.0, 3, 45, -45, 45, -45).sampleWind(t)
        quad.update(t, Ts, w_cmd, wind)
        t += Ts

        # Trajectory for Desired States 
        # ---------------------------
        traj.desiredState(quad.pos, t, Ts)        

        # Generate Commands (for next iteration)
        # ---------------------------
        desPos     = traj.sDes[0:3]
        desVel     = traj.sDes[3:6]
        desAcc     = traj.sDes[6:9] 
        desThr     = traj.sDes[9:12]
        desEul     = traj.sDes[12:15]
        desYawRate = traj.sDes[18]

        potfld = PotField(pfType=1, importedData=wall, rangeRadius=5, fieldRadius=3, kF=1)
        potfld.rep_force(quad.pos, desPos, quad.vel)

        ctrl.control(Ts, ctrlType, yawType, 
                     desPos, desVel, desAcc, desThr, desEul[2], desYawRate,
                     quad.pos, quad.vel, quad.vel_dot, quad.quat, quad.omega, quad.omega_dot, quad.psi, 
                     potfld.F_rep, potfld.pfVel, potfld.pfSatFor, potfld.pfFor)

        w_cmd = ctrl.getMotorSpeeds()
        
        # print("{:.3f}".format(t))
        t_all.append(t)
        s_all.append(quad.state)
        pos_all.append(quad.pos)
        vel_all.append(quad.vel)
        quat_all.append(quad.quat)
        omega_all.append(quad.omega)
        euler_all.append(quad.euler)
        sDes_traj_all.append(traj.sDes)
        w_cmd_all.append(w_cmd)
        wMotor_all.append(quad.wMotor)
        thr_all.append(quad.thr)
        tor_all.append(quad.tor)
        potfld_all.append(potfld.F_rep)
        fieldPointcloud.append(potfld.fieldPointcloud)
        
        i += 1
        sim_hz.append(1/(time.monotonic()-t_ini))
    
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
    w_cmd_all = np.asanyarray(w_cmd_all)
    wMotor_all = np.asanyarray(wMotor_all)
    thr_all = np.asanyarray(thr_all)
    tor_all = np.asanyarray(tor_all)
    potfld_all = np.asanyarray(potfld_all)
    fieldPointcloud = np.array(fieldPointcloud, dtype=object)

    ani = disp.sameAxisAnimation(t_all, traj.wps, pos_all, quat_all, sDes_traj_all, Ts, quad.params, 
                                  traj.xyzType, traj.yawType, ifsave, wall, potfld_all, fieldPointcloud, orient=ORIENT)

if __name__ == "__main__":
    main()