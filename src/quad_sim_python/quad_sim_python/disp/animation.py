# -*- coding: utf-8 -*-
"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from numpy import pi
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D # https://stackoverflow.com/a/56222305/7658422

numFrames = 8

def sameAxisAnimation(t_all, waypoints, pos_all, quat_all, sDes_tr_all, Ts, params, xyzType, yawType, ifsave, pointcloud, F_rep, fieldPointcloud, orient="NED"):

    global vector, withinfield

    x = pos_all[:,0]
    y = pos_all[:,1]
    z = pos_all[:,2]

    xDes = sDes_tr_all[:,0]
    yDes = sDes_tr_all[:,1]
    zDes = sDes_tr_all[:,2]

    x_wp = waypoints[:,0]
    y_wp = waypoints[:,1]
    z_wp = waypoints[:,2]

    x_pf = pointcloud[:,0]
    y_pf = pointcloud[:,1]
    z_pf = pointcloud[:,2]

    u = F_rep[:,0]
    v = F_rep[:,1]
    w = F_rep[:,2]

    if (orient == "NED"):
        z = -z
        zDes = -zDes
        z_wp = -z_wp
        z_pf = -z_pf

    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.view_init(elev=-6., azim=15)
    ax.dist = 8
    line1, = ax.plot([], [], [], lw=2, color='red')
    line2, = ax.plot([], [], [], lw=2, color='blue')
    line3, = ax.plot([], [], [], '--', lw=1, color='blue')

    # Setting the axes properties
    extraEachSide = 0.5
    maxRange = 0.5*np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() + extraEachSide
    mid_x = 0.5*(x.max()+x.min())
    mid_y = 0.5*(y.max()+y.min())
    mid_z = 0.5*(z.max()+z.min())
    
    ax.set_xlim3d([mid_x-maxRange, mid_x+maxRange])
    ax.set_xlabel('X')
    if (orient == "NED"):
        ax.set_ylim3d([mid_y+maxRange, mid_y-maxRange])
    elif (orient == "ENU"):
        ax.set_ylim3d([mid_y-maxRange, mid_y+maxRange])
    ax.set_ylabel('Y')
    ax.set_zlim3d([mid_z-maxRange, mid_z+maxRange])
    ax.set_zlabel('Altitude')

    titleTime = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    trajType = ''
    yawTrajType = ''

    if (xyzType == 0):
        trajType = 'Hover'
    else:
        ax.scatter(x_wp, y_wp, z_wp, color='green', alpha=1, marker = 'o', s = 25)
        ax.scatter(x_pf, y_pf, z_pf, color='orange', alpha=0.5, marker = 's', s = 15)
        # withinfield = ax.scatter([], [], [], color='red', alpha=1, marker = '*', s = 25)
        withinfield, = ax.plot([], [], [], linestyle="", marker="*", color='red', markersize=5)
        vector = ax.quiver(x[0],y[0],z[0],u[0],v[0],w[0], color='red')
        if (xyzType == 1 or xyzType == 12):
            trajType = 'Simple Waypoints'
        else:
            ax.plot(xDes, yDes, zDes, ':', lw=1.3, color='green')
            if (xyzType == 2):
                trajType = 'Simple Waypoint Interpolation'
            elif (xyzType == 3):
                trajType = 'Minimum Velocity Trajectory'
            elif (xyzType == 4):
                trajType = 'Minimum Acceleration Trajectory'
            elif (xyzType == 5):
                trajType = 'Minimum Jerk Trajectory'
            elif (xyzType == 6):
                trajType = 'Minimum Snap Trajectory'
            elif (xyzType == 7):
                trajType = 'Minimum Acceleration Trajectory - Stop'
            elif (xyzType == 8):
                trajType = 'Minimum Jerk Trajectory - Stop'
            elif (xyzType == 9):
                trajType = 'Minimum Snap Trajectory - Stop'
            elif (xyzType == 10):
                trajType = 'Minimum Jerk Trajectory - Fast Stop'
            elif (xyzType == 11):
                trajType = 'Minimum Snap Trajectory - Fast Stop'
            elif (xyzType == 13):
                trajType = 'Simple waypoint with waiting period'

    if (yawType == 0):
        yawTrajType = 'None'
    elif (yawType == 1):
        yawTrajType = 'Waypoints'
    elif (yawType == 2):
        yawTrajType = 'Interpolation'
    elif (yawType == 3):
        yawTrajType = 'Follow'
    elif (yawType == 4):
        yawTrajType = 'Zero'



    titleType1 = ax.text2D(0.95, 0.95, trajType, transform=ax.transAxes, horizontalalignment='right')
    titleType2 = ax.text2D(0.95, 0.91, 'Yaw: '+ yawTrajType, transform=ax.transAxes, horizontalalignment='right')   
    
    def updateLines(i):
        global vector

        time = t_all[i*numFrames]
        pos = pos_all[i*numFrames]
        f = F_rep[i*numFrames]
        pts = fieldPointcloud[i*numFrames]

        if len(pts):
            min_pt = pts.min(axis=0)
            max_pt = pts.max(axis=0)
            centre =  min_pt+abs(min_pt-max_pt)/2
            centre_pt = min_pt+abs(min_pt-max_pt)/2
            p1 = [centre_pt[0], centre_pt[1], max_pt[2]]
            p2 = [max_pt[0], centre_pt[1], centre_pt[2]]
            vn = np.cross(p1,p2)
            goal = np.array([x_wp[-1], y_wp[-1], z_wp[-1]])
            ray_dir = pos - goal
            ndotu = vn.dot(ray_dir)
            if abs(ndotu) > 1e-6:
                w = pos - centre_pt
                si = -vn.dot(w) / ndotu
                Psi = w + si * ray_dir + centre_pt
                pts = np.array([p1, p2, centre+vn/norm(vn), Psi])

        x = pos[0]
        y = pos[1]
        z = pos[2]

        u = f[0]
        v = f[1]
        w = f[2]

        x_from0 = pos_all[0:i*numFrames,0]
        y_from0 = pos_all[0:i*numFrames,1]
        z_from0 = pos_all[0:i*numFrames,2]
    
        dxm = params["dxm"]
        dym = params["dym"]
        dzm = params["dzm"]
        
        quat = quat_all[i*numFrames]
    
        if (orient == "NED"):
            z = -z
            z_from0 = -z_from0
            quat = np.array([quat[0], -quat[1], -quat[2], quat[3]])
            w = -w
            pts[:,2] = -pts[:,2]

    
        R = Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        motorPoints = np.array([[dxm, -dym, dzm], [0, 0, 0], [dxm, dym, dzm], [-dxm, dym, dzm], [0, 0, 0], [-dxm, -dym, dzm]])
        motorPoints = np.dot(R, np.transpose(motorPoints))
        motorPoints[0,:] += x 
        motorPoints[1,:] += y 
        motorPoints[2,:] += z 
        
        line1.set_data(motorPoints[0,0:3], motorPoints[1,0:3])
        line1.set_3d_properties(motorPoints[2,0:3])
        line2.set_data(motorPoints[0,3:6], motorPoints[1,3:6])
        line2.set_3d_properties(motorPoints[2,3:6])
        line3.set_data(x_from0, y_from0)
        line3.set_3d_properties(z_from0)
        titleTime.set_text(u"Time = {:.2f} s".format(time))

        vector.remove()
        vector = ax.quiver(x,y,z,u,v,w,length=(u**2+v**2+w**2)**0.5, color='red')

        # withinfield._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        withinfield.set_data (pts[:,0], pts[:,1])
        withinfield.set_3d_properties(pts[:,2])
        
        return line1, line2


    def ini_plot():

        line1.set_data(np.empty([1]), np.empty([1]))
        line1.set_3d_properties(np.empty([1]))
        line2.set_data(np.empty([1]), np.empty([1]))
        line2.set_3d_properties(np.empty([1]))
        line3.set_data(np.empty([1]), np.empty([1]))
        line3.set_3d_properties(np.empty([1]))

        return line1, line2, line3

        
    # Creating the Animation object
    line_ani = animation.FuncAnimation(fig, updateLines, init_func=ini_plot, frames=len(t_all[0:-2:numFrames]), interval=(Ts*1000*numFrames), blit=False)
    
    if (ifsave):
        line_ani.save('animation_{0}_{1}.gif'.format(xyzType,yawType), dpi=80, writer='imagemagick', fps=25)
        # line_ani.save('animation_{0:.0f}_{1:.0f}.mp4'.format(xyzType,yawType), writer=animation.FFMpegWriter(fps=25, bitrate=1000))
        
    plt.show()
    return line_ani