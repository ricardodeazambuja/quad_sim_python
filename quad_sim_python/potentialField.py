# -*- coding: utf-8 -*-
"""
original author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""
import sys, os
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
from numpy.linalg import norm




class PotField:

    def __init__(self, importedData, rangeRadius=10, fieldRadius=3, pfType=3, kF=1):
        self.pointcloud = importedData
        self.num_points = len(self.pointcloud)

        self.rangeRadius = rangeRadius
        self.fieldRadius = fieldRadius

        self.k = kF

        self.withinRange     = np.zeros(self.num_points, dtype=bool)
        self.notWithinRange  = np.zeros(self.num_points, dtype=bool)
        self.withinField     = np.zeros(self.num_points, dtype=bool)
        self.inRangeNotField = np.zeros(self.num_points, dtype=bool)

        self.force = np.zeros(3)
        self.vel   = np.zeros(3)

        self.pfVel = 0
        self.pfSatFor = 0
        self.pfFor = 0
        # These are used in the controller
        # (see ctrl_force_field.py)
        if (pfType == 1):
            self.pfVel = 1
        elif (pfType == 2):
            self.pfSatFor = 1
        elif (pfType == 3):
            self.pfFor = 1

    
    def isWithinRange(self, curr_pos):
        # Determine which points are withing a certain range
        # to avoid computing things that are too far
        # ---------------------------

        self.withinRange = (abs(curr_pos[0]-self.pointcloud[:,0]) <= self.rangeRadius) & \
                           (abs(curr_pos[1]-self.pointcloud[:,1]) <= self.rangeRadius) & \
                           (abs(curr_pos[2]-self.pointcloud[:,2]) <= self.rangeRadius)
        self.idx_withinRange = np.where(self.withinRange)[0]
        self.notWithinRange = ~(self.withinRange)
        self.idx_notWithinRange = np.where(self.notWithinRange)[0]

    def isWithinField(self, curr_pos):
        # Determine which points inside the first range is  
        # within the Potential Field Range
        # ---------------------------
        distance = norm(self.pointcloud[self.idx_withinRange,:] - curr_pos, axis=1)
        self.distance = distance

        withinField = distance <= self.fieldRadius
        
        # Distance to closest point in Field (if there are points in the Field)
        try:
            self.distanceMin = distance.min()
        except ValueError:
            self.distanceMin = -1
        
        self.idx_withinField = self.idx_withinRange[np.where(withinField)[0]]
        self.withinField = np.zeros(self.num_points, dtype=bool)
        self.withinField[self.idx_withinField] = True

        self.idx_inRangeNotField = self.idx_withinRange[np.where(~withinField)[0]]
        self.inRangeNotField = np.zeros(self.num_points, dtype=bool)
        self.inRangeNotField[self.idx_inRangeNotField] = True

        self.fieldPointcloud = self.pointcloud[self.idx_withinField]
        self.fieldDistance = distance[np.where(withinField)[0]]

    def rep_force(self, curr_pos, des_pos, curr_vel, fperpend=False):
        self.isWithinRange(curr_pos)
        self.isWithinField(curr_pos)
 
        # Repulsive Force
        # ---------------------------
        F_rep_x = self.k*(1/self.fieldDistance - 1/self.fieldRadius)*(1/(self.fieldDistance**2))*(curr_pos[0] - self.fieldPointcloud[:,0])/self.fieldDistance
        F_rep_y = self.k*(1/self.fieldDistance - 1/self.fieldRadius)*(1/(self.fieldDistance**2))*(curr_pos[1] - self.fieldPointcloud[:,1])/self.fieldDistance
        F_rep_z = self.k*(1/self.fieldDistance - 1/self.fieldRadius)*(1/(self.fieldDistance**2))*(curr_pos[2] - self.fieldPointcloud[:,2])/self.fieldDistance

        # Rotational Field
        # ---------------------------
        # Target vector
        target_vect = des_pos - curr_pos
        target_norm = norm(target_vect)
        if abs(target_norm) > 0.000001:
            # Obstacle vectors
            obst_vect = self.fieldPointcloud - curr_pos
            
            # Influence per point
            # Angle is obtained with angle = arccos(dot(u,v)/(norm(u)*norm(v)))
            # Influence is obtained with cos(angle)
            influence = np.divide(np.dot(obst_vect, target_vect), self.fieldDistance*target_norm)
            influence = influence**2

            # Extended Potential Field/Force
            # ---------------------------
            k_influence = 1
            F_rep_x = np.multiply(F_rep_x, k_influence*np.abs(influence))
            F_rep_y = np.multiply(F_rep_y, k_influence*np.abs(influence))
            F_rep_z = np.multiply(F_rep_z, k_influence*np.abs(influence))

        self.F_rep = np.array([np.sum(F_rep_x), np.sum(F_rep_y), np.sum(F_rep_z)])

        if fperpend:
            # Turn off perpendicular perturbation if curr_vel not same direction
            veldotf = np.dot(curr_vel, self.F_rep)
            if veldotf < 0:
                # Normal plane to F_rep at curr_pos
                z = lambda x,y: (self.F_rep[0]*(x-curr_pos[0])+self.F_rep[1]*(y-curr_pos[1])-self.F_rep[2]*curr_pos[2])/-self.F_rep[2]
                # Vector on plane z
                vz = np.array([0,1,z(0,1)])-curr_pos
                # Vector perpendicular to F_rep and vz
                vp = np.cross(self.F_rep,vz)
                vpmag = norm(vp)
                if vpmag>0.1:
                    # Adds a perpendicular vector that is 1/3 of original
                    self.F_rep += (vp/vpmag)*norm(self.F_rep)/3

            else:
                self.F_rep *= 0