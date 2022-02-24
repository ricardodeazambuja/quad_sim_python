# -*- coding: utf-8 -*-
"""
author: John Bass
email: john.bobzwik@gmail.com
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np


def mixerFM(thr, moment, mixerFMinv, minWmotor, maxWmotor):
    t = np.array([thr, moment[0], moment[1], moment[2]])
    w_cmd = np.sqrt(np.clip(np.dot(mixerFMinv, t), minWmotor**2, maxWmotor**2)) # clip before sqrt to avoid negative numbers

    return w_cmd


## Under here is the conventional type of mixer

# def mixer(throttle, pCmd, qCmd, rCmd, quad):
#     maxCmd = quad.params["maxCmd"]
#     minCmd = quad.params["minCmd"]

#     cmd = np.zeros([4, 1])
#     cmd[0] = throttle + pCmd + qCmd - rCmd
#     cmd[1] = throttle - pCmd + qCmd + rCmd
#     cmd[2] = throttle - pCmd - qCmd - rCmd
#     cmd[3] = throttle + pCmd - qCmd + rCmd
    
#     cmd[0] = min(max(cmd[0], minCmd), maxCmd)
#     cmd[1] = min(max(cmd[1], minCmd), maxCmd)
#     cmd[2] = min(max(cmd[2], minCmd), maxCmd)
#     cmd[3] = min(max(cmd[3], minCmd), maxCmd)
    
#     # Add Exponential to command
#     # ---------------------------
#     cmd = expoCmd(quad.params, cmd)

#     return cmd

# def expoCmd(params, cmd):
#     if params["ifexpo"]:
#         cmd = np.sqrt(cmd)*10
    
#     return cmd

# def expoCmdInv(params, cmd):
#     if params["ifexpo"]:
#         cmd = (cmd/10)**2
    
#     return cmd
