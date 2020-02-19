#!/usr/bin/python

import os, cv2
import numpy as np
from helper_functions import VisionPose
from shutil import rmtree

def main():
    binary_threshold = 100
    rotate = False
    trans_prev = None
    debug = 3
    logdir = '/home/andrew/Research/aerowake-mit/ros-packages/aerowake_vision/scripts/test_results'

    id_vals = [317, 365, 397, 402, 416, 417, 511]
    objp_coeff = [ 0.0,  0.0,  0.0, \
                        0.0, -0.161,  0.0, \
                        0.0, -0.318,  0.0, \
                        0.0, -0.476,  0.0, \
                       -0.802,    0.0, -0.256, \
                       -0.802, -0.159, -0.256, \
                       -0.802, -0.318, -0.256, \
                       -0.802, -0.472, -0.256]
    K = [1508.116662, 0, 595.527300, 0, 1508.757587, 449.996320, 0, 0, 1]
    D = [-0.352819, 0.219373, -0.001624, 0.000912, 0]
    mtx = np.array([K[0:3], K[3:6], K[6:9]])
    dist = np.array(D)
    if os.path.exists(logdir):
        rmtree(logdir)
    os.mkdir(logdir)
    # import pdb; pdb.set_trace()
    VP = VisionPose(mtx, dist, objp_coeff, debug, binary_threshold, rotate, logdir)

    for id in id_vals:
        img = cv2.imread('/home/andrew/Research/aerowake-mit/ros-packages/aerowake_vision/scripts/test_baselines/%d_[convert](original).png' % id, 0)
        pos, quat = VP.imgToPose(img, id)
        if not pos is None:
            trans = np.array([pos[0], pos[1], pos[2]])
            if not trans_prev is None:
                if np.linalg.norm(trans - trans_prev) > 5.0 and debug > 1:
                    rospy.logwarn('Whoa, experienced a large estimate jump at iteration %d!' % id)
            trans_prev = trans

if __name__ == "__main__":
    main()
