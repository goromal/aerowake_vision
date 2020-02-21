#!/usr/bin/python

import rospy, os
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from aerowake_vision.msg import VisionPose as VPmsg
from datetime import datetime
import tf.transformations as TR

from helper_functions import convert_image, VisionPose

def homogeneous_matrix(trans, quat):
    return TR.concatenate_matrices(TR.translation_matrix(trans), TR.quaternion_matrix(quat))

class VisionSystem:

    def __init__(self):
        self.binary_threshold = 100
        self.id = 0
        self.rotate = False
        self.trans_prev = np.array([-999.0, -999.0, -999.0])
        self.rate = rospy.get_param('~rate', 5.0)
        self.debug = rospy.get_param('~debug', 0) # 0 for no debug, 1 for window pop-up, 2 for debug msgs, 3 for image logging
        self.logdir = rospy.get_param('~log_dir', '/home/andrew/.ros/vision_sys_' + str(datetime.now()).replace(" ", "_").replace(":","-").replace(".","_"))

        T_u_c_coeff = rospy.get_param('~T_UAV_CAM')
        q_u_c = TR.quaternion_from_euler(T_u_c_coeff[3], T_u_c_coeff[4], T_u_c_coeff[5])
        # instead of interpreting as active transform from UAV to camera, interpreting as
        # passive transform from camera to UAV
        H_CAM_UAV = homogeneous_matrix(T_u_c_coeff[0:3], q_u_c)
        self.H_UAV_CAM = np.linalg.inv(H_CAM_UAV)

        X_s_b_coeff = rospy.get_param('~X_ship_beacon')
        q_s_b = TR.quaternion_from_euler(X_s_b_coeff[3], X_s_b_coeff[4], X_s_b_coeff[5])
        # instead of interpreting as active transform from ship to beacon, interpreting as
        # passive transform from beacon to ship
        H_BEAC_SHIP = homogeneous_matrix(X_s_b_coeff[0:3], q_s_b)
        objp_coeff_orig = rospy.get_param('~beacon_positions')
        self.objp_coeff = list()
        for i in range(0, 8):
            obj_list = objp_coeff_orig[3*i:3*i+3]
            obj_list.append(1.0)
            obj_point_BEAC = np.array(obj_list)
            obj_point_SHIP = np.dot(H_BEAC_SHIP, obj_point_BEAC)
            self.objp_coeff.append(obj_point_SHIP[0])
            self.objp_coeff.append(obj_point_SHIP[1])
            self.objp_coeff.append(obj_point_SHIP[2])

        self.transform_pub = rospy.Publisher("vision_pose", VPmsg, queue_size=1)
        self.image_sub = rospy.Subscriber("camera/image_raw", Image, self.imageCallback, queue_size=1)
        self.vecs_sub = rospy.Subscriber("rodrigues_guess", Twist, self.rodriguesCallback, queue_size=1)

        self.rvecs = None
        self.tvecs = None

        self.img = None
        self.stamp = None
        self.id_prev = -1
        # self.pos = None
        # self.quat = None
        # self.num_cent = None
        # self.solved = None
        # self.incr = None
        # self.valid = None


        K = rospy.get_param('~camera_matrix/data')
        D = rospy.get_param('~distortion_coefficients/data')
        mtx = np.array([K[0:3], K[3:6], K[6:9]])
        dist = np.array(D)
        if self.debug > 2:
            os.mkdir(self.logdir)

        self.VP = VisionPose(mtx, dist, self.objp_coeff, self.debug, self.binary_threshold, self.rotate, self.logdir)

        self.update_timer_ = rospy.Timer(rospy.Duration(1.0/self.rate), self.update)

    def imageCallback(self, msg):
        self.img = convert_image(msg, self.debug, self.id, self.logdir)
        self.stamp = msg.header.stamp
        self.id += 1

    def update(self, event):
        if not self.img is None and not (self.id == self.id_prev):
            self.id_prev = self.id
            # print 'PROCESSING FOR', self.id_prev
            pos, quat, num_cent, solved, incr, valid, num_cont = self.VP.imgToPose(self.img, self.id, self.rvecs, self.tvecs)
            self.publishTransform(self.stamp, pos, quat, num_cent, solved, incr, valid, num_cont)
            # print 'COMMAND PUBLISH FOR', self.id_prev

    # def imageCallback(self, msg):
    #     print 'CALLBACK', self.id
    #     img = convert_image(msg, self.debug, self.id, self.logdir)
    #     pos, quat, num_cent, solved, incr, valid = self.VP.imgToPose(img, self.id, self.rvecs, self.tvecs)
    #     self.publishTransform(msg.header.stamp, pos, quat, num_cent, solved, incr, valid)
    #     # if not pos is None:
    #     #     self.publishTransform(msg.header.stamp, pos, quat)
    #     self.id += 1

    def rodriguesCallback(self, msg):
        self.tvecs = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
        self.rvecs = np.array([msg.angular.x, msg.angular.y, msg.angular.z])

    def publishTransform(self, stamp, pos, quat, num_cent, solved, incr, valid, num_cont):
        # print 'PUBLISHING'
        sol_status = VPmsg.NO_SOLUTION
        if solved:
            H_SHIP_CAM = homogeneous_matrix(list(pos), quat)
            H_CAM_SHIP = np.linalg.inv(H_SHIP_CAM)
            H_UAV_SHIP = np.dot(H_CAM_SHIP, self.H_UAV_CAM)
            trans = TR.translation_from_matrix(H_UAV_SHIP)
            quatn = TR.quaternion_from_matrix(H_UAV_SHIP)
            if incr:
                sol_status = VPmsg.INCREMENTAL
            else:
                sol_status = VPmsg.DERIVED_INIT
        else:
            trans = np.array([-999.0, -999.0, -999.0])
            quatn = np.array([0, 0, 0, 1])

        outlier = False
        if not (self.trans_prev[0] == -999.0):
            if np.linalg.norm(trans - self.trans_prev) > 5.0 and self.debug > 1:
                rospy.logwarn('Whoa, experienced a large estimate jump at iteration %d!' % self.id)
                outlier = True
        self.trans_prev = trans

        T = VPmsg() # TransformStamped()
        T.header.stamp = stamp
        T.transform.translation.x = trans[0]
        T.transform.translation.y = trans[1]
        T.transform.translation.z = trans[2]
        T.transform.rotation.x = quatn[0]
        T.transform.rotation.y = quatn[1]
        T.transform.rotation.z = quatn[2]
        T.transform.rotation.w = quatn[3]
        T.sol_status = sol_status
        T.contour_count = num_cont
        T.centroid_count = num_cent
        T.outlier = outlier
        T.dynamically_valid = valid
        self.transform_pub.publish(T)

def main():
    rospy.init_node("vision_system", anonymous=True)
    vs = VisionSystem()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision system node")

if __name__ == "__main__":
    main()
