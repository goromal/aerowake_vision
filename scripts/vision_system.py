#!/usr/bin/python

import rospy, imutils, sys
import numpy as np
from math import pi
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped

from helper_functions import CentroidFinder, NoiseFilter, PnPSolver
from helper_functions import convert_image, rectify, show_image

class VisionSystem:

    def __init__(self):
        self.binary_threshold = 100 # 100
        self.flag_show_images = rospy.get_param('show_pose_img', False)
        self.flag_show_debug_images = False
        self.flag_show_debug_messages = False
        self.rotate = False

        self.cfinder = CentroidFinder(self.flag_show_debug_images,self.flag_show_debug_messages,self.binary_threshold)
        self.nfilter = NoiseFilter(self.flag_show_debug_images,self.flag_show_debug_messages,self.rotate)
        self.psolver = None

        self.transform_pub = rospy.Publisher("vision_pose", TransformStamped, queue_size=1)
        self.camera_info_sub = rospy.Subscriber("/camera/camera_info", CameraInfo, self.cameraInfoCallback)
        self.image_sub = rospy.Subscriber("/camera/image_raw", Image, self.imageCallback)

    def cameraInfoCallback(self, msg):
        mtx = np.array([msg.K[0:3],msg.K[3:6],msg.K[6:9]])
        dist = np.array(msg.D)
        self.psolver = PnPSolver(mtx, dist,self.flag_show_debug_images,self.flag_show_debug_messages,self.rotate)
        self.camera_info_sub.unregister()

    def imageCallback(self, msg):
        if not self.psolver is None:
            img = convert_image(msg, flag=self.flag_show_debug_images)
            show_image("Original Image", img, flag=self.flag_show_debug_images)

            centroids, img_cent = self.cfinder.get_centroids(img)
            # show_image("Initial Centroids", img_cent, flag=self.flag_show_images)

            centroids, img_filt = self.nfilter.filter_noise(img, centroids)
            show_image("Filtered Centroids", img_filt, flag=self.flag_show_debug_images)

            position, yawpitchroll, orientation, img_solv = self.psolver.solve_pnp(img, centroids)
            show_image("Feature Pose Extraction", img_solv, duration=1, flag=self.flag_show_images)

            if not position[0] is None:
                self.publishTransform(msg.header.stamp, position, yawpitchroll)

    def publishTransform(self, stamp, pos, ypr):
        T = TransformStamped()
        T.header.stamp = stamp

        # TODO: FIX SCALE
        T.transform.translation.x = pos[0]
        T.transform.translation.y = pos[1]
        T.transform.translation.z = pos[2]
        # TODO: FIX ROTATION
        T.transform.rotation.w = 0.0
        T.transform.rotation.x = pi * ypr[2] / 180.0
        T.transform.rotation.y = pi * ypr[1] / 180.0
        T.transform.rotation.z = pi * ypr[0] / 180.0

        self.transform_pub.publish(T)

def main(args):
    rospy.init_node("vision_system", anonymous=True)
    vs = VisionSystem()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down vision system node")

if __name__ == "__main__":
    main(sys.argv)
