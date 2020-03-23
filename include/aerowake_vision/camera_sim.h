#pragma once

#include <ros/ros.h>
#include <random>
//#include "camera_model.h"
#include "geometry-utils-lib/xform.h"
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <visualization_msgs/Marker.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

#define NUM_BEACONS     8
#define VIS_MULTIPLIER 10
#define PIXEL_RAD       5

using namespace transforms;
using namespace Eigen;

namespace camera_sim {

class CameraSim
{
public:
    CameraSim();

private:
    void onUpdate(const ros::TimerEvent &event);
    void takePicture();

    visualization_msgs::Marker marker_;
    ros::Publisher marker_pub_;
    ros::Publisher image_pub_;
    ros::Publisher caminfo_pub_;
    sensor_msgs::CameraInfo caminfo_;

    std::vector<cv::Point3d> points_SHIP_;
    geometry_msgs::TransformStamped tf_UAV_SHIP_;
    Xformd X_UAV_CAM_;

    int img_w_;
    int img_h_;
    cv::Mat K_;
    cv::Mat D_;

    double updateRate_;
    double cameraRate_;
    unsigned int camera_counter_;
    unsigned int camera_counter_limit_;

    double pix_stdev_;
    std::default_random_engine random_generator_;
    std::normal_distribution<double> gaussian_dist_;

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::Timer timer_;
    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener tfListener_;
};

} // end namespace camera_sim
