#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <geometry-utils-lib/xform.h>
#include <geometry_msgs/Twist.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

static geometry_msgs::Twist rodrigues;
static transforms::Xformd   X_UAV_CAM;

void odomCallback(const nav_msgs::OdometryConstPtr& msg)
{
    // Get rvec: SHP -> CAM (passive)
    // Message gives (robotics convention) passive transform from UAV->SHP,
    // which looks like active transform from SHP->UAV, so will be used directly
    // to construct Xform: SHP->UAV
    transforms::Quatd q_UAV_SHP =
            transforms::Quatd(Vector4d(msg->pose.pose.orientation.w,
                                       msg->pose.pose.orientation.x,
                                       msg->pose.pose.orientation.y,
                                       msg->pose.pose.orientation.z));
    Vector3d t_SHP2UAV_SHP = Vector3d(msg->pose.pose.position.x,
                                      msg->pose.pose.position.y,
                                      msg->pose.pose.position.z);
    transforms::Xformd X_SHP_UAV = transforms::Xformd(t_SHP2UAV_SHP, q_UAV_SHP);
    transforms::Xformd X_SHP_CAM = X_SHP_UAV * X_UAV_CAM;
    cv::Mat R;
    cv::eigen2cv(X_SHP_CAM.q().R(), R);
    cv::Vec3d r;
    cv::Rodrigues(R, r);
//    Vector3d t = X_SHP_CAM.H().block<3, 1>(0, 3);
    Vector3d t = X_SHP_CAM.inverse().t();
//    std::cout << "[VB] t: " << t.transpose() << std::endl;
//    std::cout << "[VB] r: " << r.t() << std::endl;
    rodrigues.linear.x = t(0);
    rodrigues.linear.y = t(1);
    rodrigues.linear.z = t(2);
    rodrigues.angular.x = r(1);
    rodrigues.angular.y = r(2);
    rodrigues.angular.z = r(3);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "vision_system_bridge");
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_("~");

    std::vector<double> T_UAV_CAM_coeffs = std::vector<double>(6);
    ROS_ASSERT(nh_private_.getParam("T_UAV_CAM", T_UAV_CAM_coeffs));
    X_UAV_CAM = transforms::Xformd(Vector3d(T_UAV_CAM_coeffs[0],
                                            T_UAV_CAM_coeffs[1],
                                            T_UAV_CAM_coeffs[2]),
              transforms::Quatd::from_euler(T_UAV_CAM_coeffs[3],
                                            T_UAV_CAM_coeffs[4],
                                            T_UAV_CAM_coeffs[5]));

    ros::Subscriber odom_sub = nh_.subscribe("rel_odometry", 1, odomCallback);
    ros::Publisher  rodr_pub = nh_.advertise<geometry_msgs::Twist>("rodrigues_guess", 1);
    ros::Rate loop_rate(10);

    while (ros::ok())
    {
        rodr_pub.publish(rodrigues);
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
