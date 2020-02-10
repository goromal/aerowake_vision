#include "aerowake_vision/camera_sim.h"

#define TO_ULONG(i) static_cast<unsigned long>(i)

namespace camera_sim {

CameraSim::CameraSim() : camera_counter_(0), nh_(), nh_private_("~"), tfBuffer_(), tfListener_(tfBuffer_)
{
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("beacon_array_marker", 1);
    marker_.header.frame_id    = "boat";
    marker_.type               = visualization_msgs::Marker::POINTS;
    marker_.action             = visualization_msgs::Marker::ADD;
    marker_.pose.orientation.x = 0.0;
    marker_.pose.orientation.y = 0.0;
    marker_.pose.orientation.z = 0.0;
    marker_.pose.orientation.w = 1.0;
    marker_.color.r            = 1.0f;
    marker_.color.g            = 0.0f;
    marker_.color.b            = 0.0f;
    marker_.color.a            = 1.0;
    marker_.lifetime           = ros::Duration();
    marker_.scale.x            = 0.125;
    marker_.scale.y            = 0.125;

    image_pub_ = nh_.advertise<sensor_msgs::Image>("simulated_camera_image", 1);

    std::vector<double> points_BOAT = std::vector<double>(3 * NUM_BEACONS);
    ROS_ASSERT(nh_private_.getParam("beacon_positions", points_BOAT));
    for (int i = 0; i < NUM_BEACONS; i++)
    {
        double point_BOAT_x = points_BOAT[TO_ULONG(3 * i + 0)];
        double point_BOAT_y = points_BOAT[TO_ULONG(3 * i + 1)];
        double point_BOAT_Z = points_BOAT[TO_ULONG(3 * i + 2)];
        points_BOAT_.push_back(Vector3d(point_BOAT_x, point_BOAT_y, point_BOAT_Z));
        geometry_msgs::Point p;
        p.x = point_BOAT_x;
        p.y = point_BOAT_y;
        p.z = point_BOAT_Z;
        marker_.points.push_back(p);
    }

    std::vector<double> T_UAV_CAM_coeffs = std::vector<double>(6);
    ROS_ASSERT(nh_private_.getParam("T_UAV_CAM", T_UAV_CAM_coeffs));
    double x = T_UAV_CAM_coeffs[0];
    double y = T_UAV_CAM_coeffs[1];
    double z = T_UAV_CAM_coeffs[2];
    double phi = T_UAV_CAM_coeffs[3];
    double tht = T_UAV_CAM_coeffs[4];
    double psi = T_UAV_CAM_coeffs[5];
    T_UAV_CAM_ = Xformd(Vector3d(x, y, z), Quatd::from_euler(phi, tht, psi));

    ROS_ASSERT(nh_private_.getParam("image_width", img_w_));
    ROS_ASSERT(nh_private_.getParam("image_height", img_h_));
    Vector2d img_size(img_w_, img_h_);
    std::vector<double> opencv_camera_matrix_coeff = std::vector<double>(9);
    ROS_ASSERT(nh_private_.getParam("camera_matrix/data", opencv_camera_matrix_coeff));
    Vector2d cam_f(opencv_camera_matrix_coeff[0], opencv_camera_matrix_coeff[4]),
             cam_c(opencv_camera_matrix_coeff[2], opencv_camera_matrix_coeff[5]);
    // Distortionless camera model
    camera_ = Camera(cam_f, cam_c, Matrix<double, 5, 1>::Zero(), img_size);

    ROS_ASSERT(nh_private_.getParam("camera/pointgrey_camera/frame_rate", cameraRate_));
    ROS_ASSERT(cameraRate_ >= 1.0);
    camera_counter_limit_ = VIS_MULTIPLIER;
    updateRate_ = VIS_MULTIPLIER * cameraRate_;

    ROS_ASSERT(nh_private_.getParam("pixel_stdev", pix_stdev_));
    gaussian_dist_ = std::normal_distribution<double>(0.0, 1.0);

    timer_ = nh_.createTimer(ros::Duration(ros::Rate(updateRate_)), &CameraSim::onUpdate, this);
}

void CameraSim::onUpdate(const ros::TimerEvent &)
{
    // Visualize beacon array in sim
    marker_.header.stamp = ros::Time::now();
    marker_pub_.publish(marker_);

    // Handle camera timing logic
    if (camera_counter_ == camera_counter_limit_)
    {
        takePicture();
        camera_counter_ = 0;
    }
    else
        camera_counter_++;
}

void CameraSim::takePicture()
{
    // Get transform from BOAT frame to UAV frame
    try
    {
        tf_BOAT_UAV_ = tfBuffer_.lookupTransform("boat", "UAV", ros::Time(0));
        q_BOAT_UAV_ = Quatd(Vector4d(tf_BOAT_UAV_.transform.rotation.w,
                                     tf_BOAT_UAV_.transform.rotation.x,
                                     tf_BOAT_UAV_.transform.rotation.y,
                                     tf_BOAT_UAV_.transform.rotation.z));
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
        return;
    }

    // Get transform from BOAT frame to CAM frame
//    std::cout << tf_BOAT_UAV_ << std::endl; // GOOD, ACTIVE FROM BOAT TO UAV
    Xformd T_BOAT_UAV = Xformd(Vector3d(tf_BOAT_UAV_.transform.translation.x,
                                        tf_BOAT_UAV_.transform.translation.y,
                                        tf_BOAT_UAV_.transform.translation.z),
                               q_BOAT_UAV_);
//    std::cout << T_BOAT_UAV << std::endl; // GOOD, SAME THING
    Xformd T_BOAT_CAM = T_BOAT_UAV * T_UAV_CAM_;
//    std::cout << T_BOAT_CAM << std::endl;

    // Draw blank black image (https://stackoverflow.com/questions/28780947/opencv-create-new-image-using-cvmat/28782436)
    cv::Mat image(img_h_, img_w_, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Scalar white(255, 255, 255);

    // Perform projection of feature points onto the camera pixel frame and draw on image
    for (int i = 0; i < NUM_BEACONS; i++)
    {
        Vector3d point_CAM = T_BOAT_CAM.transformp(points_BOAT_[TO_ULONG(i)]);
//        std::cout << point_CAM.transpose() << std::endl;
        Vector2d pixel;
        camera_.proj(point_CAM, pixel);
        int pixel_x = static_cast<int>(pixel(0, 0));
        int pixel_y = static_cast<int>(pixel(1, 0));

        // add pixel noise
        pixel_x += static_cast<int>(pix_stdev_ * gaussian_dist_(random_generator_));
        pixel_y += static_cast<int>(pix_stdev_ * gaussian_dist_(random_generator_));

        // add to image (https://stackoverflow.com/questions/31658132/c-opencv-not-drawing-circles-on-mat-image)
        //              (https://stackoverflow.com/questions/18886083/how-fill-circles-on-opencv-c)
        if (pixel_x > PIXEL_RAD + 1 && pixel_x < img_w_ - PIXEL_RAD - 1 &&
            pixel_y > PIXEL_RAD + 1 && pixel_y < img_h_ - PIXEL_RAD - 1)
        {
            cv::Point pixel_point(pixel_x, pixel_y);
            cv::circle(image, pixel_point, PIXEL_RAD, white, CV_FILLED);
//            std::cout << "DREW AT (" << pixel_x << ", " << pixel_y << ")" << std::endl;
        }
    }

    // Use CV Bridge to publish the image to ROS (https://answers.ros.org/question/9765/how-to-convert-cvmat-to-sensor_msgsimageptr/)
    cv_bridge::CvImage msg;
    msg.header.stamp = ros::Time::now();
    msg.encoding = sensor_msgs::image_encodings::RGB8;
    msg.image = image;
    image_pub_.publish(msg);
}

} // end namespace camera_sim
