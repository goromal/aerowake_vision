#include "aerowake_vision/camera_sim.h"

#define TO_ULONG(i) static_cast<unsigned long>(i)

namespace camera_sim {

CameraSim::CameraSim() : camera_counter_(0), nh_(), nh_private_("~"), tfBuffer_(), tfListener_(tfBuffer_)
{
    marker_pub_ = nh_.advertise<visualization_msgs::Marker>("beacon_array_marker", 1);
    marker_.header.frame_id    = "boatNED";
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

    image_pub_ = nh_.advertise<sensor_msgs::Image>("camera/image_raw", 1);
    caminfo_pub_ = nh_.advertise<sensor_msgs::CameraInfo>("camera/camera_info", 1);

    std::vector<double> vec_X_ship_beacon = std::vector<double>(6);
    ROS_ASSERT(nh_private_.getParam("X_ship_beacon", vec_X_ship_beacon));
    Xformd X_ship_beacon = Xformd(Vector3d(vec_X_ship_beacon[0], vec_X_ship_beacon[1], vec_X_ship_beacon[2]),
                                  Quatd::from_euler(vec_X_ship_beacon[3], vec_X_ship_beacon[4], vec_X_ship_beacon[5]));

    std::vector<double> points_BEACON_coeff = std::vector<double>(3 * NUM_BEACONS);
    ROS_ASSERT(nh_private_.getParam("beacon_positions", points_BEACON_coeff));
    for (int i = 0; i < NUM_BEACONS; i++)
    {
        Vector3d point_SHIP = X_ship_beacon.transforma(Vector3d(points_BEACON_coeff[TO_ULONG(3 * i + 0)],
                                                                points_BEACON_coeff[TO_ULONG(3 * i + 1)],
                                                                points_BEACON_coeff[TO_ULONG(3 * i + 2)]));
        points_SHIP_.push_back(cv::Point3d(point_SHIP.x(), point_SHIP.y(), point_SHIP.z()));
        geometry_msgs::Point p;
        p.x = point_SHIP.x();
        p.y = point_SHIP.y();
        p.z = point_SHIP.z();
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
    X_UAV_CAM_ = Xformd(Vector3d(x, y, z), Quatd::from_euler(phi, tht, psi));

    ROS_ASSERT(nh_private_.getParam("image_width", img_w_));
    ROS_ASSERT(nh_private_.getParam("image_height", img_h_));
    Vector2d img_size(img_w_, img_h_);
    std::vector<double> opencv_camera_matrix_coeff = std::vector<double>(9);
    ROS_ASSERT(nh_private_.getParam("camera_matrix/data", opencv_camera_matrix_coeff));
    // https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
    K_ = cv::Mat(3, 3, cv::DataType<double>::type);
    K_.at<double>(0,0) = opencv_camera_matrix_coeff[0];
    K_.at<double>(0,1) = opencv_camera_matrix_coeff[1];
    K_.at<double>(0,2) = opencv_camera_matrix_coeff[2];
    K_.at<double>(1,0) = opencv_camera_matrix_coeff[3];
    K_.at<double>(1,1) = opencv_camera_matrix_coeff[4];
    K_.at<double>(1,2) = opencv_camera_matrix_coeff[5];
    K_.at<double>(2,0) = opencv_camera_matrix_coeff[6];
    K_.at<double>(2,1) = opencv_camera_matrix_coeff[7];
    K_.at<double>(2,2) = opencv_camera_matrix_coeff[8];

    ROS_ASSERT(nh_private_.getParam("camera/pointgrey_camera/frame_rate", cameraRate_));
    ROS_ASSERT(cameraRate_ >= 1.0);
    camera_counter_limit_ = VIS_MULTIPLIER;
    updateRate_ = VIS_MULTIPLIER * cameraRate_;

    ROS_ASSERT(nh_private_.getParam("pixel_stdev", pix_stdev_));
    gaussian_dist_ = std::normal_distribution<double>(0.0, 1.0);

    caminfo_.height = static_cast<unsigned int>(img_h_);
    caminfo_.width = static_cast<unsigned int>(img_w_);
    ROS_ASSERT(nh_private_.getParam("distortion_model", caminfo_.distortion_model));
    std::vector<double> opencv_dist_matrix_coeff = std::vector<double>(5);
    ROS_ASSERT(nh_private_.getParam("distortion_coefficients/data", opencv_dist_matrix_coeff));
    ROS_ASSERT(nh_private_.getParam("distortion_coefficients/data", caminfo_.D));
    D_ = cv::Mat(5, 1, cv::DataType<double>::type);
    D_.at<double>(0) = opencv_dist_matrix_coeff[0];
    D_.at<double>(1) = opencv_dist_matrix_coeff[1];
    D_.at<double>(2) = opencv_dist_matrix_coeff[2];
    D_.at<double>(3) = opencv_dist_matrix_coeff[3];
    D_.at<double>(4) = opencv_dist_matrix_coeff[4];

    for (unsigned int i = 0; i < 9; i++)
        caminfo_.K[i] = opencv_camera_matrix_coeff[i];
    std::vector<double> opencv_proj_matrix_coeff = std::vector<double>(12);
    ROS_ASSERT(nh_private_.getParam("projection_matrix/data", opencv_proj_matrix_coeff));
    for (unsigned int i = 0; i < 12; i++)
        caminfo_.P[i] = opencv_proj_matrix_coeff[i];

    // Update timer
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
    // Get (robotics convention) passive transform from SHP frame to CAM frame
    Xformd X_SHP_CAM;
    try
    {
        tf_UAV_SHIP_ = tfBuffer_.lookupTransform("boatNED", "UAV", ros::Time(0));
        Xformd X_SHIP_UAV(Vector3d(tf_UAV_SHIP_.transform.translation.x,
                                   tf_UAV_SHIP_.transform.translation.y,
                                   tf_UAV_SHIP_.transform.translation.z),
                    Quatd(Vector4d(tf_UAV_SHIP_.transform.rotation.w,
                                   tf_UAV_SHIP_.transform.rotation.x,
                                   tf_UAV_SHIP_.transform.rotation.y,
                                   tf_UAV_SHIP_.transform.rotation.z)));
        X_SHP_CAM = X_SHIP_UAV * X_UAV_CAM_;
    }
    catch (tf2::TransformException &ex)
    {
        ROS_WARN("%s",ex.what());
        ros::Duration(1.0).sleep();
        return;
    }

    // Extract rvec and tvec from X_SHP_CAM
    cv::Mat R;
    cv::eigen2cv(X_SHP_CAM.q().R(), R);
    cv::Vec3d rvec_SHP_CAM;
    cv::Rodrigues(R, rvec_SHP_CAM);
    Vector3d tvec_eig = X_SHP_CAM.inverse().t();
    cv::Vec3d tvec_SHP_CAM;
    cv::eigen2cv(tvec_eig, tvec_SHP_CAM);

    // Draw blank black image (https://stackoverflow.com/questions/28780947/opencv-create-new-image-using-cvmat/28782436)
    cv::Mat image(img_h_, img_w_, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Scalar white(255, 255, 255);

    // Perform projection of feature points onto the camera pixel frame
    std::vector<cv::Point2d> pixels;
    cv::projectPoints(points_SHIP_, rvec_SHP_CAM, tvec_SHP_CAM, K_, D_, pixels);

    // Add pixel measurement noise and draw on image for each feature point
    for (int i = 0; i < NUM_BEACONS; i++)
    {
        pixels[TO_ULONG(i)] += cv::Point2d(static_cast<int>(pix_stdev_ * gaussian_dist_(random_generator_)),
                                           static_cast<int>(pix_stdev_ * gaussian_dist_(random_generator_)));
        // add to image (https://stackoverflow.com/questions/31658132/c-opencv-not-drawing-circles-on-mat-image)
        //              (https://stackoverflow.com/questions/18886083/how-fill-circles-on-opencv-c)
        if (pixels[TO_ULONG(i)].x > PIXEL_RAD + 1 && pixels[TO_ULONG(i)].x < img_w_ - PIXEL_RAD - 1 &&
            pixels[TO_ULONG(i)].y > PIXEL_RAD + 1 && pixels[TO_ULONG(i)].y < img_h_ - PIXEL_RAD - 1)
        {
            cv::circle(image, pixels[TO_ULONG(i)], PIXEL_RAD, white, CV_FILLED);
        }
    }

    // Use CV Bridge to publish the image to ROS (https://answers.ros.org/question/9765/how-to-convert-cvmat-to-sensor_msgsimageptr/)
    cv_bridge::CvImage msg;
    msg.header.stamp = ros::Time::now();
    msg.encoding = sensor_msgs::image_encodings::RGB8;
    msg.image = image;
    image_pub_.publish(msg);
    caminfo_.header = msg.header;
    caminfo_pub_.publish(caminfo_);
}

} // end namespace camera_sim
