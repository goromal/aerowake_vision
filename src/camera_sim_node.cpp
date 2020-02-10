#include "aerowake_vision/camera_sim.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "camera_sim_node");
    camera_sim::CameraSim CS;
    ros::spin();
    return 0;
}
