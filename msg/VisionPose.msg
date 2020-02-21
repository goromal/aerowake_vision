Header    header
string    child_frame_id
geometry_msgs/Transform transform
uint8     sol_status
uint8     contour_count                # LIMIT: 12
uint8     centroid_count               # LIMIT: 12
bool      outlier
bool      dynamically_valid            # Eccentricity threshold: 10

# Solution Status Flags
uint8 NO_SOLUTION = 0
uint8 INCREMENTAL = 1
uint8 DERIVED_INIT = 2
