# Robot-independent Cartesian state. All linear values use meters and the
# quaternion field follows geometry_msgs ordering: x, y, z, w.
std_msgs/Header header
geometry_msgs/Point ee_position_m
geometry_msgs/Quaternion ee_quaternion_xyzw
float64 gripper_width_m
ObjectState object_state
string source_robot
bool fresh
bool valid
string[] invalid_reasons
