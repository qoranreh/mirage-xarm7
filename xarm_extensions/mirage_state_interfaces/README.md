# mirage_state_interfaces

ROS 2 interface definitions for Mirage's robot-neutral state boundary.

`CanonicalState` represents one timestamped observation with SI units,
`xyzw` quaternion order, an explicit coordinate frame, physical gripper width,
and structured object state. Producers must set `fresh` and `valid` and explain
rejected samples in `errors`; consumers must fail closed on either flag.

`ObjectState` stores named arrays as flattened values plus explicit shapes and
identifies their semantic layout through `schema`. The initial Robomimic
contract uses the `object` entry and schemas such as
`PickPlaceCan:robomimic-v1`.
