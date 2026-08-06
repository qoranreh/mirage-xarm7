import pytest

pytest.importorskip('mirage_state_interfaces')

from mirage_state import NamedArray, make_canonical_state  # noqa: E402
from mirage_state.ros import (  # noqa: E402
    canonical_state_from_ros,
    canonical_state_to_ros,
)


def test_canonical_ros_round_trip_preserves_schema_and_values():
    state = make_canonical_state(
        ee_position_m=(0.1, 0.2, 0.3),
        ee_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        gripper_width_m=0.04,
        object_schema='PickPlaceCan:robomimic-v1',
        object_state=(NamedArray.from_values('object', range(14)),),
        frame_id='panda_sim_world',
        timestamp_ns=12_345_678_901,
        source_robot='panda_robosuite',
        fresh=True,
    )

    restored = canonical_state_from_ros(canonical_state_to_ros(state))

    assert restored == state
