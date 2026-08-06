import ast
import inspect

import numpy as np

from mirage_state.calibration import ToolOffset
from mirage_state.transforms import CartesianPose
from mirage_xarm_state import transform_capture
from mirage_xarm_state.transform_capture import PoseCaptureBuffer


def pointer_tool():
    return ToolOffset(
        parent_frame='link_eef',
        tool_frame='calibration_tip',
        translation_m=(0.0, 0.0, 0.1),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_capture_applies_tool_tip_offset_and_rejects_stale_samples():
    buffer = PoseCaptureBuffer(
        expected_frame='link_base',
        tool=pointer_tool(),
        sample_count=2,
        max_age_sec=0.2,
    )
    pose = CartesianPose(
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
        'link_base',
    )

    assert not buffer.add(pose, stamp_ns=1, now_ns=300_000_001)
    assert 'stale' in buffer.last_rejection
    assert buffer.add(pose, stamp_ns=400_000_000, now_ns=500_000_000)
    assert buffer.add(pose, stamp_ns=500_000_000, now_ns=600_000_000)

    evidence = buffer.evidence()
    np.testing.assert_allclose(evidence.position_m, (1.0, 2.0, 3.1))
    assert evidence.sample_count == 2


def test_capture_rejects_wrong_frame_and_duplicate_timestamp():
    buffer = PoseCaptureBuffer(
        expected_frame='link_base',
        tool=pointer_tool(),
        sample_count=2,
        max_age_sec=0.2,
    )
    wrong = CartesianPose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        'wrong',
    )
    right = CartesianPose(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        'link_base',
    )

    assert not buffer.add(wrong, stamp_ns=1, now_ns=1)
    assert buffer.add(right, stamp_ns=2, now_ns=2)
    assert not buffer.add(right, stamp_ns=2, now_ns=2)
    assert 'not newer' in buffer.last_rejection


def test_capture_module_has_no_command_capable_ros_endpoints():
    tree = ast.parse(inspect.getsource(transform_capture))
    forbidden = {
        'create_publisher',
        'create_client',
        'create_service',
        'create_action_client',
    }
    used = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert forbidden.isdisjoint(used)
