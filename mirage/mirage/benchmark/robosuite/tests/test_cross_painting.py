import numpy as np
import pytest

from mirage.benchmark.robosuite.xarm_transfer.cross_painting import (
    CrossPainter,
    CrossPaintingConfig,
    CrossPaintingError,
    CrossPaintingInput,
)
from mirage.benchmark.robosuite.xarm_transfer.image_runtime import (
    align_source_camera_frame,
    transform_manipulated_object_states,
)
from mirage.benchmark.robosuite.xarm_transfer.runtime import (
    ManipulatedObjectState,
    RobotCameraFrame,
)
from mirage.benchmark.robosuite.xarm_transfer.simulation_transform import (
    SimulationTaskFrameTransform,
    SimulationToolFrameCorrection,
    SimulationTransform,
)


def frame(*, target_depth=2.0, source_depth=1.0):
    target = np.full((7, 7, 3), 40, dtype=np.uint8)
    target[2:5, 2:5] = 200
    source = np.zeros_like(target)
    source[3:6, 3:6] = (10, 100, 240)
    target_mask = np.zeros((7, 7), dtype=bool)
    target_mask[2:5, 2:5] = True
    source_mask = np.zeros((7, 7), dtype=bool)
    source_mask[3:6, 3:6] = True
    object_mask = np.zeros((7, 7), dtype=bool)
    return CrossPaintingInput(
        target,
        np.full((7, 7), target_depth),
        target_mask,
        target_mask.copy(),
        object_mask,
        source,
        np.full((7, 7), source_depth),
        source_mask,
    )


def test_cross_painter_removes_target_and_inserts_visible_source():
    painter = CrossPainter(
        CrossPaintingConfig(target_mask_dilation_px=0, depth_epsilon_m=0.0)
    )

    result = painter.compose(frame())

    assert result.target_robot_pixels == 9
    assert result.source_robot_pixels == 9
    assert result.source_visible_pixels == 9
    np.testing.assert_array_equal(result.policy_rgb[4, 4], (10, 100, 240))
    assert not np.any(result.background_rgb[2:5, 2:5] == 200)
    processed = result.policy_chw_float32()
    assert processed.shape == (3, 7, 7)
    assert processed.dtype == np.float32
    np.testing.assert_allclose(
        processed[:, 4, 4], np.asarray((10, 100, 240)) / 255.0, atol=1 / 255
    )


def test_cross_painter_preserves_closer_target_object_outside_removed_robot():
    value = frame(target_depth=0.5, source_depth=1.0)
    value.target_rgb[5, 5] = (5, 6, 7)
    value.target_object_mask[5, 5] = True
    painter = CrossPainter(CrossPaintingConfig(target_mask_dilation_px=0))

    result = painter.compose(value)

    assert not result.source_visible_mask[5, 5]
    np.testing.assert_array_equal(result.policy_rgb[5, 5], (5, 6, 7))
    assert result.source_visible_mask[3, 3]


def test_cross_painter_dilates_target_removal_mask():
    result = CrossPainter(
        CrossPaintingConfig(target_mask_dilation_px=1)
    ).compose(frame())

    assert np.count_nonzero(result.target_inpaint_mask) == 25


def test_cross_painter_protects_object_pixels_from_gripper_dilation():
    value = frame(target_depth=0.5, source_depth=1.0)
    value.target_rgb[1, 2] = (220, 10, 10)
    value.target_object_mask[1, 2] = True

    result = CrossPainter(
        CrossPaintingConfig(target_mask_dilation_px=1)
    ).compose(value)

    assert not result.target_inpaint_mask[1, 2]
    np.testing.assert_array_equal(result.policy_rgb[1, 2], (220, 10, 10))
    assert result.protected_object_pixels == 1
    assert result.object_pixel_loss == 0


@pytest.mark.parametrize(
    "field, malformed, message",
    (
        ("target_rgb", np.zeros((7, 7)), "target_rgb"),
        ("source_depth_m", np.zeros((7, 7)), "positive finite"),
        (
            "target_robot_mask",
            np.zeros((7, 7), dtype=bool),
            "at least one pixel",
        ),
    ),
)
def test_cross_painter_rejects_invalid_inputs(field, malformed, message):
    value = frame()
    # Frozen dataclasses protect runtime messages; object.__setattr__ is used
    # only here to exercise validation of malformed external arrays.
    object.__setattr__(value, field, malformed)

    with pytest.raises(CrossPaintingError, match=message):
        CrossPainter().compose(value)


def test_source_robot_layer_reprojects_to_target_camera_with_z_buffer():
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[1, 1] = (20, 40, 60)
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    intrinsic = np.eye(3)
    source = RobotCameraFrame(
        rgb=rgb,
        depth_m=np.ones((4, 4)),
        robot_mask=mask,
        gripper_mask=mask.copy(),
        object_mask=np.zeros_like(mask),
        camera_name="agentview",
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=np.eye(4),
    )
    target_extrinsic = np.eye(4)
    target_extrinsic[0, 3] = 1.0
    target = RobotCameraFrame(
        rgb=np.zeros_like(rgb),
        depth_m=np.full((4, 4), 2.0),
        robot_mask=mask,
        gripper_mask=mask.copy(),
        object_mask=np.zeros_like(mask),
        camera_name="agentview",
        intrinsic_matrix=intrinsic,
        extrinsic_matrix=target_extrinsic,
    )

    aligned = align_source_camera_frame(source, target)

    assert aligned.robot_mask[1, 0]
    np.testing.assert_array_equal(aligned.rgb[1, 0], (20, 40, 60))
    assert aligned.depth_m[1, 0] == pytest.approx(1.0)


def test_object_free_joint_qpos_and_qvel_use_only_task_frame_transform():
    half_sqrt = np.sqrt(0.5)
    transform = SimulationTransform(
        SimulationTaskFrameTransform(
            "target",
            "source",
            (1.0, 2.0, 3.0),
            (0.0, 0.0, half_sqrt, half_sqrt),
            "test",
        ),
        SimulationToolFrameCorrection("xarm_tool", "panda_tool"),
    )
    states = {
        "Can_joint0": ManipulatedObjectState(
            np.asarray((1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)),
            np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0)),
        )
    }

    result = transform_manipulated_object_states(states, transform)["Can_joint0"]

    np.testing.assert_allclose(result.qpos[:3], (1.0, 3.0, 3.0), atol=1e-12)
    np.testing.assert_allclose(result.qpos[3:], (half_sqrt, 0.0, 0.0, half_sqrt))
    np.testing.assert_allclose(result.qvel[:3], (0.0, 1.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(result.qvel[3:], (-1.0, 0.0, 0.0), atol=1e-12)
