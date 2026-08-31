"""Depth-aware cross-painting for image-policy embodiment transfer.

The compositor is deliberately independent from Robosuite and ROS. Runtime
adapters are responsible for rendering camera-aligned RGB, metric depth, and
robot masks; this module only validates and combines those arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class CrossPaintingError(RuntimeError):
    """Raised when a cross-paint input cannot be composed safely."""


@dataclass(frozen=True)
class CrossPaintingConfig:
    """Pixel-space settings matching the Mirage analytic compositor."""

    inpaint_radius_px: int = 3
    target_mask_dilation_px: int = 1
    depth_epsilon_m: float = 1e-4

    def __post_init__(self) -> None:
        if type(self.inpaint_radius_px) is not int or self.inpaint_radius_px <= 0:
            raise CrossPaintingError("inpaint radius must be a positive integer")
        if (
            type(self.target_mask_dilation_px) is not int
            or self.target_mask_dilation_px < 0
        ):
            raise CrossPaintingError("mask dilation must be a non-negative integer")
        epsilon = float(self.depth_epsilon_m)
        if not np.isfinite(epsilon) or epsilon < 0.0:
            raise CrossPaintingError("depth epsilon must be finite and non-negative")
        object.__setattr__(self, "depth_epsilon_m", epsilon)


@dataclass(frozen=True)
class CrossPaintingInput:
    """Camera-aligned source and target layers in OpenCV image convention."""

    target_rgb: np.ndarray
    target_depth_m: np.ndarray
    target_robot_mask: np.ndarray
    target_gripper_mask: np.ndarray
    target_object_mask: np.ndarray
    source_rgb: np.ndarray
    source_depth_m: np.ndarray
    source_robot_mask: np.ndarray


@dataclass(frozen=True)
class CrossPaintingResult:
    """Composite plus the intermediate masks required for audit videos."""

    policy_rgb: np.ndarray
    background_rgb: np.ndarray
    target_inpaint_mask: np.ndarray
    source_visible_mask: np.ndarray
    target_robot_pixels: int
    source_robot_pixels: int
    source_visible_pixels: int
    protected_object_pixels: int
    object_pixel_loss: int

    def policy_chw_float32(self) -> np.ndarray:
        """Return the exact processed RGB layout expected by Robomimic."""

        return np.ascontiguousarray(
            self.policy_rgb.transpose(2, 0, 1), dtype=np.float32
        ) / np.float32(255.0)


def _rgb(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise CrossPaintingError(f"{name} must have shape (H, W, 3)")
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    if not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array)):
        raise CrossPaintingError(f"{name} must contain finite RGB values")
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    if minimum < 0.0 or maximum > 1.0:
        raise CrossPaintingError(
            f"{name} floating-point values must be normalized to [0, 1]"
        )
    return np.ascontiguousarray(np.rint(array * 255.0).astype(np.uint8))


def _depth(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise CrossPaintingError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise CrossPaintingError(f"{name} must contain positive finite metric depths")
    return np.ascontiguousarray(array)


def _mask(
    value: np.ndarray,
    shape: tuple[int, int],
    name: str,
    *,
    require_non_empty: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[..., 0]
    if array.shape != shape:
        raise CrossPaintingError(f"{name} must have shape {shape}")
    result = np.ascontiguousarray(array.astype(bool))
    if require_non_empty and not np.any(result):
        raise CrossPaintingError(f"{name} must contain at least one pixel")
    return result


class CrossPainter:
    """Remove a target embodiment and paint the source embodiment in its place."""

    def __init__(self, config: CrossPaintingConfig | None = None) -> None:
        self.config = config or CrossPaintingConfig()

    def compose(self, frame: CrossPaintingInput) -> CrossPaintingResult:
        target_rgb = _rgb(frame.target_rgb, "target_rgb")
        source_rgb = _rgb(frame.source_rgb, "source_rgb")
        if source_rgb.shape != target_rgb.shape:
            raise CrossPaintingError("source and target RGB shapes must match")
        shape = target_rgb.shape[:2]
        target_depth = _depth(frame.target_depth_m, shape, "target_depth_m")
        source_depth = _depth(frame.source_depth_m, shape, "source_depth_m")
        target_mask = _mask(
            frame.target_robot_mask,
            shape,
            "target_robot_mask",
            require_non_empty=True,
        )
        target_gripper_mask = _mask(
            frame.target_gripper_mask, shape, "target_gripper_mask"
        )
        target_object_mask = _mask(
            frame.target_object_mask, shape, "target_object_mask"
        )
        source_mask = _mask(
            frame.source_robot_mask,
            shape,
            "source_robot_mask",
            require_non_empty=True,
        )
        if np.any(target_gripper_mask & ~target_mask):
            raise CrossPaintingError("target gripper mask must be inside robot mask")

        dilated_gripper_mask = target_gripper_mask
        if self.config.target_mask_dilation_px:
            size = 2 * self.config.target_mask_dilation_px + 1
            kernel = np.ones((size, size), dtype=np.uint8)
            dilated_gripper_mask = cv2.dilate(
                target_gripper_mask.astype(np.uint8), kernel, iterations=1
            ).astype(bool)
        # Preserve visible task-object pixels even when the gripper dilation
        # reaches across a contact boundary. The undilated robot mask remains
        # authoritative because element segmentation assigns one geom per
        # pixel and therefore cannot label the same pixel as an object.
        inpaint_mask = (target_mask | dilated_gripper_mask) & ~target_object_mask

        background = cv2.inpaint(
            target_rgb,
            inpaint_mask.astype(np.uint8) * np.uint8(255),
            inpaintRadius=self.config.inpaint_radius_px,
            flags=cv2.INPAINT_TELEA,
        )

        # Pixels genuinely occupied by the removed target robot have no
        # trustworthy background depth, so the source layer wins there.
        # Protected object pixels remain depth-tested: the Panda may cover the
        # object only when its rendered depth is actually nearer.
        source_visible = source_mask & (
            inpaint_mask
            | (source_depth + self.config.depth_epsilon_m < target_depth)
        )
        policy_rgb = background.copy()
        policy_rgb[source_visible] = source_rgb[source_visible]
        object_in_front = target_object_mask & ~source_visible
        policy_rgb[object_in_front] = target_rgb[object_in_front]
        protected_object = target_object_mask & (
            target_mask | dilated_gripper_mask
        )
        protected_object_in_front = protected_object & ~source_visible
        object_pixel_loss = int(
            np.count_nonzero(
                np.any(
                    policy_rgb[protected_object_in_front]
                    != target_rgb[protected_object_in_front],
                    axis=1,
                )
            )
        )

        return CrossPaintingResult(
            policy_rgb=np.ascontiguousarray(policy_rgb),
            background_rgb=np.ascontiguousarray(background),
            target_inpaint_mask=np.ascontiguousarray(inpaint_mask),
            source_visible_mask=np.ascontiguousarray(source_visible),
            target_robot_pixels=int(np.count_nonzero(target_mask)),
            source_robot_pixels=int(np.count_nonzero(source_mask)),
            source_visible_pixels=int(np.count_nonzero(source_visible)),
            protected_object_pixels=int(np.count_nonzero(protected_object)),
            object_pixel_loss=object_pixel_loss,
        )
