"""Projection of CanonicalState into Robomimic low-dimensional inputs."""

from typing import Mapping, Sequence

import numpy as np

from mirage_state.model import CanonicalState


class CanonicalStateError(ValueError):
    """Raised when a canonical state cannot safely drive policy inference."""


class CanonicalPolicyObservationProjector:
    """Validate a canonical sample and build checkpoint-shaped observations."""

    SUPPORTED_KEYS = {
        'robot0_eef_pos',
        'robot0_eef_quat',
        'robot0_gripper_qpos',
        'object',
    }

    def __init__(
        self,
        *,
        expected_shapes: Mapping[str, Sequence[int]],
        expected_frame_id: str,
        expected_object_schema: str,
        max_state_age_sec: float = 0.15,
    ) -> None:
        self.expected_shapes = {
            key: tuple(int(size) for size in shape)
            for key, shape in expected_shapes.items()
        }
        unsupported = set(self.expected_shapes) - self.SUPPORTED_KEYS
        if unsupported:
            raise CanonicalStateError(
                'unsupported checkpoint observations: ' + ', '.join(sorted(unsupported))
            )
        self.expected_frame_id = expected_frame_id
        self.expected_object_schema = expected_object_schema
        self.max_state_age_sec = float(max_state_age_sec)

    @classmethod
    def from_shape_metadata(
        cls,
        shape_metadata: Mapping[str, object],
        **kwargs,
    ) -> 'CanonicalPolicyObservationProjector':
        shapes = shape_metadata.get('all_shapes')
        if not isinstance(shapes, Mapping):
            raise CanonicalStateError('checkpoint shape_metadata has no all_shapes')
        return cls(expected_shapes=shapes, **kwargs)

    @staticmethod
    def _ensure_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
        if tuple(value.shape) != shape:
            raise CanonicalStateError(
                f'{name} shape {tuple(value.shape)} does not match {shape}'
            )

    def project(
        self,
        state: CanonicalState,
        *,
        now_ns: int | None = None,
    ) -> dict[str, np.ndarray]:
        if not state.valid:
            raise CanonicalStateError(
                'canonical state invalid: ' + ', '.join(state.invalid_reasons)
            )
        if not state.fresh:
            raise CanonicalStateError('canonical state is not fresh')
        if now_ns is not None:
            age = (int(now_ns) - state.timestamp_ns) / 1e9
            if age < 0.0 or age > self.max_state_age_sec:
                raise CanonicalStateError(f'canonical state age {age:.6f}s is invalid')
        if state.frame_id != self.expected_frame_id:
            raise CanonicalStateError(
                f'frame {state.frame_id!r} is not transformed to '
                f'{self.expected_frame_id!r}'
            )
        if state.object_schema != self.expected_object_schema:
            raise CanonicalStateError(
                f'object schema {state.object_schema!r} does not match '
                f'{self.expected_object_schema!r}'
            )

        values = {
            'robot0_eef_pos': np.asarray(state.ee_position_m, dtype=np.float32),
            'robot0_eef_quat': np.asarray(
                state.ee_quaternion_xyzw, dtype=np.float32
            ),
            'robot0_gripper_qpos': np.asarray(
                [state.gripper_width_m / 2.0, -state.gripper_width_m / 2.0],
                dtype=np.float32,
            ),
        }
        object_entry = state.object_entry('object')
        if object_entry is not None:
            values['object'] = np.asarray(
                object_entry.values, dtype=np.float32
            ).reshape(object_entry.shape)

        projected: dict[str, np.ndarray] = {}
        for key, shape in self.expected_shapes.items():
            if key not in values:
                raise CanonicalStateError(f'canonical state has no {key!r} entry')
            value = values[key]
            self._ensure_shape(key, value, shape)
            projected[key] = value
        return projected
