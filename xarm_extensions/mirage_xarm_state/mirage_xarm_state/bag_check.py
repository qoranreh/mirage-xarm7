"""Offline topic-coverage check for a finalized Gate A rosbag."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Iterable

import rosbag2_py


CORE_TOPICS = (
    '/joint_states',
    '/tf',
    '/mirage/xarm/arm_joint_states',
    '/mirage/xarm/ee_pose',
    '/mirage/xarm/state_diagnostics',
)
GRIPPER_TOPICS = ('/mirage/xarm/gripper_joint_state',)
ROBOT_TOPICS = ('/xarm/robot_states',)


def evaluate_topic_counts(
    topic_counts: Dict[str, int],
    *,
    require_gripper: bool,
    require_robot_state: bool,
) -> dict:
    """Return a deterministic pass/fail report for recorded topics."""
    required_topics = list(CORE_TOPICS)
    if require_gripper:
        required_topics.extend(GRIPPER_TOPICS)
    if require_robot_state:
        required_topics.extend(ROBOT_TOPICS)
    missing = [
        topic for topic in required_topics if topic_counts.get(topic, 0) <= 0
    ]
    return {
        'passed': not missing,
        'required_topics': required_topics,
        'missing_or_empty_topics': missing,
        'topic_counts': dict(sorted(topic_counts.items())),
    }


def inspect_bag(
    bag_path: Path,
    *,
    storage_id: str,
    require_gripper: bool,
    require_robot_state: bool,
) -> dict:
    """Read rosbag metadata without replaying or publishing any messages."""
    metadata = rosbag2_py.Info().read_metadata(
        str(bag_path), storage_id
    )
    topic_counts = {
        item.topic_metadata.name: int(item.message_count)
        for item in metadata.topics_with_message_count
    }
    report = evaluate_topic_counts(
        topic_counts,
        require_gripper=require_gripper,
        require_robot_state=require_robot_state,
    )
    report.update(
        {
            'schema_version': 1,
            'gate': 'A-bag',
            'bag_path': str(bag_path),
            'storage_id': storage_id,
            'checked_at': datetime.now(timezone.utc).isoformat(),
            'total_message_count': int(metadata.message_count),
        }
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Verify that a finalized Gate A rosbag contains every required '
            'raw and normalized state stream.'
        )
    )
    parser.add_argument('bag_path', type=Path)
    parser.add_argument('--storage-id', default='sqlite3')
    parser.add_argument(
        '--no-gripper', action='store_true', help='Do not require gripper state.'
    )
    parser.add_argument(
        '--require-robot-state',
        action='store_true',
        help='Require the real-driver /xarm/robot_states stream.',
    )
    parser.add_argument(
        '--report-path',
        type=Path,
        help='JSON output path; defaults to BAG_PATH/gate_a_bag_report.json.',
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    bag_path = args.bag_path.expanduser().resolve()
    report = inspect_bag(
        bag_path,
        storage_id=args.storage_id,
        require_gripper=not args.no_gripper,
        require_robot_state=args.require_robot_state,
    )
    report_path = (
        args.report_path.expanduser()
        if args.report_path
        else bag_path / 'gate_a_bag_report.json'
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report, ensure_ascii=False, indent=2, sort_keys=True
    )
    report_path.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
