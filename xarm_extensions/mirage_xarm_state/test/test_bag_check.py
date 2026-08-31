from mirage_xarm_state.bag_check import CORE_TOPICS, evaluate_topic_counts


def test_bag_topic_coverage_passes_with_required_real_streams():
    counts = {topic: 10 for topic in CORE_TOPICS}
    counts['/mirage/xarm/gripper_joint_state'] = 2
    counts['/xarm/robot_states'] = 10

    result = evaluate_topic_counts(
        counts, require_gripper=True, require_robot_state=True
    )

    assert result['passed']
    assert not result['missing_or_empty_topics']


def test_bag_topic_coverage_reports_missing_streams():
    counts = {topic: 10 for topic in CORE_TOPICS}

    result = evaluate_topic_counts(
        counts, require_gripper=True, require_robot_state=True
    )

    assert not result['passed']
    assert '/mirage/xarm/gripper_joint_state' in result[
        'missing_or_empty_topics'
    ]
    assert '/xarm/robot_states' in result['missing_or_empty_topics']
