from glob import glob
from setuptools import find_packages, setup


package_name = 'mirage_xarm_state'


setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['numpy', 'PyYAML', 'setuptools'],
    zip_safe=True,
    maintainer='qoranreh',
    maintainer_email='qoranreh@example.com',
    description=(
        'Timestamp-aligned xArm joint, end-effector, gripper, and diagnostic '
        'state recorder.'
    ),
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'xarm_state_monitor = mirage_xarm_state.monitor:main',
            'xarm_gate_a_validator = mirage_xarm_state.gate_a:main',
            'xarm_gate_a_bag_check = mirage_xarm_state.bag_check:main',
            'xarm_canonical_state = mirage_xarm_state.canonical_node:main',
            'mirage-transform-validate = mirage_state.transform_cli:main',
            'mirage-transform-calibrate = mirage_state.calibration_cli:main',
            'xarm_transform_capture = '
            'mirage_xarm_state.transform_capture:main',
        ],
    },
)
