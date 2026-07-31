from glob import glob
from setuptools import find_packages, setup


package_name = 'mirage_xarm_state'


setup(
    name=package_name,
    version='0.1.0',
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
    install_requires=['setuptools'],
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
        ],
    },
)
