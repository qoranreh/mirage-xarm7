import os
import sys

from setuptools import find_packages, setup

root_dir = os.path.dirname(os.path.realpath(__file__))

setup(
    name='mirage',
    version='0.1',
    description='Repo for Mirage',
    author='Lawrence Chen, Karthik Dharmarajan, Kush Hari',
    package_dir = {'': '.'},
    packages=find_packages(include=['mirage', 'mirage.*']),
    include_package_data=True,
    package_data={
        "mirage.benchmark.robosuite.xarm_transfer": [
            "assets/LICENSE.*",
            "assets/*.md",
            "assets/grippers/*.xml",
            "assets/grippers/meshes/xarm7_gripper/*.stl",
            "assets/robots/xarm7/*.xml",
            "assets/robots/xarm7/meshes/*.stl",
        ],
    },
    install_requires=[
        "numpy",
        "pyyaml",
        "prettytable",
        "scikit-learn",
        "open3d"
    ],
    extras_require={
        "docs": [
            "sphinx",
            "pydata-sphinx-theme"
        ]
    },
    entry_points={
        "console_scripts": [
            "mirage-canonical-policy-probe = "
            "mirage.benchmark.robosuite.canonical_policy_probe:main",
            "mirage-source-landmarks = "
            "mirage.benchmark.robosuite.source_landmarks:main",
            "mirage-xarm-transfer = "
            "mirage.benchmark.robosuite.xarm_transfer.cli:main",
        ],
    },
)
