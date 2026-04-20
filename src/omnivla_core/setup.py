from setuptools import find_packages, setup
import os

package_name = 'omnivla_core'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andy',
    maintainer_email='andytsai104@gmail.com',
    description='TODO: Package description',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        "inference_node = omnivla_core.inference_node:main",
        "nav2_goal_bridge_node = omnivla_core.nav2_goal_bridge_node:main",
        ],
    },
    options={
        'build_scripts': {
            'executable': os.environ.get(
                'PYTHON_FOR_ROS_NODES',
                '/usr/bin/python3'
            )
        }
    },
)
