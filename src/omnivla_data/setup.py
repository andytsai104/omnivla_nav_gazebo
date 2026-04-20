from setuptools import setup
import os
from glob import glob

package_name = 'omnivla_data'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alan',
    maintainer_email='hcheng57@asu.edu',
    description='Data collection package for OmniVLA navigation.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'data_logger_node = omnivla_data.data_logger_node:main',
            'episode_manager_node = omnivla_data.episode_manager_node:main',
        ],
    },
)
