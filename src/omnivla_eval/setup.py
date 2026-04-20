from setuptools import find_packages, setup

package_name = 'omnivla_eval'

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
    maintainer='Roy',
    maintainer_email='mengjuyu@asu.edu',
    description='Evaluation system for OmniVLA-edge and Nav2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'eval_runner_node = omnivla_eval.eval_runner_node:main',
        ],
    },
)
