from setuptools import setup
from glob import glob
import os

package_name = 'vl53l8cx_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/' + package_name, ['package.xml']),        
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='talhen',
    maintainer_email='Tal.Hen@mapcore.com',
    description='ROS2 package for VL53L8CX depth sensing',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vl53l8cx_ros2_node = vl53l8cx_ros2.vl53l8cx_ros2_node:main',
        ],
    },
)
