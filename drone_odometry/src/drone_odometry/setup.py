from setuptools import setup
from glob import glob
import os

package_name = 'drone_odometry'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='talhen',
    maintainer_email='Tal.Hen@mapcore.com',
    description='6D linear Kalman-filter odometry',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'drone_odometry_node = drone_odometry.kf_odometry_node:main',
        ],
    },
)
