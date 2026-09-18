from setuptools import setup
from glob import glob
import os

package_name = 'local_path_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, f'{package_name}.scripts'],
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
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'local_path_planner_node = local_path_planner.local_path_planner_node:main',
        ],
    },
)
