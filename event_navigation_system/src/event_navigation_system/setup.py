from setuptools import setup, find_packages
from glob import glob
import os

package_name = 'event_navigation_system'

# colcon requires relative data_files paths.
_config_files = glob('config/*.yaml')
if not _config_files:
    _config_files = glob('../../config/*.yaml')

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=('test',)),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), _config_files),
    ],
    install_requires=['setuptools', 'opencv-python', 'networkx', 'numba'],
    zip_safe=True,
    maintainer='talhen',
    maintainer_email='Tal.Hen@mapcore.com',
    description='Event-based navigation system for micro-drones in GPS-denied environments',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'event_system_node = radical_event_navigation_system.event_system_node:main',
            'smart_navigation_node = radical_event_navigation_system.event_navigation_node:main',
            'web_dashboard_node = radical_event_navigation_system.web_dashboard_node:main',
        ],
    },
)
