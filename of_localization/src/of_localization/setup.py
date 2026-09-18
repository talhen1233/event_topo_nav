from glob import glob
import os
from setuptools import find_packages, setup

package_name = "of_localization"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.launch.py"),
        ),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*"),
        ),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="talhen",
    maintainer_email="Tal.Hen@mapcore.com",
    description="Optical-flow localization with hybrid local-depth support.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "of_localization_node = of_localization.of_localization_node:main",
            "of_debug_record_bag = of_localization.offline_debug.record_debug_bag:main",
            "of_debug_export_images = of_localization.offline_debug.export_debug_images:main",
        ],
    },
)
