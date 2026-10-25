import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'obs_avoidance'

setup(
    name=package_name,
    version='1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Matthew Troy',
    maintainer_email='matthew.troy271@gmail.com',
    description='Obstacle-avoidance Control Algorithm for BlueROV2 with Ping-sonar model',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'start = obs_avoidance.start:main',
            'pid = obs_avoidance.pid:main',
            'nmpc = obs_avoidance.nmpc:main',
        ],
    },
)
