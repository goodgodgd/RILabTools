from setuptools import setup

package_name = 'yolo_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='falcon',
    maintainer_email='qmrmqmrm@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detector_alone = yolo_pkg.YOLOU_ONNX:main',
            'detector_server = yolo_pkg.YOLOU_ONNX_server:main',
            'detector_test = yolo_pkg.YOLOU_ONNX_test:main',
        ],
    },
)
