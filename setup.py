from setuptools import setup, find_packages
import sys
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_dir)

setup(
    name="tplus",
    version="0.1.9.1",
    author="Zeping Zhang",
    license="GPL-v3",
    author_email="zhangzp9970@outlook.com",
    description="Torchplus is a utilities library that extends pytorch and torchvision",
    url="https://github.com/zhangzp9970/torchplus",
    include_package_data=True,
    packages=find_packages(parent_dir),
    python_requires=">=3.8",
    install_requires=["torch", "torchvision", "pillow", "pandas", "tensorboard"],
)
