from setuptools import find_packages, setup
from pathlib import Path

requirements = Path("requirements.txt").read_text().splitlines()

setup(name="src", packages=find_packages(), install_requires=requirements)
