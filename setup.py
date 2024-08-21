from pathlib import Path

from setuptools import find_packages, setup

requirements = Path("requirements.txt").read_text().splitlines()

setup(name="fmri_timescales", packages=find_packages(), install_requires=requirements)
