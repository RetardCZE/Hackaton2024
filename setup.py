from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements(filepath: Path):
    with filepath.open("r") as f:
        return f.read().splitlines()

setup(
    name='AI_Tutorial',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=parse_requirements(Path(__file__).parent / 'requirements.txt'),  # Add any dependencies your project needs
    entry_points={
        'console_scripts': [],
    },
)