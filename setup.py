import os
from setuptools import setup, find_packages


here = os.path.dirname(__file__)

about = {}

with open(os.path.join(here, 'src', 'planingfsi', '__version__.py')) as f:
    exec(f.read(), about)

def readme():
    with open('README.md') as f:
        return f.read()


requirements = [
    'matplotlib',
    'numpy',
    'scipy',
    'six',
    'click',
    'pytest-runner'
]
test_requirements = [
    'pytest',
    'pytest-cov',
    'coverage',
]

setup(
    name='planingfsi',
    version=about['__version__'],
    description='FSI for large-deformation planing surfaces',
    long_description=readme(),
    author='Matthew Kramer',
    author_email='matthew.robert.kramer@gmail.com',
    license='MIT',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    setup_requires=requirements,
    tests_require=test_requirements,
    entry_points={
        'console_scripts': [
            'planingFSI=planingfsi.command_line.planingfsi:main',
            'generateMesh=planingfsi.command_line.mesh:main'
        ]
    }
)
