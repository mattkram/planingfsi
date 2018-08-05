from setuptools import setup

import planingfsi


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
    version=planingfsi.__version__,
    description='FSI for large-deformation planing surfaces',
    long_description=readme(),
    author='Matthew Kramer',
    author_email='matthew.robert.kramer@gmail.com',
    license='MIT',
    packages=['planingfsi'],
    setup_requires=requirements,
    tests_require=test_requirements,
    entry_points={
        'console_scripts': [
            'planingFSI=planingfsi.command_line.planingfsi:main',
            'generateMesh=planingfsi.command_line.mesh:main'
        ]
    }
)
