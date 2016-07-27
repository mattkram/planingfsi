from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='planingfsi',
      version='0.1',
      description='FSI for planing surfaces',
      long_description=readme(),
      author='Matthew Kramer',
      author_email='matthew.robert.kramer@gmail.com',
      license='MIT',
      packages=['planingfsi'],
      scripts=['bin/planingFSI'])
