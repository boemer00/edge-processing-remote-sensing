from setuptools import setup, find_packages

# Read requirements.txt and use its contents as dependencies
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='edge',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,

    author='Renato Boemer',
    author_email='boemer00@gmail.com',
    description='A package for a lightweight real-time satellite image classification on edge computing environments.',
    keywords='satellite image classification neural-network edge computing',
    url='https://github.com/boemer00/edge-processing-remote-sensing',
)
