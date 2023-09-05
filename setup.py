from ez_setup import use_setuptools
use_setuptools()  # nopep8

from setuptools import setup, find_packages

setup(
    name='populationsim',
    version='0.6.0',
    description='Population Synthesis',
    author='Ben Stabler',
    author_email='ben.stabler@rsginc.com',
    license='BSD-3',
    url='https://github.com/ActivitySim/populationsim',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: BSD License'
    ],
    packages=find_packages(exclude=['*.tests']),
    include_package_data=True,
    install_requires=[
        'activitysim == 1.1.3',
        'numpy >= 1.24.4',
        'pandas >= 1.5.3',
        'ortools >= 9.6.2534',
        'tqdm >= 4.62.3',
    ]
)
