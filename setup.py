#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
	requirements = f.read().splitlines()

DISTNAME = 'dx'

setup(name=DISTNAME,
        version='0.1.21',
        packages=find_packages(include=['dx', 'dx.*']),
        description='DX Analytics',
        author='Dr. Yves Hilpisch',
        author_email='dx@tpq.io',
        url='http://dx-analytics.com/',
        install_requires=requirements)
