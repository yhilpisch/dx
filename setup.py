#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='dx',
    version='0.1',
    description='DX Analytics',
    classifiers=[
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Programming Language :: Python',
    ],
    author='Dr. Yves Hilpisch',
    author_email='dx@tpq.io',
    url='http://dx-analytics.com/',
    license='AGPLv3+',
    install_requires=[
        'pandas-datareader',
        'scipy',
        'statsmodels',
    ],
    packages=find_packages(),
)
