#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
===============================
HtmlTestRunner
===============================


.. image:: https://img.shields.io/pypi/v/eh22tools.svg
        :target: https://pypi.python.org/pypi/eh22tools
.. image:: https://img.shields.io/travis/robertahamme/eh22tools.svg
        :target: https://travis-ci.org/robertahamme/eh22tools

Python scripts for Emerson and Hamme 2022 Chemical Oceanography


Links:
---------
* `Github <https://github.com/robertahamme/eh22tools>`_
"""

from setuptools import setup, find_packages

requirements = ['numpy']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Roberta Hamme",
    author_email='rhamme@uvic.ca',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Python scripts for Emerson and Hamme 2022 Chemical Oceanography",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=__doc__,
    include_package_data=True,
    keywords='eh22tools',
    name='eh22tools',
    packages=find_packages(include=['eh22tools']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/robertahamme/eh22tools',
    version='0.1.0',
    zip_safe=False,
)
