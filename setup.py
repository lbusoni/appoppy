#!/usr/bin/env python
from setuptools import setup


setup(name='appoppy',
      description='test of poppy',
      version='0.1',
      classifiers=['Development Status :: 4 - Beta',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3',
                   ],
      long_description=open('README.md').read(),
      url='',
      author_email='lorenzo.busoni@inaf.it',
      author='Lorenzo Busoni',
      license='',
      keywords='adaptive optics',
      packages=['appoppy',
                ],
      install_requires=["numpy",
                        "poppy",
                        "astropy",
                        ],
      test_suite='test',
      )
