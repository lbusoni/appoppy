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
      license='MIT',
      keywords='adaptive optics',
      packages=['appoppy',
                'appoppy.mains',
                ],
      install_requires=["numpy",
                        "poppy",
                        "astropy",
                        "scikit-image",
                        ],
      test_suite='test',
      package_data={
          'appoppy': ['data/*'],
      },
      include_package_data=True,
      )
