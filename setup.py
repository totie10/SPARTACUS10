#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pathlib
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='SPARTACUS',
    version='0.1.2',
    license='MIT',
    description='A package to perform spatial hierarchical agglomerative clustering as well as spatially constrained ensemble clustering. Further includes implementations of the silhouette coefficient, the simplified silhouette coefficient and spatial adaptations thereof.',
    long_description=README,
    long_description_content_type="text/markdown",
    author='Tobias Tietz',
    author_email='tobias.tietz10@gmail.com',
    url='https://github.com/totie10/SPARTACUS',
    #packages=find_packages('src', exclude=("tests",)),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Utilities',
    ],
    project_urls={
        # 'Documentation': 'https://SPARTACUS.readthedocs.io/',
        # 'Changelog': 'https://SPARTACUS.readthedocs.io/en/latest/changelog.html',
        # 'Issue Tracker': 'https://github.com/totie10/SPARTACUS/issues',
    },
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    python_requires='>=3.6',
    install_requires=[# TODO versioning
        "numpy", "scipy", "scikit-learn"
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={# Put here requires needed for testing
                    'dev': ['pytest']
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        # 'console_scripts': [
        #     'SPARTACUS = SPARTACUS.cli:main',
        # ]
        # 'console_scripts': [
        #     'SPARTACUS = SPARTACUS.__main__:main',
        # ]
    },
)
