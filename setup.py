
# python setup.py sdist
# pip install twine

import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 7):
    sys.exit('Sorry, Python < 3.7 is not supported.')

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="quad_sim_python",
    packages=[package for package in find_packages()],
    scripts=['examples/run_online.py','examples/run_trajectory.py'],
    version="0.0.1",
    license="MIT",
    description="Simple quadcopter simulator in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Ricardo de Azambuja",
    author_email="ricardo.azambuja@gmail.com",
    url="https://github.com/ricardodeazambuja/quad_sim_python",
    download_url="https://github.com/ricardodeazambuja/quad_sim_python/archive/refs/tags/v0.0.1.tar.gz",
    keywords=['CogniFly', 'Betaflight', 'iNAV', 'drone', 'UAV', 'Multi Wii Serial Protocol', 'MSP', 'Quadcopter'],
    install_requires=['numpy>=1.20', 'scipy>=1.6'],
    classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Information Technology',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python',
          'Framework :: Robot Framework :: Library',
          'Topic :: Education',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)