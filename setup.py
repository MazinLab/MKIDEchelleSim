import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="ucsbsim",
    version="0.1",
    author="MazinLab, J. Bailey, C. Kim",
    ext_modules=cythonize("ucsbsim/filterphot.pyx"),
    include_dirs=[np.get_include()],
    author_email="mazinlab@ucsb.edu",
    description="A UVOIR MKID Echelle Simulator",
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MKIDEchelleSim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"],
)
