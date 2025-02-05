import numpy as np
from setuptools import setup, find_packages
from Cython.Build import cythonize

setup(
    name="momospecsim",
    version="0.1",
    author="MazinLab, J. Bailey, C. Kim",
    ext_modules=cythonize("filterphot.pyx"),
    include_dirs=[np.get_include()],
    author_email="mazinlab@ucsb.edu",
    description="A UVOIR MKID Echelle Spectrograph Simulator",
    long_description_content_type="text/markdown",
    url="https://github.com/MazinLab/MOMOSpecSim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research"],
)
