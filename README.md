# The Multi-Object MKID Optical Spectrometer (MOMOS) Simulator and Data Reduction Pipeline
### Introduction

This package is described in depth in [this AAS paper](insertlinklater). It contains 2 core scripts:
1) `simulate.py` reduces a model spectrum to a photon table in a simulated optical path.

2) `reduce.py` will reduce an observation photon table into a spectrum using 3 photon tables, one each of a continuum, an emission lamp, and, of course, the observation.
   
     a. MSF: The continuum photon table is separated into pixels, Gaussian models are fit to the orders, and the virtual
        pixel boundaries are extracted. The covariance among orders is calculated. These various properties make up the
        MKID Spread Function.
   
     b. Order-sorting: The other 2 photon tables (emission lamp and observation) are sorted using the MSF.
   
     c. Wavecal: The sorted emission lamp spectrum is compared to its line atlas to determine the wavecal solution.
   
     d. Extraction: The observed star is updated with the wavecal solution and extraction is complete.

### Read the [MOMOSpecSim User Manual](insertlinklater) for instructions on running and reducing sample simulations.
