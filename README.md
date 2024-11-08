# The MKID Echelle Spectrograph Simulator and Data Reduction Pipeline
### Introduction

This package contains 2 core scripts:
1) `simulate.py` reduces a model spectrum to a photon table in a simulated optical path.

2) `mkidspec.py` will reduce an observation photon table into a spectrum using 3 photon tables, one each of a continuum, an emission lamp, and, of course, the observation.
   
     a. MSF: The continuum photon table is separated into pixels, Gaussian models are fit to the orders, and the virtual
        pixel boundaries are extracted. The covariance among orders is calculated. These various properties make up the
        MKID Spread Function.
   
     b. Order-sorting: The other 2 photon tables (emission lamp and observation) are sorted using the MSF.
   
     c. Wavecal: The sorted emission lamp spectrum is compared to its atlas to determine the wavecal solution.
   
     d. Extraction: The observed star is updated with the wavecal solution and extraction is complete.

### Tutorial on running sample simulations and extracting the final observation:

Note: an [MKIDPipeline](https://github.com/MazinLab/MKIDPipeline) enviroment is required to execute these steps without issue.

After cloning the repository, the Cython files need to be compiled. Go into `KIDSpecSim` and run the following lines: 

`conda activate pipeline`

`pip install -e .`

`cd ucsbsim/scripts`


#### Spectrum simulation steps:

Now that everything is ready, run the following to obtain a continuum photon table:

`python simulate.py --type_spectra flat`

Run the following to obtain a HgAr lamp photon table:

`python simulate.py --type_spectra emission -sf ../mkidspec/linelists/hgar.csv`

Run the following to obtain a Phoenix star model photon table:

`python simulate.py --type_spectra phoenix --on_sky`


#### Now to recover the PHOENIX model spectrum, run:

`python mkidspec.py`


   #### Help: 
      
      Run
      
      python simulate.py --help
      
      or
      
      python mkidspec.py --help
      
      to see the passable arguments.


   #### Passing a file:
   
      Writing a .txt file containing command line arguments exactly as shown is 
      another option. For example, to get a PHOENIX model spectrum:

      python simulate.py --option_file path/to/option_file.txt

      where option_file.txt is a file that contains:

      --type_spectra phoenix

      --on_sky
