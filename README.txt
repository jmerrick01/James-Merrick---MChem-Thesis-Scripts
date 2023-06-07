# James-Merrick---MChem-Thesis-Scripts

In each script, code taken from external scripts is duly credited; additional authors contributing to each script are detailed below. 
Each Python script is also summarised below:

- 'centroid files with delay and filter': This is a script ran for all run numbers (and their associated binary and HDF5 files) prior to covariance
   analysis. Included is the centroiding algorithm and all data-processing and filtering mentioned in the Methods chapter. The final concatenated 
   data output is saved as a .npy file. Note that this script was run on the Maxwell cluster, operated at DESY, in order to access raw PImMS and 
   DAQ data files. Contributors to cells within this code include: James Unwin and Emily Warne. The main Cython-based centroiding function was 
   developed by Felix Allum.
   
- 'Recoil-frame and radial-frame covariance (user-friendly)': This is a script which takes in the .npy output file from the 'centroid files with
   delay and filter' script as an input. It can also take in the output file of the CEI simulation script detailed below. Included is a function to
   centre and plot VMI ion-images, and the code for performing recoil- and radial-frame covariance calculations. This code was partially developed
   from code written by Emily Warne.
   
- 'ToF ToF covariance slices and yield':  A script for calculating and plotting 2D-TOF covariances for a given data-set. It takes in the .npy output
   file from the 'centroid files with delay and filter' script as an input. The main covariance function was developed by Tiffany Walmsley.
   
- 'Three-dimensional TOF covariance Parallelised': A script for calculating and plotting 3D-TOF covariances for a given data-set. It takes in the
   .npy output file from the 'centroid files with delay and filter' script as an input. The three-fold cumulant functions were developed by James
   Somper, and the parallelisation of the code was implemented by James Unwin.
   
- 'CEI simulation - vibrational motion (220928)': This is the script used to simulate CEI experiments, and it was developed by Louis Minion. It
   uses functions defined in a separate file ('coulombexp_sim_2021_routines') and it takes in a user-defined channels settings .txt
   file as well as a Gaussian .log file following a geometry optimisation calculation (which simultaneously calculates normal modes for the
   molecule under study). For this project, the Gaussian calculation for the geometry optimisation of ethene was performed by Zhihao Liu.
   More information on the CEI simulation software is provided by Minion et al. Test files for ethene are provided in this GitHub repository.
