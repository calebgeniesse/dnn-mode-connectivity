#!/bin/bash

# process landscapes with coordinate based information
python3 batchProcessing.py --input=/paraview_files_dim3/
python3 batchProcessing.py --input=/paraview_files_unet/

# process landscapes with 1D profile only
python3 batchProcessing_profileOnly.py --input=/paraview_files_artificial/
python3 batchProcessing_profileOnly.py --input=/paraview_files_CSGLD/
python3 batchProcessing_profileOnly.py --input=/paraview_files_dim2/
python3 batchProcessing_profileOnly.py --input=/paraview_files_dim3/
python3 batchProcessing_profileOnly.py --input=/paraview_files_dim4/

# process landscapes with 1D profile only and iterate over all files
python3 batchProcessing_profileOnly_iter.py --input=/paraview_CSGLD/

echo "Finished processing all landscapes processing and generated 1D profile information files."
