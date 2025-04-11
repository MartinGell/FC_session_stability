#!/bin/bash

module load workbench/1.5.0

echo -e "\n\nCorrelating timeseries using wb cmd..." 

# Arguments
cifti_out=${1}
pconn=${2}
motion=${3}

# Run the correlation command
#wb_command -cifti-correlation ${cifti_out} ${pconn} -weights ${motion} -fisher-z

# Build the arguments
args="-cifti-correlation ${cifti_out} ${pconn} -weights ${motion}"

# Run and print
wb_command ${args}
echo -e ${args}

echo -e "Done!"