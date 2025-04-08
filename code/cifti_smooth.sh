#!/bin/bash

module load workbench/1.5.0

echo -e "\n\n Smoothing timeseries using wb cmd..." 

# Arguments
cifti_in=${1}
cifti_out=${2}
kernel_surface=${3}
kernel_volume=${3}
surf_L=${4}
surf_R=${5}

# Build the arguments
args="-cifti-smoothing ${cifti_in} ${kernel_surface} ${kernel_volume} COLUMN ${cifti_out} -left-surface ${surf_L} -right-surface ${surf_R}"

# Run and print
wb_command ${args}
echo -e ${args}

echo -e "Done!"
