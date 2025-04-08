#!/bin/bash

module load workbench/1.5.0

echo -e "\n\nCalculating standard deviation using wb cmd..." 

# Arguments
cifti_out=${1}
std_txt=${2}

# Run the statistics command
wb_command -cifti-stats ${cifti_out} -reduce STDEV > ${std_txt}

# Build the arguments
# args="-cifti-stats ${cifti_out} -reduce STDEV > ${std_txt}"

# # Run and print
# wb_command ${args}
# echo -e ${args}

echo -e "Done!"