#!/bin/bash

#TR=$(wb_command -file-information ${dtseries_in} -only-step-interval) 
#WB_CMD='/common/software/install/migrated/workbench/1.5.0/bin_rh_linux64/wb_command'

# Note: This script uses the `eval` command to construct and execute the `wb_command` call with dynamic arguments. 

module load workbench/1.5.0

cifti_out=${1}

cifti_1=${2}
cifti_2=${3:-""}  # Use an empty string if not provided
cifti_3=${4:-""}  # Use an empty string if not provided
cifti_4=${5:-""}  # Use an empty string if not provided


# Construct the wb_command arguments dynamically
wb_command_args="-cifti-merge ${cifti_out}"
if [[ -n ${cifti_1} ]]; then
    wb_command_args+=" -cifti ${cifti_1}"
fi
if [[ -n ${cifti_2} ]]; then
    wb_command_args+=" -cifti ${cifti_2}"
fi
if [[ -n ${cifti_3} ]]; then
    wb_command_args+=" -cifti ${cifti_3}"
fi
if [[ -n ${cifti_4} ]]; then
    wb_command_args+=" -cifti ${cifti_4}"
fi


echo -e "\n\nConcatenating ciftis using wb cmd..."  
echo -e ${wb_command_args}

# Run the workbench command
eval "wb_command ${wb_command_args}"

echo -e "Done!"

