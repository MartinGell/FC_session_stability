#!/bin/bash

#TR=$(wb_command -file-information ${dtseries_in} -only-step-interval) 
#WB_CMD='/common/software/install/migrated/workbench/1.5.0/bin_rh_linux64/wb_command'

# Note: This script uses the `eval` command to construct and execute the `wb_command` call with dynamic arguments. 

module load workbench/1.5.0

cifti_out=${1}
# Remove the first argument (output file) from the list of arguments to dynamically construct the call
shift

# Construct the wb_command arguments dynamically
wb_command_args="-cifti-merge ${cifti_out}"
for cifti in "$@"; do
    wb_command_args+=" -cifti ${cifti}"
done

echo -e "\n\nConcatenating ciftis using wb cmd..."  
echo -e ${wb_command_args}

# Run the workbench command
eval "wb_command ${wb_command_args}"

echo -e "Done!"

