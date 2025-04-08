#!/bin/bash

#TR=$(wb_command -file-information ${dtseries_in} -only-step-interval) 
#WB_CMD='/common/software/install/migrated/workbench/1.5.0/bin_rh_linux64/wb_command'

s3loc=${1}
cifti_in=${2}

#echo -e "\nGetting ${cifti_in}"

s3cmd get ${s3loc} ${cifti_in} --skip-existing

echo -e "Done!" 