#!/bin/bash

cifti_in=${1}

module load workbench/1.5.0

wb_command -file-information -only-step-interval ${cifti_in}

