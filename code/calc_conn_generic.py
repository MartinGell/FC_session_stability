
import glob
import os
import subprocess
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np

from functions.handling_outliers import isthisanoutlier
from functions.utils import filter_output
from functions.dconn_shrinker import dconn_to_hdf5


######## OPTIONS ########
# S3 Bucket location
datasetdir = 's3://btc-r01prelim/processed/xcpd'

# name the directory to save data to
dataset = 'r01prelim' #'subpop'

# Motion filter options
fd_threshold = 0.2

# Outlier removal options
remove_outliers = True

# Smoothing options
smooth = True
smoothing_kernel = 2


## File identification options ##
# HELPER: sub-XXXXXX_ses-X_task-{task}_space-fsLR_{metric}.{ext_in}
task = 'restMENORDICrmnoisevols'
metric = 'den-91k_desc-interpolated_bold'
run = '' # if only one run, specify - e.g.: 'run-01', otherwise leave as empty string

# file extensions. If using dense files use dtseries.nii and dconn.nii
# If using parcellated files use ptseries.nii and pconn.nii
ext_in = 'dtseries.nii'
ext_out = 'dconn.nii'

# Name of the QC file which contains the framewise displacement (FD) values
qc = 'dcan_qc' # 'abcc_qc'

# List of subjects and sessions to process, excluding the 'sub-' and prefix
sub = ['412001']
ses = ['ses-3']
###########################



# set up for naming purposes
fd_str = str(fd_threshold).replace('.', '')
smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''

# Prep
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out = wd / 'input'

for sub_i in sub:
    print(f'\n\nSub: {sub_i}')

    outdir = out / dataset / f'sub-{sub_i}'
    # Create the directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)

    for ses_i in ses:
        print(f'Session: {ses_i}')

        outfile = outdir / ses_i
        # Create the directory if it doesn't exist
        outfile.mkdir(parents=True, exist_ok=True)

        print('\nGetting data...')
        #s3_loc =  f'{datasetdir}/{sub_i}/sub-{sub_i}/{ses_i}'
        s3_loc =  f'{datasetdir}/sub-{sub_i}/xcp_d/sub-{sub_i}/{ses_i}'
        run_i = f'{run}_' if run else ''

        # BOLD
        s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}space-fsLR_{metric}.{ext_in}'
        #file_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'
        cifti_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'

        output = subprocess.run(['./get_data.sh',str(s3_file),str(cifti_in)], capture_output=True, text=True, check=True)
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{output.stdout.strip()}")        

        # Motion
        s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}desc-{qc}.hdf5'
        #file_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'
        motion_file = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-{qc}.hdf5'

        output = subprocess.run(['./get_data.sh',str(s3_file),str(motion_file)], capture_output=True, text=True, check=True)
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{output.stdout.strip()}")

        # Load the motion file and extract the binary mask indicating frame removal based on framewise displacement (FD) threshold.
        with h5py.File(motion_file, 'r') as f:
            # Extract the binary mask indicating frame removal based on framewise displacement (FD) threshold.
            # Frames with FD > threshold are marked as 1 (removed), and frames with FD <= fd_threshold are marked as 0 (kept).
            motion = f['dcan_motion'][f'fd_{fd_threshold}']['binary_mask'][()].astype(int)
            TR = f['dcan_motion']['fd_0.2']['remaining_seconds'][()]/f['dcan_motion']['fd_0.2']['remaining_total_frame_count'][()]
            print(f"TR: {TR}")
        
        #motion = np.concatenate(motion_list)
        inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
        motion_file = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}.txt'
        print('\nSaving concatenated motion file...')
        print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}')
        np.savetxt(motion_file, inverted_motion, fmt="%d")

        # optionally identify outliers. Removal happens when creating the d/pconn
        if remove_outliers:
            # First run wb_command and load the std.txt file as its faster
            print('\nIdentifying outliers using the median approach...')
            std_txt = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std.txt'
            stats_args = ['./cifti_std.sh', str(cifti_in), str(std_txt)]
            output = subprocess.run(stats_args, capture_output=True, text=True)
            #print(f'{stats_args}')
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            # Next check if there are nans in the data and if so use numpy instead of wb_command
            std = np.loadtxt(std_txt)
            if np.isnan(std).any():
                print('Nan values in wb cmd file, using numpy instead...')
                X = nb.load(f'{cifti_in}')
                concat_cifti = X.get_fdata()
                stdevnp = np.nanstd(concat_cifti,axis=1).round(5)
                std = stdevnp
                # save
                std_txt = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_np.txt'
                np.savetxt(std_txt, std, fmt='%.5f')

            print('\nFlagging outliers...')
            [outlier,_,_,_] = isthisanoutlier(std)
            # turn outlier into a binary mask (0 = remove, 1 = keep)
            outlier = outlier.astype(int)
            inverted_outlier = 1-outlier
            outlier_file = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_outlier.txt'
            np.savetxt(outlier_file, inverted_outlier, fmt='%d')
            print(f'Flagged {np.sum(outlier).astype(int)} additional frames as outliers.')

            # combine motion and outlier files
            print('\nCombining motion and outlier files and saving...')
            combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
            combined_file = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}_and_outliers_combined.txt'
            np.savetxt(combined_file, combined, fmt="%d")
            motion_file = combined_file
            minutes = sum(combined)*TR/60
            print(f'Participant left with {minutes} minutes of data in this session.')

        # Smooth if necessary
        if smooth:
            # First need to get the midthickness surfaces
            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
            surf_R = outfile / 'anat' / f'sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")

            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'
            surf_L = outfile / 'anat' /f'sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_L)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")

            # Now actually smooth            
            print('\nSmoothing...')
            smooth_out = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_smoothed_{smoothing_kernel}.{ext_in}'
            smooth_args = ['./cifti_smooth.sh', str(cifti_in), str(smooth_out), str(smoothing_kernel), str(surf_L), str(surf_R)]
            output = subprocess.run(smooth_args, capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            cifti_in = smooth_out

        # create p/dconn
        print('\nCreating p/dconn...')
        conn = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(cifti_in), str(conn), str(motion_file)]
        output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
        #print(f'{correlation_args}')
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        if ext_out == 'dconn.nii':
            # Convert dconn to hdf5
            print('Converting dconn to hdf5...')
            dconn_to_hdf5(conn)
            # and now remove the dconn file
            print('Removing dconn file...')
            print(f"{conn} --> deleted.")
            os.remove(conn)

