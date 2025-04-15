
import glob
import os
import subprocess
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import shutil

from functions.handling_outliers import isthisanoutlier
from functions.utils import filter_output


######## OPTIONS ########
# S3 Bucket location
datasetdir = 's3://btc-hcp-ya/processed/xcpd_v0.10.1'

# name the directory to save data to
dataset = 'HCP_YA' #'HCPtrt_cneuro' #'subpop' 

# Motion filter options
fd_threshold = 0.2

# Outlier removal options
remove_outliers = True

# Smoothing options
smooth = False
smoothing_kernel = 2

# remove subjects with too few minutes left after filtering
min_minutes = 45
remove_subjects = True


## File identification options ##
# HELPER: sub-XXXXXX_ses-X_task-{task}_space-fsLR_{metric}.{ext_in}
# task = 'restMENORDICtrimmed'
# metric = 'den-91k_desc-denoised_bold'
# ext_in = 'dtseries.nii'
# ext_out = 'dconn.nii'

task = 'rest'
metric = 'seg-Glasser_den-91k_stat-mean_timeseries'
ext_in = 'ptseries.nii'
ext_out = 'pconn.nii'


### Subjects, sessions and runs ###
#sub = ['sub-753150'] 
sub = pd.read_csv('/home/btervocl/shared/projects/martin_FC_stability/code/sublist/Unrelated_S900_Subject_multilist1_with_physio.csv')['Subject'].tolist()
ses = ['ses-3T']  # in order to find the file it will have to be: 'combined' but would be better to rename this to ses-1 ...
run = ['run-1','run-2','run-3','run-4']
######## END OF OPTIONS ########


#'s3://btc-hcp-ya/processed/xcpd_v0.10.1/sub-753150/sub-753150/ses-3T/func/sub-753150_ses-3T_task-restMENORDICtrimmed_run-01_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii'
#'s3://btc-hcp-ya/processed/xcpd_v0.10.1/sub-753150/sub-753150/ses-3T/func/sub-753150_ses-3T_task-rest_run-4_space-fsLR_seg-Glasser_den-91k_stat-mean_timeseries.ptseries.nii

# set up for naming purposes
fd_str = str(fd_threshold).replace('.', '')
smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''

# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out = wd / 'data'

# save all subjects with too few minutes left after filtering
subs_to_remove = []

# Get data and create d/pconn
for sub_i in sub:
    print(f'\n\nSub: {sub_i}')

    outdir = out / dataset / f'sub-{sub_i}'
    # Create the directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    for ses_i in ses:
        print(f'\nSession: {ses_i}')

        outfile = outdir / ses_i
        # Create the directory if it doesn't exist
        outfile.mkdir(parents=True, exist_ok=True)

        print('\nGetting data...')
        for run_i in run:
            print(f'File: {run_i}')

            s3_loc =  f'{datasetdir}/sub-{sub_i}/sub-{sub_i}/{ses_i}'

            # BOLD
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'
            file_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(file_in)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                if output.stderr.__contains__('does not exist'):
                    continue
                else:
                    raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")        

            # Motion
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'
            file_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(file_in)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")


        # concat all of ses-x time series
        print('\nConcatenating d/pseries...')
        runs = glob.glob(f'{outfile}/func/*{metric}.{ext_in}')
        cifti_out = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}.{ext_in}'
        merge_args = ['./cifti_merge.sh', str(cifti_out)]
        merge_args.extend([str(run) for run in runs if run is not None])  # Only include non-None runs
        output = subprocess.run(merge_args, capture_output=True, text=True, check=True)
        # should look like this: ./concat_cifti_merge.sh cifti_out run1 run2 ...
        if output.stderr.strip():
            if output.stderr.__contains__('no inputs specified'):
                print('\nNo data for subject found.')
                print('\n\nSKIPPING!!!!\n\n')
                continue
            else:
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{output.stdout.strip()}")        

        # concat all motion files
        motion_files = glob.glob(f'{outfile}/func/*.hdf5')
        motion_list = []
        for m_file_i in motion_files:
            with h5py.File(m_file_i, 'r') as f:
                # Extract the binary mask indicating frame removal based on framewise displacement (FD) threshold.
                # Frames with FD > threshold are marked as 1 (removed), and frames with FD <= fd_threshold are marked as 0 (kept).
                m_i = f['dcan_motion'][f'fd_{fd_threshold}']['binary_mask'][()].astype(int)
                TR = f['dcan_motion']['fd_0.2']['remaining_seconds'][()]/f['dcan_motion']['fd_0.2']['remaining_total_frame_count'][()]
                print(f"TR: {TR}")
            motion_list.append(m_i)

        motion = np.concatenate(motion_list)
        inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
        motion_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}.txt'
        print('\nSaving concatenated motion file...')
        print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}')
        np.savetxt(motion_file, inverted_motion, fmt="%d")

        # optionally identify outliers. Removal happens when creating the d/pconn
        if remove_outliers:
            # First run wb_command and load the std.txt file as its faster
            print('\nIdentifying outliers using the median approach...')
            std_txt = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std.txt'
            stats_args = ['./cifti_std.sh', str(cifti_out), str(std_txt)]
            output = subprocess.run(stats_args, capture_output=True, text=True)
            #print(f'{stats_args}')
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            # Next check if there are nans in the data and if so use numpy instead of wb_command
            std = np.loadtxt(std_txt)
            if np.isnan(std).any():
                print('Nan values in wb cmd file, using numpy instead...')
                X = nb.load(f'{cifti_out}')
                concat_cifti = X.get_fdata()
                stdevnp = np.nanstd(concat_cifti,axis=1).round(5)
                std = stdevnp
                # save
                std_txt = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_np.txt'
                np.savetxt(std_txt, std, fmt='%.5f')

            print('\nFlagging outliers...')
            [outlier,_,_,_] = isthisanoutlier(std)
            # turn outlier into a binary mask (0 = remove, 1 = keep)
            outlier = outlier.astype(int)
            inverted_outlier = 1-outlier
            outlier_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_outlier.txt'
            np.savetxt(outlier_file, inverted_outlier, fmt='%d')
            print(f'Flagged {np.sum(outlier).astype(int)} additional frames as outliers.')

            # combine motion and outlier files
            print('\nCombining motion and outlier files and saving...')
            combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
            combined_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}_and_outliers_combined.txt'
            np.savetxt(combined_file, combined, fmt="%d")
            motion_file = combined_file
            minutes = sum(combined)*TR/60
            print(f'Participant left with {minutes} minutes of data in this session.')

        # Optionally collect subjects with too few minutes left after filtering
        if remove_subjects:
            if minutes < min_minutes:
                print(f"Participant {sub_i} left with {minutes} minutes of data in this session. Will remove!!")
                subs_to_remove.append(sub_i)

        # Smooth if necessary
        if smooth:
            # First need to get the midthickness surfaces
            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
            surf_R = outfile / 'anat' / f'sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'

            if not os.path.isfile(surf_R):
                output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)
                if not os.path.isfile(surf_R):
                    s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
                    output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)

                if output.stderr.strip():
                    raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
                print(f"{output.stdout.strip()}")

            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'
            surf_L = outfile / 'anat' /f'sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'

            if not os.path.isfile(surf_L):
                output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_L)], capture_output=True, text=True, check=True)
                if not os.path.isfile(surf_L):
                    s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'
                    output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_L)], capture_output=True, text=True, check=True)

                if output.stderr.strip():
                    raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
                print(f"{output.stdout.strip()}")

            # Now actually smooth            
            print('\nSmoothing...')
            smooth_out = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_smoothed_{smoothing_kernel}.{ext_in}'
            smooth_args = ['./cifti_smooth.sh', str(cifti_out), str(smooth_out), str(smoothing_kernel), str(surf_L), str(surf_R)]
            output = subprocess.run(smooth_args, capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            cifti_out = smooth_out

        # create p/dconn
        print('\nCreating p/dconn...')
        pconn = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(cifti_out), str(pconn), str(motion_file)]
        output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
        #print(f'{correlation_args}')
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        
        # remove individual runs and keep only concatenated session:
        print('\nRemoving individual run data...')
        for run_j in runs:
            if os.path.exists(run_j):
                os.remove(run_j)
            print(f"{run_j} --> deleted.")
            
        print('\nRemoving individual motion data...')
        for motion_j in motion_files:
            if os.path.exists(motion_j):
                os.remove(motion_j)
            print(f"{motion_j} --> deleted.")


# Now remove folders with subjects with too few minutes left after filtering
if remove_subjects:
    print('Removing subjects with too few minutes left after filtering...')
    print(f"Will remove {len(subs_to_remove)} subjects.")
    for sub_i in subs_to_remove:
        print(f"Removing subject {sub_i} from dataset.")
        subdir = out / dataset / f'sub-{sub_i}'
        if os.path.exists(subdir):
            shutil.rmtree(subdir)
        print(f"{subdir} --> deleted.")

