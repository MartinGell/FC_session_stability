
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
from functions.utils import filter_output, build_subject_session_run_map
from functions.dconn_shrinker import dconn_to_hdf5


######## OPTIONS ########
# S3 Bucket location
#datasetdir = 's3://subpop/derivatives/xcpd/output'
datasetdir = 's3://subpop-v1.0/derivatives/xcpd0.12.0/output'

# total minutes of data to keep per session
total_minutes = 25

# name the directory to save data to
dataset = f'subpop{total_minutes}_UMN' #'subpop'

# Motion filter options
fd_threshold = 0.2

# Outlier removal options
remove_outliers = True

# Smoothing options
smooth = False
smoothing_kernel = 1.7


## File identification options ##
# HELPER: sub-XXXXXX_ses-X_task-{task}_space-fsLR_{metric}.{ext_in}
# task = 'restMENORDICtrimmed'
# metric = 'den-91k_desc-denoised_bold'
# ext_in = 'dtseries.nii'
# ext_out = 'dconn.nii'

task = 'restNORDIC'
metric = 'seg-Glasser_den-91k_stat-mean_timeseries'
ext_in = 'ptseries.nii'
ext_out = 'pconn.nii'
######## END OF OPTIONS ########




# set up for naming purposes
fd_str = str(fd_threshold).replace('.', '')
smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''

# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out = wd / 'data'


# ### FOR TESTING ###
# sub_i = '1004401'
# new_ses = 'ses-3'
# runs = ['run-04', 'run-05', 'run-06']
# run_i = 'run-04'
# ###################

### Subjects, sessions and runs ###
sublist = wd / 'code/sublist/Subpop_v1_sub_ses_3plus_run_25min.csv'
sub_ses_run_map = build_subject_session_run_map(sublist)
ses_combined = 'ses-combined'  # in order to find the file it will have to be: 'combined' but would be better to rename this to ses-1 ...

# Get data and create d/pconn
for s_i, ses_dict in sub_ses_run_map.items():
    sub_i = s_i.split('-')
    sub_i = sub_i[1]+sub_i[2]

    print(f'\n\n\n\nSub: {sub_i}')

    outdir = out / dataset / f'sub-{sub_i}'
    # Create the directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    for new_ses, runs in ses_dict.items():
        ses_i = ses_combined
        print(f'\n\n\nSession: {ses_i}, creating {new_ses}')
        print(f'Number of runs to concatenate: {runs}')

        outfile = outdir / new_ses
        # Create the directory if it doesn't exist
        outfile.mkdir(parents=True, exist_ok=True)

        # Check how many runs there are for this session and calculate the number of minutes that will be kept per run
        n_runs = len(runs)
        print(f'\nGoing to keep {total_minutes} minutes across all runs')
        print(f'\nFound {n_runs} runs for this session')
        if sub_i == '1007501' and runs == ['run-07', 'run-08', 'run-09'] and total_minutes >= 25:
            # Special case for sub-1007501, ses-3, which has only 5.2 minutes of data in run-09
            shorter_run_minutes = 5.78
            leftover_minutes = total_minutes - shorter_run_minutes
            longer_run_minutes = leftover_minutes * (16/32)
            print(f'Splitting into: {longer_run_minutes} for run 1, {longer_run_minutes} for run 2, and {shorter_run_minutes} for run 3')
            print(f'Which is: {longer_run_minutes * 2 + shorter_run_minutes} minutes in total')
        elif n_runs == 3:
            longer_run_minutes = total_minutes * (16/42)  # longer runs are 16 minutes long
            shorter_run_minutes = total_minutes * (10/42) # shorter run is 10 minutes long
            print(f'Splitting into: {longer_run_minutes} for run 1, {longer_run_minutes} for run 2, and {shorter_run_minutes} for run 3')
            print(f'Which is: {longer_run_minutes * 2 + shorter_run_minutes} minutes in total')
        elif n_runs == 2:
            longer_run_minutes = total_minutes * (16/32)
            print(f'Splitting into: {longer_run_minutes} for run 1, and {longer_run_minutes} for run 2')
            print(f'Which is: {longer_run_minutes * 2} minutes in total')
        else:
            raise ValueError(f"Unexpected number of runs: {n_runs}. Expected 2 or 3.")

        # Now get run data and motion files prepare for shortening and concatenation
        print('\n\nGetting data...')
        for run_i in runs:
            print(f'File: {run_i}')

            s3_loc =  f'{datasetdir}/{sub_i}/sub-{sub_i}/{ses_i}'

            # BOLD
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'
            cifti_out = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(cifti_out)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")

            # Motion
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'
            motion_file_run = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_desc-abcc_qc.hdf5'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(motion_file_run)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")

            # Load the motion file
            with h5py.File(motion_file_run, 'r') as f:
            # Extract the binary mask indicating frame removal based on framewise displacement (FD) threshold.
            # Frames with FD > threshold are marked as 1 (removed), and frames with FD <= fd_threshold are marked as 0 (kept).
                motion = f['dcan_motion'][f'fd_{fd_threshold}']['binary_mask'][()].astype(int)
                total_frames = f['dcan_motion']['fd_0.2']['total_frame_count'][()]
                remaining_frames = f['dcan_motion']['fd_0.2']['remaining_total_frame_count'][()]
                TR = f['dcan_motion']['fd_0.2']['remaining_seconds'][()]/remaining_frames
                print(f"TR: {TR}") # Should be 1.761

            TR = 1.761 # hardocode in case its calculated incorrectly 
            
            inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
            motion_file_run = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_desc-FD_{fd_str}.txt'
            print('\nSaving extracted motion file...')
            print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}') # Here counting non-inverted motion
            np.savetxt(motion_file_run, inverted_motion, fmt="%d")

            # optionally identify outliers. Removal happens when creating the d/pconn
            if remove_outliers:
                # First run wb_command and load the std.txt file as its faster
                print('\nIdentifying outliers using the median approach...')
                std_txt = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_space-fsLR_{metric}_std.txt'
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
                    std_txt = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_space-fsLR_{metric}_std_np.txt'
                    np.savetxt(std_txt, std, fmt='%.5f')

                print('\nFlagging outliers...')
                [outlier,_,_,_] = isthisanoutlier(std)
                # turn outlier into a binary mask (0 = remove, 1 = keep)
                outlier = outlier.astype(int)
                inverted_outlier = 1-outlier
                outlier_file = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_space-fsLR_{metric}_std_outlier.txt'
                np.savetxt(outlier_file, inverted_outlier, fmt='%d')
                n_extra_outliers = np.sum(outlier).astype(int)
                print(f'Flagged {n_extra_outliers} additional frames as outliers.')

                # combine motion and outlier files
                print('\nCombining motion and outlier files and saving...')
                combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
                combined_file = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_desc-FD_{fd_str}_and_outliers_combined.txt'
                
                minutes = sum(combined)*TR/60
                print(f'\nParticipant has {minutes} minutes of data in this run.')

                ##### Isnt the maximum number of frames written in the motion file????
                this_run_total_minutes = total_frames * TR / 60
                if this_run_total_minutes > 12: 
                    # longer runs are 16 minutes long so we can assume if the run has more than 12 minutes it is a longer run
                    x_min = longer_run_minutes
                else:
                    x_min = shorter_run_minutes
                    
                print(f'\nShortening this run to {x_min} minutes of data...')
                max_frames = int(np.round(x_min*60/TR, 0)) # maximum frames for x_min minutes at TR = 1.761
                one_indices = np.flatnonzero(combined)
                if len(one_indices) >= max_frames:
                    cut_off_index = one_indices[max_frames-1]
                    print(f'\nCutting off at index {cut_off_index} to keep only the first ~ {x_min} minutes ({max_frames} frames).')
                    combined[cut_off_index + 1:] = 0
                elif len(one_indices) <= max_frames and n_runs == 2:
                    print(f'\n\n\n\nWARNING !!!!!!!!!!!!!')
                    print(f'Subject has only 2 runs and one does not have enough frames to cut off to {x_min} minutes.')
                    print(f'Found {len(one_indices)} frames, Taking all.\n\n\n\n')
                else:
                    print(f'\n\n\n\nWARNING !!!!!!!!!!!!!')
                    print(f'Not enough frames to cut off to {x_min} minutes. Found {len(one_indices)} frames, expected at least {max_frames}.\n\n\n\n')
                    print(f'Will take all frames instead.')
                    # raise ValueError(
                    #     f"Not enough frames to cut off to 10 minutes. Found {len(one_indices)} frames, expected at least {max_frames}."
                    # )

                minutes = sum(combined)*TR/60
                print(f'Participant now left with {minutes} minutes of data in this run.')

                print(f'\nSaving shortened motion file...\n')
                np.savetxt(combined_file, combined, fmt="%d")

                motion_file_run = combined_file # make sure to use the combined file if outliers were removed for the next steps


        # concat all motion files
        print('\n______________________DONE_WITH_INDIVIDUAL_RUNS______________________')
        print('\nConcatenating motion files...')
        ending = motion_file_run.as_posix().split('-').pop()
        motion_files = glob.glob(f'{outfile}/func/*{ending}')
        motion_list = []
        # load all motion files
        for m_file_i in motion_files:
            m_i = np.loadtxt(m_file_i, dtype=int)
            motion_list.append(m_i)
        motion = np.concatenate(motion_list)
        # Check that shortening worked
        minutes = sum(motion)*TR/60
        # This is already INVERTED for wb_command: 0 = remove, 1 = keep
        print(f'\n\nAFTER CONCATENATION participant left with {minutes} minutes of data.')
        print(f'This amounts to {np.sum(motion).astype(int)} frames out of {len(motion)}') # Here summing inverted motion
        # Save
        motion_file = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_desc-FD_{fd_str}.txt'
        print('\nSaving concatenated motion file...')
        np.savetxt(motion_file, motion, fmt="%d")


        # concat all of ses-x time series
        print('\nConcatenating d/pseries...')
        runs = glob.glob(f'{outfile}/func/*{metric}.{ext_in}')
        cifti_out = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}.{ext_in}'
        merge_args = ['./cifti_merge.sh', str(cifti_out)]
        merge_args.extend([str(run) for run in runs if run is not None])  # Only include non-None runs
        output = subprocess.run(merge_args, capture_output=True, text=True, check=True)
        # should look like this: ./concat_cifti_merge.sh cifti_out run1 run2 ...
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")


        # Smooth if necessary
        if smooth:
            # First need to get the midthickness surfaces
            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
            surf_R = outfile / 'anat' / f'sub-{sub_i}_{new_ses}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'

            if not os.path.isfile(surf_R):
                output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)
                if not os.path.isfile(surf_R):
                    s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
                    output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)

                if output.stderr.strip():
                    raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
                print(f"{output.stdout.strip()}")

            s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'
            surf_L = outfile / 'anat' /f'sub-{sub_i}_{new_ses}_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'

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
            smooth_out = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_smoothed_{smoothing_kernel}.{ext_in}'
            smooth_args = ['./cifti_smooth.sh', str(cifti_out), str(smooth_out), str(smoothing_kernel), str(surf_L), str(surf_R)]
            output = subprocess.run(smooth_args, capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            cifti_out = smooth_out


        # create p/dconn
        print('\nCreating p/dconn...')
        conn = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(cifti_out), str(conn), str(motion_file)]
        output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
        #print(f'{correlation_args}')
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        # # Convert to hdf5 if making dconns
        # if ext_out == 'dconn.nii':
        #     # Convert dconn to hdf5
        #     print('Converting dconn to hdf5...')
        #     dconn_to_hdf5(conn)
        #     # and now remove the dconn file
        #     print('Removing dconn file...')
        #     print(f"{conn} --> deleted.")
        #     os.remove(conn)
        
        # remove individual runs and keep only concatenated session:
        print('\nRemoving individual run and motion data by deleting the func folder...')
        func_folder = outfile / 'func'
        if os.path.exists(func_folder):
            shutil.rmtree(func_folder)
            print(f"{func_folder} --> deleted.")

print('\n\nFINISHED!!')
