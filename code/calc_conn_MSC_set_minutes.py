
import glob
import os
import subprocess
from pathlib import Path

import h5py
import nibabel as nb
import numpy as np

from functions.handling_outliers import isthisanoutlier
from functions.utils import filter_output, MSC_build_subject_session_map


######## OPTIONS ########
# S3 Bucket location
datasetdir = 's3://msc/derivatives/xcpd_0_10_7/output'

# total minutes of data to keep per session
total_minutes = 25

# name the directory to save data to
dataset = f'MSC{total_minutes}_4runs'

# Motion filter options
fd_threshold = 0.2

# Outlier removal options
remove_outliers = True

# Smoothing options
smooth = False
smoothing_kernel = 2


## File identification options ##
# HELPER: sub-XXXXXX_ses-X_task-{task}_space-fsLR_{metric}.{ext_in}
task = 'rest'
metric = 'seg-Glasser_den-91k_stat-mean_timeseries'
ext_in = 'ptseries.nii'
ext_out = 'pconn.nii'

run = ''

# Name of the QC file which contains the framewise displacement (FD) values
qc = 'abcc_qc' # 'abcc_qc'
###########################


# set up for naming purposes
fd_str = str(fd_threshold).replace('.', '')
smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''

# Prep
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out = wd / 'data'

# List of subjects and sessions to process, excluding the 'sub-' and prefix
sublist = wd / 'code/sublist/MSC_subs_good_sessions_4runs.csv'
sub_ses_map = MSC_build_subject_session_map(sublist)

for sub_i, ses_list in sub_ses_map.items():
    print(f'\n\n\nSub: {sub_i}')

    outdir = out / dataset / f'sub-{sub_i}'
    # Create the directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)

    for ses_i in ses_list:
        print(f'\n\nSession: {ses_i}')

        outfile = outdir / ses_i
        # Create the directory if it doesn't exist
        outfile.mkdir(parents=True, exist_ok=True)

        print(f'\nGoing to keep {total_minutes} minutes across all runs')
        print('\nMSC ==> only one run for each session')
        print('\nGetting data...')
        #s3_loc =  f'{datasetdir}/{sub_i}/sub-{sub_i}/{ses_i}'
        s3_loc =  f'{datasetdir}/{sub_i}/sub-{sub_i}/{ses_i}'
        run_i = f'{run}_' if run else ''

        # BOLD
        s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}space-fsLR_{metric}.{ext_in}'
        cifti_in = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'
        cifti_in.parent.mkdir(parents=True, exist_ok=True)
        output = subprocess.run(['./get_data.sh',str(s3_file),str(cifti_in)], capture_output=True, text=True, check=True)
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{output.stdout.strip()}")

        # Motion
        s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}desc-{qc}.hdf5'
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
            total_frames = f['dcan_motion']['fd_0.2']['total_frame_count'][()]
            remaining_frames = f['dcan_motion']['fd_0.2']['remaining_total_frame_count'][()]
            TR = f['dcan_motion']['fd_0.2']['remaining_seconds'][()]/remaining_frames
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
            n_extra_outliers = np.sum(outlier).astype(int)
            print(f'Flagged {n_extra_outliers} additional frames as outliers.')

            # combine motion and outlier files
            print('\nCombining motion and outlier files and saving...')
            combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
            combined_file = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}_and_outliers_combined.txt'
            np.savetxt(combined_file, combined, fmt="%d")
            motion_file = combined_file
            this_run_minutes = sum(combined)*TR/60
            percentage = ((remaining_frames-n_extra_outliers)/total_frames)*100
            print(f'Participant left with {this_run_minutes} minutes of data in this session.')
            print(f'Participant left with {np.round(percentage,2)}% of total scan in this Run.')

        # Shorten data to the requested total minutes if necessary
        total_plusminus_3TR = total_minutes - (3*TR/60)
        if this_run_minutes >= total_minutes:
            print(f'\nShortening this session to {total_minutes} minutes of data...')
            max_frames = int(np.round(total_minutes*60/TR, 0)) # maximum frames for minimum set minutes at TR.
            one_indices = np.flatnonzero(combined)
            if len(one_indices) >= max_frames:
                # Get the index of the cut-off frame
                cut_off_index = one_indices[max_frames-1]
                # Add cut-off to mask used for scrubbing (the frames after cut-off are now set to 0)
                combined[cut_off_index + 1:] = 0
            else:
                raise ValueError('Error in calculating minutes of data remaining after motion and outlier removal.')
        elif this_run_minutes >= total_plusminus_3TR:
            print(f'\nParticipant has the requested minutes: {total_minutes} min +-3TR. Keeping session as is...\n')
            pass
        elif this_run_minutes < total_minutes: # note that MSC only has one run per session.
            print(f'\n\nParticipant left with less than minimum requested minutes: {total_minutes} min. SKIPPING SESSION...\n\n')
            continue
        else:
            raise ValueError('Error in calculating minutes of data remaining after motion and outlier removal.')

        # Recalculate minutes of data remaining
        minutes_left = sum(combined)*TR/60
        print(f'Participant now left with {minutes_left} minutes of data in this session.')

        print('\nSaving shortened motion file...\n')
        np.savetxt(combined_file, combined, fmt="%d")
        print(f'{combined_file} saved.')

        motion_file = combined_file # make sure to use the combined file if outliers were removed for the next steps


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
        pconn = outfile / 'func' / f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(cifti_in), str(pconn), str(motion_file)]
        output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
        #print(f'{correlation_args}')
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

print('\n\nFINISHED!!')
