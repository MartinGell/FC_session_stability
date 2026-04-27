
import os
import subprocess
import shutil
import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import glob

from pathlib import Path

from functions.handling_outliers import isthisanoutlier
from functions.utils import filter_output, build_subject_session_run_map, RunInfo, allocate_minutes_with_grace, trim_mask_to_minutes
from functions.dconn_shrinker import dconn_to_hdf5


######## OPTIONS ########
# S3 Bucket location
datasetdir = 's3://prelim-secret-data/xcpd'

# total minutes of data to keep per session
total_minutes = 25

# name the directory to save data to
dataset = f'adultcontrols{total_minutes}'

# Motion filter options
fd_threshold = 0.2

# Outlier removal options
remove_outliers = True

# Smoothing options
smooth = False
smoothing_kernel = 1.7

## File identification options ##
# HELPER: sub-XXXXXX_ses-X_task-{task}_space-fsLR_{metric}.{ext_in}
# task = 'restMENORDICrmnoisevols'
# metric = 'den-91k_desc-denoised_bold'
# ext_in = 'dtseries.nii'
# ext_out = 'dconn.nii'

task = 'restMENORDICrmnoisevols'
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

### Subjects, sessions and runs ###
sublist = wd / 'code/sublist/adultcontrols_subject_session_individual_runs.csv' # best place to find this information is in the json files in derivatives/nordic/ on MSI
sub_ses_run_map = build_subject_session_run_map(sublist)
ses_combined = 'ses-combined'  # in order to find the file it will have to be: 'combined' but would be better to rename this to ses-1 ...


# Get data and create d/pconn
for s_i, ses_dict in sub_ses_run_map.items():
    sub_i = s_i.split('-')
    sub_i = sub_i[1]

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
        # if n_runs == 4:
        #     mintutes_per_run = total_minutes/4
        #     print(f'Grabbing {mintutes_per_run} minutes per run')
        # else:
        #     raise ValueError(f"Unexpected number of runs: {n_runs}. Expected 4.")

        # Now get run data and motion files to prepare for shortening and concatenation
        runs_info = []
        print('\nGetting data...')
        for run_i in runs:
            print(f'\nFile: {run_i}')

            s3_loc =  f'{datasetdir}/sub-{sub_i}_{ses_combined}/sub-{sub_i}/{ses_i}'

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
                print(f"TR: {TR}")

            inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
            motion_file_run = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_{run_i}_desc-FD_{fd_str}.txt'
            print('\nSaving extracted motion file...')
            print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}') # Here counting non-inverted motion
            np.savetxt(motion_file_run, inverted_motion, fmt="%d")

            # optionally identify outliers. Removal happens when creating the d/pconn
            if remove_outliers:
                # First run wb_command and load the std.txt file as its faster
                print('\nIdentifying outliers using the median approach...')
                std_txt = outfile / 'func' / f'sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_std.txt'
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

                this_run_minutes = sum(combined)*TR/60
                print(f'\nParticipant has {this_run_minutes} minutes of data in this run.')

                runs_info.append(RunInfo(run_name=run_i,
                             keep_mask=combined,
                             usable_minutes=this_run_minutes,
                             TR=TR))


        # Now set up everything for concatenation and trim run data to the requested mintues across whole session
        print('\n______________________DONE_WITH_INDIVIDUAL_RUNS______________________')
        print(f'\nShortening this session to {total_minutes} minutes of data...')
        
        # Check if TR differs across runs
        TR_ref = runs_info[0].TR
        if any(abs(r_i.TR - TR_ref) > 1e-6 for r_i in runs_info):
            print("Warning: TR differs across runs; using first run's TR for grace calculation.")
            raise RuntimeError("TR differs across runs; cannot proceed.")

        # Find optimal minute allocation across runs given differences in usable minutes per run
        assigned, target, total_usable, grace = allocate_minutes_with_grace(
            [r_i.usable_minutes for r_i in runs_info], total_minutes, TR_ref
        )

        if assigned is None:
            print(f"\nThere are only {total_usable:.2f} usable minutes across session)")
            print(f"\nThis is below requested threshold. SKIPPING SESSION.\n")
            continue
        # elif target < total_minutes:
        #     print(f"\nAccepting session under target by {total_minutes - target:.2f} min "
        #         f"(≤3TR={grace:.2f} min grace). Target={target:.2f} min.\n")
        
        print("\nPer-run allocation and trimming:")
        session_trimmed_masks = []
        for r_i, minutes_keep in zip(runs_info, assigned):
            print(f"  {r_i.run_name}: usable = {r_i.usable_minutes:.2f}, assigned = {minutes_keep:.2f}")
            # Trim using each run's own TR → minutes->frames mapping stays correct per run
            trimmed_mask = trim_mask_to_minutes(r_i.keep_mask, r_i.TR, minutes_keep)
            session_trimmed_masks.append(trimmed_mask.astype(int))
        
        # Now finally concat all motion files
        # Concatenate trimmed masks in the same order as `runs` and save
        print('\nConcatenating motion files with trim and saving...')
        session_mask = np.concatenate(session_trimmed_masks, axis=0).astype(int)
        session_motion = (outfile / f"sub-{sub_i}_{new_ses}_task-{task}_concatenated_desc-FD_{fd_str}_and_outliers_combined_trimmed.txt")
        np.savetxt(session_motion, session_mask.astype(int), fmt="%d")

        # Report the final kept minutes (sum across runs after trimming)
        kept_minutes_total = (session_mask.sum() * TR_ref) / 60.0
        print(f"\nSaved final session mask. Minutes kept: {kept_minutes_total:.2f}.\n")

        # Next concat all rest runs files
        print('\nConcatenating d/pseries...')
        runs = glob.glob(f'{outfile}/func/*{metric}.{ext_in}')
        file_in = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}.{ext_in}'
        merge_args = ['./cifti_merge.sh', str(file_in)]
        merge_args.extend([str(run) for run in runs if run is not None])  # Only include non-None runs
        output = subprocess.run(merge_args, capture_output=True, text=True, check=True)
        # should look like this: ./concat_cifti_merge.sh file_in run1 run2 ...
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
            smooth_args = ['./cifti_smooth.sh', str(file_in), str(smooth_out), str(smoothing_kernel), str(surf_L), str(surf_R)]
            output = subprocess.run(smooth_args, capture_output=True, text=True, check=True)
            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{filter_output(output.stdout.strip())}")

            file_in = smooth_out

        # create p/dconn
        print('\nCreating p/dconn...')
        conn_file_out = f'{outfile}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(file_in), str(conn_file_out), str(session_motion)]
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
