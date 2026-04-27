
import glob
import os
import subprocess
import h5py
import nibabel as nb
import numpy as np
import pandas as pd
import shutil

from pathlib import Path

from functions.handling_outliers import isthisanoutlier
from functions.utils import filter_output, build_subject_session_run_map, RunInfo, allocate_minutes_with_grace, trim_mask_to_minutes



######## OPTIONS ########
# S3 Bucket location
datasetdir = 's3://btc-hcp-ya/processed/xcpd_v0.10.1'

# name the directory to save data to
dataset = 'HCP_YA25' #'HCPtrt_cneuro' #'subpop' 

# total minutes of data to keep per session
total_minutes = 25

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
######## END OF OPTIONS ########


# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
out = wd / 'data'

### Subjects, sessions and runs ###
#sub = ['sub-753150']
sublist = wd / 'code' / 'sublist' / 'Unrelated_S900_Subject_multilist1_with_physio.csv'
sub = pd.read_csv(sublist)['Subject'].tolist()
ses = ['ses-3T']  # in order to find the file it will have to be: 'combined' but would be better to rename this to ses-1 ...
run = ['run-1','run-2','run-3','run-4']

# set up for naming purposes
fd_str = str(fd_threshold).replace('.', '')
smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''


# Get data and create d/pconn
for sub_i in sub:
    print(f'\n\n')
    print('=================')
    print(f'Subject: {sub_i}')
    print('=================')
    outdir = out / dataset / f'sub-{sub_i}'
    # Create the directory if it doesn't exist
    outdir.mkdir(parents=True, exist_ok=True)
    
    for ses_i in ses:
        print(f'\nSession: {ses_i}')
        print(f'\nNumber of runs to concatenate: {len(run)}')

        outfile = outdir / ses_i
        # Create the directory if it doesn't exist
        outfile.mkdir(parents=True, exist_ok=True)

        # Check how many runs there are for this session and calculate the number of minutes that will be kept per run
        n_runs = len(run)
        print(f'\nGoing to keep {total_minutes} minutes across all runs')
        print(f'\nFound {n_runs} runs for this session')

        # Now get run data and motion files prepare for shortening and concatenation
        runs_info = []
        print('\nGetting data...')
        for run_i in run:
            print(f'File: {run_i}')

            s3_loc =  f'{datasetdir}/sub-{sub_i}/sub-{sub_i}/{ses_i}'

            # BOLD
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'
            cifti_out = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}.{ext_in}'

            output = subprocess.run(['./get_data.sh',str(s3_file),str(cifti_out)], capture_output=True, text=True, check=True)
            if output.stderr.strip():
                if output.stderr.__contains__('does not exist'):
                    continue
                else:
                    raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")        

            # Motion
            s3_file = f'{s3_loc}/func/sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'
            motion_file_run = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-abcc_qc.hdf5'

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
                print(f"TR: {TR}") # should be 0.72s for HCP

            inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
            motion_file_run = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-FD_{fd_str}.txt'
            print('\nSaving extracted motion file...')
            print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}') # Here counting non-inverted motion
            np.savetxt(motion_file_run, inverted_motion, fmt="%d")

            # optionally identify outliers. Removal happens when creating the d/pconn
            if remove_outliers:
                # First run wb_command and load the std.txt file as its faster
                print('\nIdentifying outliers using the median approach...')
                std_txt = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}_std.txt'
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
                    std_txt = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}_std_np.txt'
                    np.savetxt(std_txt, std, fmt='%.5f')

                print('\nFlagging outliers...')
                [outlier,_,_,_] = isthisanoutlier(std)
                # turn outlier into a binary mask (0 = remove, 1 = keep)
                outlier = outlier.astype(int)
                inverted_outlier = 1-outlier
                outlier_file = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_space-fsLR_{metric}_std_outlier.txt'
                np.savetxt(outlier_file, inverted_outlier, fmt='%d')
                n_extra_outliers = np.sum(outlier).astype(int)
                print(f'Flagged {n_extra_outliers} additional frames as outliers.')

                # combine motion and outlier files
                print('\nCombining motion and outlier files and saving...')
                combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
                combined_file = outfile / 'func' / f'sub-{sub_i}_{ses_i}_task-{task}_{run_i}_desc-FD_{fd_str}_and_outliers_combined.txt'

                this_run_minutes = sum(combined)*TR/60
                print(f'\nParticipant has {this_run_minutes} minutes of data in this run.')

                runs_info.append(RunInfo(run_name=run_i,
                             keep_mask=combined,
                             usable_minutes=this_run_minutes,
                             TR=TR))

        # Now set up everything for concatenation and trim run data to the requested mintues across whole session
        print('\n______________________DONE_WITH_INDIVIDUAL_RUNS______________________')
        #print('\nConcatenating motion files...')
        print(f'\nShortening this session to {total_minutes} minutes of data...')

        # Check if runs_info is empty (i.e., no runs found for this session) and if so skip to next session
        if not runs_info:
            print(f"\nNo data for {sub_i}. SKIPPING SUBJECT!!\n\n")
            # remove the subject folder that was created since there is no data for this session
            if os.path.exists(outdir):
                shutil.rmtree(outdir)
                print(f"{outdir} --> deleted.\n\n")
            continue

        # Check if TR differs across runs
        TR_ref = runs_info[0].TR
        if any(abs(r_i.TR - TR_ref) > 1e-6 for r_i in runs_info):
            print("Warning: TR differs across runs; using first run's TR for grace calculation.")
            raise RuntimeError("TR differs across runs; cannot proceed.")

        # Find optimal minute allocation across runs given differences in usable minutes per run
        assigned, target, total_usable, grace = allocate_minutes_with_grace(
            [r_i.usable_minutes for r_i in runs_info], total_minutes, TR_ref, leaway_TR=7.0
        ) # +-7 TR = approx. 12 seconds

        if assigned is None:
            print(f"\nThere are only {total_usable:.2f} usable minutes across session")
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
        session_motion = (outfile / f"sub-{sub_i}_{ses_i}_task-{task}_concatenated_desc-FD_{fd_str}_and_outliers_combined_trimmed.txt")
        np.savetxt(session_motion, session_mask.astype(int), fmt="%d")

        # Report the final kept minutes (sum across runs after trimming)
        kept_minutes_total = (session_mask.sum() * TR_ref) / 60.0
        print(f"\nSaved final session mask. Minutes kept: {kept_minutes_total:.2f}.\n")


        # concat all of ses-x time series
        print('\nConcatenating d/pseries...')
        runs = glob.glob(f'{outfile}/func/*{metric}.{ext_in}')
        file_in = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}.{ext_in}'
        merge_args = ['./cifti_merge.sh', str(file_in)]
        merge_args.extend([str(run) for run in runs if run is not None])  # Only include non-None runs
        output = subprocess.run(merge_args, capture_output=True, text=True, check=True)
        # should look like this: ./concat_cifti_merge.sh cifti_out run1 run2 ...
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        # # concat all of ses-x time series
        # print('\nConcatenating d/pseries...')
        # runs = glob.glob(f'{outfile}/func/*{metric}.{ext_in}')
        # cifti_out = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}.{ext_in}'
        # merge_args = ['./cifti_merge.sh', str(cifti_out)]
        # merge_args.extend([str(run) for run in runs if run is not None])  # Only include non-None runs
        # output = subprocess.run(merge_args, capture_output=True, text=True, check=True)
        # # should look like this: ./concat_cifti_merge.sh cifti_out run1 run2 ...
        # if output.stderr.strip():
        #     if output.stderr.__contains__('no inputs specified'):
        #         print('\nNo data for subject found.')
        #         print('\n\nSKIPPING!!!!\n\n')
        #         continue
        #     else:
        #         raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        # print(f"{output.stdout.strip()}")        

        # # concat all motion files
        # motion_files = glob.glob(f'{outfile}/func/*.hdf5')
        # motion_list = []
        # for m_file_i in motion_files:
        #     with h5py.File(m_file_i, 'r') as f:
        #         # Extract the binary mask indicating frame removal based on framewise displacement (FD) threshold.
        #         # Frames with FD > threshold are marked as 1 (removed), and frames with FD <= fd_threshold are marked as 0 (kept).
        #         m_i = f['dcan_motion'][f'fd_{fd_threshold}']['binary_mask'][()].astype(int)
        #         TR = f['dcan_motion']['fd_0.2']['remaining_seconds'][()]/f['dcan_motion']['fd_0.2']['remaining_total_frame_count'][()]
        #         print(f"TR: {TR}")
        #     motion_list.append(m_i)

        # motion = np.concatenate(motion_list)
        # inverted_motion = 1-motion     # NEED TO INVERT FOR wb_command as it expects 0 = remove, 1 = keep
        # motion_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}.txt'
        # print('\nSaving concatenated motion file...')
        # print(f'Will remove {np.sum(motion).astype(int)} frames at FD > {fd_threshold}')
        # np.savetxt(motion_file, inverted_motion, fmt="%d")

        # # optionally identify outliers. Removal happens when creating the d/pconn
        # if remove_outliers:
        #     # First run wb_command and load the std.txt file as its faster
        #     print('\nIdentifying outliers using the median approach...')
        #     std_txt = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std.txt'
        #     stats_args = ['./cifti_std.sh', str(cifti_out), str(std_txt)]
        #     output = subprocess.run(stats_args, capture_output=True, text=True)
        #     #print(f'{stats_args}')
        #     if output.stderr.strip():
        #         raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        #     print(f"{filter_output(output.stdout.strip())}")

        #     # Next check if there are nans in the data and if so use numpy instead of wb_command
        #     std = np.loadtxt(std_txt)
        #     if np.isnan(std).any():
        #         print('Nan values in wb cmd file, using numpy instead...')
        #         X = nb.load(f'{cifti_out}')
        #         concat_cifti = X.get_fdata()
        #         stdevnp = np.nanstd(concat_cifti,axis=1).round(5)
        #         std = stdevnp
        #         # save
        #         std_txt = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_np.txt'
        #         np.savetxt(std_txt, std, fmt='%.5f')

        #     print('\nFlagging outliers...')
        #     [outlier,_,_,_] = isthisanoutlier(std)
        #     # turn outlier into a binary mask (0 = remove, 1 = keep)
        #     outlier = outlier.astype(int)
        #     inverted_outlier = 1-outlier
        #     outlier_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_std_outlier.txt'
        #     np.savetxt(outlier_file, inverted_outlier, fmt='%d')
        #     print(f'Flagged {np.sum(outlier).astype(int)} additional frames as outliers.')

        #     # combine motion and outlier files
        #     print('\nCombining motion and outlier files and saving...')
        #     combined = np.logical_and(inverted_motion, inverted_outlier).astype(int)
        #     combined_file = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_desc-FD_{fd_str}_and_outliers_combined.txt'
        #     np.savetxt(combined_file, combined, fmt="%d")
        #     motion_file = combined_file
        #     minutes = sum(combined)*TR/60
        #     print(f'Participant left with {minutes} minutes of data in this session.')

        # # Optionally collect subjects with too few minutes left after filtering
        # if remove_subjects:
        #     if minutes < min_minutes:
        #         print(f"Participant {sub_i} left with {minutes} minutes of data in this session. Will remove!!")
        #         subs_to_remove.append(sub_i)


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

            file_in = smooth_out


        # create p/dconn
        print('\nCreating p/dconn...')
        conn_file_out = f'{outfile}/sub-{sub_i}_{ses_i}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
        correlation_args = ['./cifti_correlation.sh', str(file_in), str(conn_file_out), str(session_motion)]
        output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
        #print(f'{correlation_args}')
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        # Remove the full func folder
        sub_func_folder = outfile / 'func' 
        print('\nRemoving individual run data...')
        if os.path.exists(sub_func_folder):
            shutil.rmtree(sub_func_folder)
            print(f"{sub_func_folder} --> deleted.")

        # # remove individual runs and keep only concatenated session:
        # print('\nRemoving individual run data...')
        # for run_j in runs:
        #     if os.path.exists(run_j):
        #         os.remove(run_j)
        #     print(f"{run_j} --> deleted.")

        # print('\nRemoving individual motion data...')
        # for motion_j in motion_files:
        #     if os.path.exists(motion_j):
        #         os.remove(motion_j)
        #     print(f"{motion_j} --> deleted.")


# Now remove folders with subjects with too few minutes left after filtering
# if remove_subjects:
#     print('Removing subjects with too few minutes left after filtering...')
#     print(f"Will remove {len(subs_to_remove)} subjects.")
#     for sub_i in subs_to_remove:
#         print(f"Removing subject {sub_i} from dataset.")
#         subdir = out / dataset / f'sub-{sub_i}'
#         if os.path.exists(subdir):
#             shutil.rmtree(subdir)
#         print(f"{subdir} --> deleted.")

