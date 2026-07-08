
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
from functions.utils import filter_output, build_subject_session_run_map, RunInfo, sample_continuous_chunk
from functions.dconn_shrinker import dconn_to_hdf5

def sample_data(minutes, sub_id, ses_id, directory, dataset, feature):

    ######## OPTIONS ########
    # S3 Bucket location
    #datasetdir = 's3://subpop/derivatives/xcpd/output'
    datasetdir = 's3://subpop-v1.0/derivatives/xcpd0.12.0/output'

    # scrach location to save everything to
    wd = Path('/scratch.global/mgell/FC_stability')

    # total minutes of data to keep per session
    #total_minutes = 25
    total_minutes = minutes
    feature = 'Gordon'
    sub_i = sub_id
    #new_ses = 'ses-12'
    new_ses = ses_id

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
    metric = f'seg-{feature}_den-91k_stat-mean_timeseries'
    ext_in = 'ptseries.nii'
    ext_out = 'pconn.nii'
    ######## END OF OPTIONS ########




    # set up for naming purposes
    fd_str = str(fd_threshold).replace('.', '')
    smooth_str = f'_smoothed_{smoothing_kernel}mm' if smooth else ''

    # Prep folder locations
    # wd = os.getcwd()
    # wd = Path(os.path.dirname(wd))
    outdir = wd / 'data'
    indir = directory / 'data' / dataset / f'sub-{sub_i}' / new_ses
    outdir.mkdir(parents=True, exist_ok=True)

    # For getting surfaces if smoothing, currently hard coded
    s3_loc =  f'{datasetdir}/{sub_i}/sub-{sub_i}/ses-combined'

    # find the concatenated p/dtseries file
    file_in = glob.glob(f'{indir}/*{feature}*.{ext_in}')[0]
    
    # Get TR
    output = subprocess.run(['./cifti_get_TR.sh', str(file_in)], capture_output=True, text=True, check=True)
    if output.stderr.strip():
        raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
    output.stdout.strip()
    #print(f"TR = {filter_output(output.stdout.strip())}")

    # TR
    TR_ref = float(filter_output(output.stdout.strip()))

    # Load motion file
    session_motion = glob.glob(f'{indir}/*.txt')[0]
    # session_motion = (indir / f"sub-{sub_i}_{new_ses}_task-{task}_concatenated_desc-FD_{fd_str}_and_outliers_combined_trimmed.txt")
    session_mask = np.loadtxt(session_motion, dtype=int)

    # Report the final kept minutes (sum across runs after trimming)
    mask_total = (session_mask.sum() * TR_ref) / 60.0 # should be same as below total_usuable_minutes
    
    # Sample a continuous subset of data from the concatenated d/ptseries
    sampled_mask, _, total_usable_minutes, grace_minutes = sample_continuous_chunk(session_mask, TR_ref, minutes_to_sample=total_minutes, leaway_TR=2.0, offset=0, seed=None)

    kept_minutes = (sampled_mask.sum() * TR_ref) / 60.0
    print(f"\nSampled {kept_minutes:.2f} minutes from motion mask with {mask_total:.2f} minutes total.\n")

    # Save the sampled mask for wb_command
    sampled_mask_file = f'{outdir}/sub-{sub_i}_{new_ses}_sampled_mask.txt'
    np.savetxt(sampled_mask_file, sampled_mask, fmt='%d')


    # Smooth if necessary
    if smooth:
        # First need to get the midthickness surfaces
        s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
        surf_R = indir / 'anat' / f'sub-{sub_i}_{new_ses}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'

        if not os.path.isfile(surf_R):
            output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)
            if not os.path.isfile(surf_R):
                s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_space-fsLR_den-32k_hemi-R_desc-hcp_midthickness.surf.gii'
                output = subprocess.run(['./get_data.sh',str(s3_file),str(surf_R)], capture_output=True, text=True, check=True)

            if output.stderr.strip():
                raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
            print(f"{output.stdout.strip()}")

        s3_file = f'{s3_loc}/anat/sub-{sub_i}_{ses_i}_run-01_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'
        surf_L = indir / 'anat' /f'sub-{sub_i}_{new_ses}_space-fsLR_den-32k_hemi-L_desc-hcp_midthickness.surf.gii'

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
        smooth_out = f'{indir}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_smoothed_{smoothing_kernel}.{ext_in}'
        smooth_args = ['./cifti_smooth.sh', str(file_in), str(smooth_out), str(smoothing_kernel), str(surf_L), str(surf_R)]
        output = subprocess.run(smooth_args, capture_output=True, text=True, check=True)
        if output.stderr.strip():
            raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
        print(f"{filter_output(output.stdout.strip())}")

        file_in = smooth_out


    # create p/dconn
    #print('\nCreating p/dconn...')
    conn_file_out = f'{outdir}/sub-{sub_i}_{new_ses}_task-{task}_space-fsLR_{metric}_FD_{fd_str}{smooth_str}.{ext_out}'
    correlation_args = ['./cifti_correlation.sh', str(file_in), str(conn_file_out), str(sampled_mask_file)]
    output = subprocess.run(correlation_args, capture_output=True, text=True, check=True)
    #print(f'{correlation_args}')
    if output.stderr.strip():
        raise RuntimeError(f"Error from wb cmd:\n{output.stderr.strip()}")
    #print(f"{filter_output(output.stdout.strip())}")

    return conn_file_out
