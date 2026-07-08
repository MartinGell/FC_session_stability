
# USE Conda env: FC_stability

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import nibabel as nb
import matplotlib.pyplot as plt
import contextlib

from pathlib import Path
from nilearn import connectome
from itertools import combinations

# custom functions
from calc_conn_subpop_set_minutes_spatial_sim import sample_data


# Which feature and dataset (folder) to use
feature = 'Gordon' # '4S1056Parcels' 'Glasser'
dataset = 'subpop_UMN50' # 'subpop', 'MSC', HCPtrt_cneuro, MSC_4runs subpop_UMN25 adultcontrols25

# amount of data to sample from half one
minutes_to_sample = [5, 10, 15, 20, 25]

# number of repetitions for random sampling (for error bars)
n_reps = 100
####################################### 



# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
indir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

# Get all subjects data locations
subs = glob.glob(f'{indir}/*')
file_paths = glob.glob(f"{indir}/sub-*/ses-34/*{feature}*.pconn.nii")

# initialize
data = []

# Loop over all subjects
for file_i in sorted(file_paths):
    # extract subject and session IDs from the file path
    parts = file_i.split("/")
    sub_id = parts[-3].split("-")[1]
    ses_id = parts[-2]
    
    print(f'Loading: {sub_id}, {ses_id}')

    # load data and get upper triangle
    nii = nb.load(file_i)
    dat = nii.get_fdata()

    #dat = dat.astype(np.float16)
    dat = np.clip(dat, -0.999999, 0.999999) # shouldnt be necessary, avoids nans
    dat_z = np.arctanh(dat)  # Fisher z-transform

    # upper triangle for this subjects half 2, i.e. held-out session (here 'ses-34', concat of all runs in ses-3 nad ses-4)
    upper_held_out = connectome.sym_matrix_to_vec(dat_z, discard_diagonal = True)
    
    # Sample
    for minute_i in minutes_to_sample:
        print(f'\nStarting sampling procedure for {minute_i} minutes of data...')

        # Save output of sampling function to log file
        log_dir = wd / 'code' / 'logs' / 'sampling'
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f'sample_data_{sub_id}_{minute_i}_output.log', 'a') as log_f:
            with contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
                # Actually sample now
                for rep_i in range(n_reps):
                    # function for actually sampling and creating FC
                    sampled_conn_file = sample_data(minute_i, sub_id, ses_id='ses-12', directory=wd, dataset=dataset, feature=feature)

                    # load
                    nii_sampled = nb.load(sampled_conn_file)
                    dat_sampled = nii_sampled.get_fdata()
                    #dat_sampled = dat_sampled.astype(np.float16)
                    dat_sampled = np.clip(dat_sampled, -0.999999, 0.999999)
                    dat_sampled_z = np.arctanh(dat_sampled)

                    upper = connectome.sym_matrix_to_vec(dat_sampled_z, discard_diagonal = True)

                    df = pd.DataFrame({
                        'held_out': upper_held_out,
                        'sampled': upper
                    })

                    # save
                    corr = df.corr(method='pearson').iloc[0,1]
                    data.append([sub_id, minute_i, np.round(corr, 3)]) 



cols = ['subject_id', 'minutes_sampled', 'spatial_similarity']
df = pd.DataFrame(data, columns=cols)

# Save the results
out_file = outdir / f'{dataset}_{feature}_spatial_similarity_over_time.csv'
df.to_csv(out_file, index=False)
print(f'\nSaved results to {out_file}\n')

print("FINISHED!")
