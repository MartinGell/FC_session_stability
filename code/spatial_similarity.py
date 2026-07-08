
# USE Conda env: SNR

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from nilearn import connectome
from itertools import combinations
#from statsmodels.regression.mixed_linear_model import MixedLM


#n_batches = 80 # Number of parallel jobs to run

# Which feature and dataset (folder) to use
feature = 'Gordon' # '4S1056Parcels' 'Glasser'
dataset = 'subpop_UMN25' # 'subpop', 'MSC', HCPtrt_cneuro, MSC_4runs subpop_UMN25 adultcontrols25
####################################### 



# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
indir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

if feature == 'Glasser':
    # Load Glasser sorting index
    indsort = np.loadtxt(f'{wd}/data/cortex_subcortex_community_order.txt',dtype=int) -1
    indsort.shape = (len(indsort),1)
elif feature == 'Gordon':
    atlas = pd.read_csv(f'{wd}/data/Gordon_network_order.csv')
    atlas = atlas.drop(range(333, len(atlas)), axis=0)  # drop the extra rows that are not in the matrix
    indsort = atlas['parcel_order'].values.argsort()
    indsort.shape = (len(indsort),1)
    netsort = atlas['net_order'].values.argsort()
    netsort.shape = (len(netsort),1)
else:
    raise ValueError(f'Unknown feature order: {feature}')

# Get all subjects data locations
subs = glob.glob(f'{indir}/*')
file_paths = glob.glob(f"{indir}/sub-*/ses-*/*{feature}*.pconn.nii")

# file_paths = file_paths[:6] + file_paths[12:] remove subjects below 17 years old for adult controls
# subs = subs[:1] + subs[3:]

# initialise
data = []


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

    # sort and plot FC matrix
    # sorted_mat = dat[indsort,indsort.T]

    # cmap_custom = plt.cm.RdBu_r

    # file2save = outdir / 'plots' / 'FC' / f"{parts[-1].split(".")[0]}.png"
    # file2save.parent.mkdir(parents=True, exist_ok=True)
    # print(f'saving: {file2save}')

    # plt.figure(figsize=(7, 7))
    # plt.imshow(sorted_mat, origin='lower', cmap=cmap_custom, vmin=-1, vmax=1)
    # cbar = plt.colorbar(fraction=0.046)
    # plt.savefig(f'{file2save}', dpi=180)
    # plt.close()

    # save upper triangle
    print(dat.shape)
    upper = connectome.sym_matrix_to_vec(dat, discard_diagonal = True)
    
    # save
    data.append([sub_id, ses_id, *upper])

cols = ['subject_id', 'session_id'] + [f"conn_{i}" for i in range(len(data[0]) - 2)]
df = pd.DataFrame(data, columns=cols)

# Now correalate each connection across subjects and sessions and average for within vs between subject correlation
data_cols = [c for c in df.columns if c not in ['subject_id', 'session_id']]

# Correlation matrix across all rows (each row = one observation/session)
corr_matrix = df[data_cols].T.corr()  # shape: (n_obs x n_obs)

# Add subject labels to index
labels = df[['subject_id', 'session_id']].reset_index(drop=True)
corr_matrix.index = range(len(labels))
corr_matrix.columns = range(len(labels))

# Anonymize here, before anything downstream
sub_map = {sub: f"sub-{i+1:02d}" for i, sub in enumerate(sorted(labels['subject_id'].unique()))}
labels['subject_id'] = labels['subject_id'].map(sub_map)
df['subject_id'] = df['subject_id'].map(sub_map)  # keep df in sync if you need it later

# initiate
records = []
n = len(labels)

# correlate
for sub in labels['subject_id'].unique():
    sub_idx = labels.index[labels['subject_id'] == sub].tolist()
    other_idx = labels.index[labels['subject_id'] != sub].tolist()

    # Within: all pairs among this subject's sessions
    for i, j in combinations(sub_idx, 2):
        records.append({
            'focal_sub': sub,
            'r': corr_matrix.iloc[i, j],
            'pair_type': 'within'
        })

    # Between: this subject's sessions vs. all other subjects' sessions
    for i in sub_idx:
        for j in other_idx:
            records.append({
                'focal_sub': sub,
                'r': corr_matrix.iloc[i, j],
                'pair_type': 'between'
            })

# Save
corr_long = pd.DataFrame(records)
path_to_res = outdir / f'{feature}_sub_corr_long.csv'
#corr_long.to_csv(path_to_res, index=False)


# Now plot the correlation matrix with subjects grouped together and boundaries between them
# Order by subject then session
df_sorted = df.sort_values(['subject_id', 'session_id'])
order = df_sorted.index
corr_ordered = corr_matrix.iloc[order, order]

# Boundary lines
sub_sizes = df.sort_values(['subject_id', 'session_id']).groupby('subject_id').size()
boundaries = sub_sizes.cumsum().values[:-1]

# Centers for x and y labels
sub_sizes_sorted = df_sorted.groupby('subject_id', sort=False).size()
boundaries = sub_sizes_sorted.cumsum().values[:-1]
centers = sub_sizes_sorted.cumsum().values - sub_sizes_sorted.values / 2

# save
path_to_plot = outdir / f'{feature}_sub_corr_matrix.png'

# plot
fig, ax = plt.subplots(figsize=(7.2, 6))
sns.heatmap(
    corr_ordered,
    ax=ax,
    cmap='RdYlBu',
    vmin=0, vmax=1,
    xticklabels=False,
    yticklabels=False,
    cbar_kws={'label': 'r'},
    annot_kws={"size": 20, "fontname": "Arial", "weight": "bold"}
)
for b in boundaries:
    ax.axvline(b, color='black', linewidth=0.5)
    ax.axhline(b, color='black', linewidth=0.5)
    
ax.invert_yaxis()

cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
cbar.ax.set_ylabel('r', fontsize=18)

ax.set_xticks(centers)
ax.set_xticklabels(sub_sizes_sorted.index, rotation=90, fontsize=18, fontname="Arial")
ax.set_yticks(centers)
ax.set_yticklabels(sub_sizes_sorted.index, rotation=0, fontsize=18, fontname="Arial")

plt.tight_layout()
#plt.show()
plt.savefig(path_to_plot, dpi=300)

print(f'saved: {path_to_plot}')