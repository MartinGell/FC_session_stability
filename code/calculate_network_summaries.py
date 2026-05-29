
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nb
import glob

from nilearn import connectome
from pathlib import Path


######
# ADD CHECK IF NET LABELS ARE SAME LENGTH AS MATRIX DIMENSIONS!!!!!!!!!!!!!!!
######
def summarize_connectivity(matrix, network_labels, network_names=None):
    """
    Summarizes within-network and between-network connectivity.

    Parameters:
        matrix (numpy.ndarray): Symmetric functional connectivity matrix.
        network_labels (numpy.ndarray): Array of integers indicating network membership for each row/column.
        network_names (numpy.ndarray, optional): Array of network name strings parallel to network_labels.

    Returns:
        dict with keys:
            summary_matrix (numpy.ndarray): Matrix of mean connectivity per network pair.
            long_format (pd.DataFrame): Long-format DataFrame with columns:
                network: network label (within) or "Net_i-Net_j" label (between)
                network_name: ROI network name string (if network_names provided)
                value: individual edge connectivity value
                type: "within" or "between"
    """
    unique_labels = np.unique(network_labels)
    n_networks = len(unique_labels)
    summary_matrix = np.zeros((n_networks, n_networks))
    rows = []

    label_to_name = {}
    if network_names is not None:
        for lbl, name in zip(network_labels, network_names):
            label_to_name[lbl] = name

    for i, label_i in enumerate(unique_labels):
        indices_i = np.where(network_labels == label_i)[0]
        network_i = matrix[np.ix_(indices_i, indices_i)]
        # Extract upper triangle excluding diagonal for within-network
        upper_triangle_i = network_i[np.triu_indices_from(network_i, k=1)]
        summary_matrix[i, i] = np.mean(upper_triangle_i)
        name_i = label_to_name.get(label_i)
        for val in upper_triangle_i:
            rows.append({"network": label_i, "network_name": name_i, "value": val, "type": "within"})

        for j, label_j in enumerate(unique_labels):
            if i < j:  # Avoid duplicates since the matrix is symmetric
                indices_j = np.where(network_labels == label_j)[0]
                network_ij = matrix[np.ix_(indices_i, indices_j)]
                mean_ij = np.mean(network_ij)
                summary_matrix[i, j] = mean_ij
                summary_matrix[j, i] = mean_ij  # Symmetric assignment
                name_j = label_to_name.get(label_j)
                for val in network_ij.flatten():
                    rows.append({"network": f"{label_i}-{label_j}", "network_name": f"{name_i}-{name_j}", "value": val, "type": "between"})

    return {
        "summary_matrix": summary_matrix,
        "data_long": pd.DataFrame(rows)
    }




feature = 'Gordon' #4S1056Parcels
dataset = 'MSC25' # 'subpop', 'MSC', HCPtrt_cneuro, MSC_4runs, subpop_UMN25, subpop_WashU25, adultcontrols25
####################################### 

# Go into this in ipython and test with different dataset. Then finish MSC25 and adultcontrols25

# Prep folder locations
wd = os.getcwd()
wd = Path(os.path.dirname(wd))
datadir = wd / 'data' / dataset
outdir = wd / 'res' / dataset
outdir.mkdir(parents=True, exist_ok=True)

# load data
icc_var_data = outdir / f'{dataset}_icc_variances_{feature}.csv'
df = pd.read_csv(icc_var_data)

# atlas
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


# Create a mask of nans across all subjects and sessions
file_paths = glob.glob(f"{wd}/data/{feature}_nan_rows/*.txt")
all_nan_rows = []
for file_i in sorted(file_paths):
    nan_rows = np.loadtxt(file_i, dtype=bool)
    all_nan_rows.append(nan_rows)

all_nan_rows = pd.DataFrame(all_nan_rows)
template_nan_rows = np.any(all_nan_rows, axis=0)


# Get example file for data dims
example_sub = glob.glob(f'{datadir}/*')[0]
file = glob.glob(f"{example_sub}/ses-*/*{feature}*.pconn.nii")[0]
nii = nb.load(file)
dat = nii.get_fdata()

#insort = atlas['7_network_index'].values.argsort()

#all_metrics = ['icc', 'between_sub_var', 'within_sub_var', 'var_within_person_session', 'var_within_person_run']
all_metrics = ['icc']

# reorder the network labels to match matrix sorting and remove nodes with nans
idx = indsort.flatten()
net_order_sorted = atlas['net_order'].values[idx][~template_nan_rows]
roi_names_sorted = atlas['ROI'].values[idx][~template_nan_rows]

for metric in all_metrics:
    upper = df[metric]
    mat = connectome.vec_to_sym_matrix(upper,diagonal=np.repeat(np.nan,len(dat)))
    np.fill_diagonal(mat, 1)

    sorted_mat = mat[indsort, indsort.T]
    sorted_mat = sorted_mat[~template_nan_rows, :][:, ~template_nan_rows]

    summary_stats = summarize_connectivity(sorted_mat, network_labels=net_order_sorted, network_names=roi_names_sorted)

    # Display the summary matrix
    summary_matrix = summary_stats["summary_matrix"]
    print("Summary Connectivity Matrix:")
    print(summary_matrix)

    cmap_custom = plt.cm.YlGnBu

    plt.figure(figsize=(7, 7))
    plt.imshow(summary_matrix, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
    cbar = plt.colorbar(fraction=0.046)
    plt.show()

    file2save = f"{outdir}/plots/{metric}_NETWORK_{feature}.png"
    print(f'saving: {file2save}')
    plt.savefig(f'{file2save}', dpi=300)
    plt.close()

    # now save the long format dataframe
    data_out = summary_stats["data_long"]
    data_out.to_csv(f"{outdir}/{metric}_NETWORK_{feature}.csv", index=False)  


