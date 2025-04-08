import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import nibabel as nb
from nilearn import connectome


feature = '4S1056Parcels' #


indir = '/home/btervocl/shared/projects/martin_SNR/input/subpop'
#indir = '/Users/mgell/Work/SNR/input/subpop'
outdir = '/home/btervocl/shared/projects/martin_SNR/res/subpop'

nii = nb.load(f'{indir}/sub-1007501/ses-2/sub-1007501_ses-2_task-restMENORDICtrimmed_space-fsLR_seg-{feature}_den-91k_stat-mean_timeseries.ptseries.nii')
dat = nii.get_fdata()


def summarize_connectivity(matrix, network_labels):
    """
    Summarizes within-network and between-network connectivity.

    Parameters:
        matrix (numpy.ndarray): Symmetric functional connectivity matrix.
        network_labels (numpy.ndarray): Array of integers indicating network membership for each row/column.

    Returns:
        dict: Summary statistics for within-network and between-network connectivity.
        numpy.ndarray: Matrix of summary statistics matching the network order of the input matrix.
    """
    unique_labels = np.unique(network_labels)
    n_networks = len(unique_labels)
    summary_matrix = np.zeros((n_networks, n_networks))

    for i, label_i in enumerate(unique_labels):
        indices_i = np.where(network_labels == label_i)[0]
        network_i = matrix[np.ix_(indices_i, indices_i)]
        # Extract upper triangle excluding diagonal for within-network
        upper_triangle_i = network_i[np.triu_indices_from(network_i, k=1)]
        summary_matrix[i, i] = np.mean(upper_triangle_i)

        for j, label_j in enumerate(unique_labels):
            if i < j:  # Avoid duplicates since the matrix is symmetric
                indices_j = np.where(network_labels == label_j)[0]
                network_ij = matrix[np.ix_(indices_i, indices_j)]
                mean_ij = np.mean(network_ij)
                summary_matrix[i, j] = mean_ij
                summary_matrix[j, i] = mean_ij  # Symmetric assignment

    return {
        "summary_matrix": summary_matrix
    }


atlas = pd.read_table('/home/btervocl/shared/projects/martin_SNR/text_files/atlas-4S1056Parcels_dseg.tsv')
df = pd.read_csv(f'/home/btervocl/shared/projects/martin_SNR/res/subpop/subpop_icc_variances_{feature}.csv')

#insort = atlas['7_network_index'].values.argsort()

upper = df['icc']
upper = df['between_sub_var']
upper = df['within_sub_var']

mat = connectome.vec_to_sym_matrix(upper,diagonal=np.repeat(np.nan,dat.shape[1]))
np.fill_diagonal(mat, 1)

# Apply the sorting order to both axes (rows and columns)
#sorted = mat[insort, :][:, insort]


summary_stats = summarize_connectivity(mat, atlas['network_label_id'])

# Display the summary matrix
summary_matrix = summary_stats["summary_matrix"]
print("Summary Connectivity Matrix:")
print(summary_matrix)

cmap_custom = plt.cm.YlGnBu

plt.figure(figsize=(7, 7))
plt.imshow(summary_matrix, origin='lower', cmap=cmap_custom, vmin=0, vmax=1)
cbar = plt.colorbar(fraction=0.046)
plt.show()

#file2save = parts[10].split('task-')[1].split('_den')  # just want this: 'restMENORDICtrimmed_space-fsLR_seg-4S956Parcels'    ##### possibly just 4S956Parcels????
file2save = f"{outdir}/plots/WH_NETWORK_{feature}.png"
print(f'saving: {file2save}')
plt.savefig(f'{file2save}', dpi=300)
plt.close()





