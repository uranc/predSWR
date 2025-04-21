
# this can be with input waveforms predefined
import pdb
import pickle
import numpy as np
import approx_umap
import time 
import h5py
import scipy.io

X = np.load('/cs/projects/OWVinckSWR/Dataset/TopologicalData/ripple_data.pkl', allow_pickle=True)['ripples']
with h5py.File('/mnt/hpc/projects/OWVinckSWR/Dataset/TopologicalData/JuanPabloDB_struct.mat', 'r') as file:
    XX = file['ripples'][:]  # Replace 'your_variable_name' with the actual variable name in the MATLAB file

XX = np.transpose(XX)
X = np.concatenate((X, XX), axis=0)

# pdb.set_trace()
# model = approx_umap.ApproxUMAP(n_neighbors=15, n_components=2)


# t = time.time()
# emb_exact = approx_umap.ApproxAlignedUMAP(fn='exp', k=1).fit_transform(X)  # exact UMAP projections
# elapsed = time.time() - t
# print("Elapsed time for exact UMAP projection: ", elapsed)

t = time.time()
projector = approx_umap.ApproxAlignedUMAP(fn='exp', k=1).fit(X)
elapsed = time.time() - t
print("Elapsed time for approximate UMAP projection: ", elapsed)

all_times = []
for ii in range(100):
    # time.sleep(0.01)
    t = time.time()
    # emb_approx = projector.transform(np.expand_dims(X[ii,:],axis=0))  # approximate UMAP projection
    emb_approx = projector.transform(X[ii:ii+8,:])  # approximate UMAP projection
    elapsed = time.time() - t
    all_times.append(elapsed)
    # print("Elapsed time for approximate UMAP projection: ", elapsed)

print("Mean elapsed time for approximate UMAP projection: ", np.mean(all_times))
# t = time.time()
# emb_approx = projector.transform(np.expand_dims(X[0,:], axis=0))  # approximate UMAP projection
# elapsed = time.time() - t
# print("Elapsed time for approximate UMAP projection: ", elapsed)

# t = time.time()
# emb_approx_exact = projector.update_transform(np.expand_dims(X[0,:], axis=0))  # exact UMAP projection
# elapsed = time.time() - t
# print("Elapsed time for approximate UMAP projection: ", elapsed)


pdb.set_trace()

# pdb.set_trace()
with open("umap_approx_model.pkl", "wb") as f:
    pickle.dump(projector, f)
