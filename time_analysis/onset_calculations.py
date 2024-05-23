import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from scipy.signal import correlate



dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/'
y_true = np.load(dir + 'val_labels_0.npy')
model = 'prob_rippAI_CNN1D_0.npy'
y_probs = np.load(dir + model)

window_size = 100  

onset_indices = np.where(np.diff(y_true) == 1)[0] 
offset_indices = np.where(np.diff(y_true) == -1)[0]

aligned_segments = []
aligned_segments_mid = []

for onset,offset in zip(onset_indices, offset_indices):
    # Centered in onset
    start = onset - window_size 
    end = onset + window_size
    segment = y_probs[start:end]
    aligned_segments.append(segment)

    # Centered in midpoint
    midpoint = (onset + offset) // 2
    start_index = midpoint - window_size
    end_index = midpoint + window_size 
    segment_mid = y_probs[start_index:end_index]
    aligned_segments_mid.append(segment_mid)



# Stack aligned segments into an array
aligned_segments_array = np.vstack(aligned_segments)
aligned_segments_mid_array = np.vstack(aligned_segments_mid)

# Compute the mean across aligned segments for each time point
average_probabilities = np.mean(aligned_segments_array, axis=0)
average_probabilities_mid = np.mean(aligned_segments_mid_array, axis=0)

# Plot the segment
time_axis = np.arange(-window_size, window_size)

plt.figure()
plt.plot(time_axis, average_probabilities, color = 'blue', label = 'onset')
plt.plot(time_axis, average_probabilities_mid, color = 'k', label = 'midpoint')
plt.axvline(x=0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Probability')
plt.legend()
plt.title('Onset-centered probability')
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + '{0}.png'.format(model))
plt.close()
