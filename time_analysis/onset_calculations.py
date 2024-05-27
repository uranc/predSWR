import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from scipy import signal
import importlib
import sys
from scipy.signal import butter
from scipy.signal import sosfiltfilt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model.cnn_ripple_utils import get_performance, intersection_over_union
from model.input_fn import rippleAI_load_dataset

fs = 1250
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

# # Plot the segment
# time_axis = np.arange(-window_size, window_size)

# plt.figure()
# plt.plot(time_axis, average_probabilities, color = 'blue', label = 'onset')
# plt.plot(time_axis, average_probabilities_mid, color = 'k', label = 'midpoint')
# plt.axvline(x=0, color='r', linestyle='--')
# plt.xlabel('Time')
# plt.ylabel('Probability')
# plt.legend()
# plt.title('Onset-centered probability')
# directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images_avg/'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# plt.savefig(directory + '{0}.png'.format(model))
# plt.close()

# # Computing the cross-correlation between the average probability and the onset-centered probability
# correlation = signal.correlate(average_probabilities, average_probabilities_mid, mode='full')
# plt.figure()
# plt.plot(correlation)
# plt.xlabel('Lag')
# plt.ylabel('Correlation')
# plt.title('Cross-correlation between onset-centered and midpoint-centered probability')
# directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images_crosscorrel/'
# if not os.path.exists(directory):
#     os.makedirs(directory)
# plt.savefig(directory + '{0}.png'.format(model))
# plt.close()



# F1 rippleAI-style 

# The input should be Nx2 matrix with start and end of pred/true events [seconds]
onset_indices_true = np.where(np.diff(y_true) == 1)[0] / fs
offset_indices_true = np.where(np.diff(y_true) == -1)[0] / fs
true_events = np.vstack((onset_indices_true, offset_indices_true)).T

# Need to choose a threshold to binarize the probabilities
threshold = 0.5
y_probs_aux = (y_probs >= threshold).astype(int)
onset_indice_pred = np.where(np.diff(y_probs_aux) == 1)[0] / fs
offset_indices_pred = np.where(np.diff(y_probs_aux) == -1)[0] / fs
pred_events = np.vstack((onset_indice_pred, offset_indices_pred)).T

[precision, recall, F1, TP, FN, IOU] = get_performance(pred_events, true_events)
# Precision -> avg of TP values (1 if all predictions are correct)
# Recall -> 1 - avg of FN values (1 if all GT is present in the predictions)

iou_threshold = 0.2 # 0 by default in rippleAI
# Determine TP and FP
tp = np.sum(IOU >= iou_threshold, axis=1) > 0  # True if any IoU >= threshold for a given predicted interval
fp = ~tp  # False positive if no IoU >= threshold for a given predicted interval
fn = np.sum(IOU >= iou_threshold, axis=0) == 0  # False negative if no predicted interval overlaps significantly with a ground truth interval

# Print results
print("True Positives (TP):", np.sum(tp))
print("False Positives (FP):", np.sum(fp))


false_positive_intervals = pred_events[fp]
true_positive_intervals = pred_events[tp]

# Loading the LFP so we can plot it
val_datasets, val_labels = rippleAI_load_dataset(params=None, mode='test')
LFP = val_datasets[0][:,4]
# avg_fp_aux = []
# for interval in false_positive_intervals:
#     # Compute the LFP average during the FP intervals 
#     start = int(interval[0] * fs)
#     end = int(interval[1] * fs)
#     LFP_interval = LFP[start:end]
#     avg_fp_aux.append(LFP_interval)
# avg_tp_aux = []
# for interval in true_positive_intervals:
#     # Compute the LFP average during the TP intervals 
#     start = int(interval[0] * fs)
#     end = int(interval[1] * fs)
#     LFP_interval = LFP[start:end]
#     avg_tp_aux.append(LFP_interval)

# Need to align in the center of the ripple: first try the average and if it
# doesnt work bandpass the LFP and np.max for finding the center. 
# For the difference in length of the ripples get extra LFP to fix that out 

    

# # Compute the average
# TP_avg = np.mean(np.array(avg_tp_aux), axis=0)
# FP_avg = np.mean(np.array(avg_fp_aux), axis=0)


# we need to do this ripple-wise 

for interval in false_positive_intervals:
    pdb.set_trace()
    signal = LFP[np.int32(interval[0]*fs):np.int32(interval[1]*fs)]
    
    filter = butter( 5, [150, 250], btype='bandpass', output='sos', fs=fs)
    filtered_signal = sosfiltfilt(filter, signal)
    lfp_abs = np.abs(filtered_signal)

    filter_envelope = butter(5, [ 1, 50], btype='bandpass', output='sos', fs=fs)
    filtered_signal = sosfiltfilt(filter_envelope, lfp_abs)

    # we take the max of the filtered signal to find the center of the ripple
    center = np.argmax(filtered_signal)
    window_size = 100
    start = center - window_size
    end = center + window_size
    ripple = LFP[start:end]


pdb.set_trace()