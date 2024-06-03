import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
from scipy import signal
import importlib
import sys
from scipy.signal import butter
from scipy.signal import sosfilt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from model.cnn_ripple_utils import get_performance, intersection_over_union
def split_label(label):
    mid = len(label) // 2
    return label[:mid] + '\n' + label[mid:]
def split_label_6(label): 
    mid1 = len(label) // 3
    mid2 = mid1 // 2
    mid3 = mid1 + mid2
    mid4 = mid3 + mid2
    mid5 = mid4 + mid2
    return label[:mid2] + '\n' + label[mid2:mid1] + '\n' + label[mid1:mid3] + '\n' + label[mid3:mid4] + '\n' + label[mid4:mid5]+ '\n' + label[mid5:]


fs = 1250
dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/'
y_true_aux = np.load(dir + 'val_labels_0.npy')
# Loading the LFP so we can plot it
val_datasets = np.load(dir + 'val_dataset_0.npy')
y_true_complete = np.zeros(val_datasets.shape[0])
for lab in y_true_aux:
    y_true_complete[int(lab[0]*1250):int(lab[1]*1250)] = 1


#model = 'prob_rippAI_CNN1D_0.npy'
dir = '/cs/projects/OWVinckSWR/DL/predSWR/probs/'
# model_list = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
#              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense'
#              ]
# model_list = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#                 'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#               'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
#               'Base_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#                'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#                'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense']
model_list = [  #'Average_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense', 
            #'Base_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense',
            #  'Average_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense',
            #  'Base_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense', 
              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense']
# Create a .npy with the validation dataset 
fpr = []
tpr = []
thresholds = []
roc_auc = []
roc_auc_prida = []
tp_prida = []
fp_prida = []
f1 = []
f1_prida = []
recall = []
recall_prida = []
precision = []
precision_prida = []
optimal_thrs = []
for model in model_list: 
    y_probs = np.load(dir + 'preds_val0_' + model + '.npy')
    
    # If they don't have the same length, remove the samples from y_true that are not present in y_probs
    y_true = y_true_complete[:len(y_probs)]
    window_size = 100  

    onset_indices = np.where(np.diff(y_true) == 1)[0] 
    offset_indices = np.where(np.diff(y_true) == -1)[0]
   
    aligned_segments = []
    aligned_segments_mid = []
    aligned_segments_off = []

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

        # Centered in offset 
        start_off = offset - window_size
        end_off = offset + window_size
        segment_off = y_probs[start_off:end_off]
        aligned_segments_off.append(segment_off)


    # Stack aligned segments into an array
    aligned_segments_array = np.vstack(aligned_segments)
    aligned_segments_mid_array = np.vstack(aligned_segments_mid)
    aligned_segments_off_array = np.vstack(aligned_segments_off)

    # Compute the mean across aligned segments for each time point
    average_probabilities = np.mean(aligned_segments_array, axis=0)
    average_probabilities_mid = np.mean(aligned_segments_mid_array, axis=0)
    average_probabilities_off = np.mean(aligned_segments_off_array, axis=0)

    # Plot the segment
    time_axis = np.arange(-window_size, window_size)

    plt.figure()
    plt.plot(time_axis, average_probabilities, color = 'blue', label = 'onset')
    plt.plot(time_axis, average_probabilities_mid, color = 'k', label = 'midpoint')
    plt.plot(time_axis, average_probabilities_off, color = 'gray', label = 'offset')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.title('Onset-centered probability')
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images_avg_prob/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '{0}.png'.format(model))
    plt.close()

    
    # # Computing the cross-correlation between the average probability and the onset-centered probability
    # correlation = signal.correlate(average_probabilities, average_probabilities_mid, mode='full')
    # time_axis = np.arange(-len(correlation)/2, len(correlation)/2)
    # plt.figure()
    # plt.plot(time_axis,correlation)
    # plt.xlabel('Time Stamp')
    # plt.ylabel('Correlation')
    # plt.title('Cross-correlation between onset-centered and midpoint-centered probability')
    # directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images_crosscorrel/'
    # if not os.path.exists(directory):
    #    os.makedirs(directory)
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
    f1_prida.append(F1)
    tp_prida.append(TP)
    fp_prida.append(FN)
    recall_prida.append(recall)
    precision_prida.append(precision)
    

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

    # We want the LFP of the 4th channel where the pyr layer is 
    LFP = val_datasets[:,4]

    # Need to align in the center of the ripple: first try the average and if it
    # doesnt work bandpass the LFP and np.max for finding the center. 
    # For the difference in length of the ripples get extra LFP to fix that out 



    ripple_list_fp = []
    window_size = 500
    filter = butter( 5, [150, 250], btype='bandpass', output='sos', fs=fs)
    filter_envelope = butter(5, [ 1, 50], btype='bandpass', output='sos', fs=fs)

    for interval in false_positive_intervals:
        
        rip = LFP[round(interval[0]*fs):round(interval[1]*fs)]
        filtered_signal = sosfilt(filter, rip)
        lfp_abs = np.abs(filtered_signal)
        filtered_signal = sosfilt(filter_envelope, lfp_abs)

        # we take the max of the filtered signal to find the center of the ripple
        center = np.argmax(filtered_signal) + round(interval[0]*fs)
        start = center - window_size
        if start < 0:   continue
        end = center + window_size
        if end > len(LFP): break
        ripple = sosfilt(filter, LFP[start : end])   
        #ripple =  LFP[start : end] 
        ripple_list_fp.append(ripple)
    
    ripple_list_tp = []
    for interval in true_positive_intervals:
        
        rip = LFP[round(interval[0]*fs):round(interval[1]*fs)]    
        filtered_signal = sosfilt(filter, rip)
        lfp_abs = np.abs(filtered_signal)
        filtered_signal = sosfilt(filter_envelope, lfp_abs)

        # we take the max of the filtered signal to find the center of the ripple
        center = np.argmax(filtered_signal) + round(interval[0]*fs)
        start = center - window_size
        end = center + window_size
        if start < 0:   continue
        if end > len(LFP): 
            # If it is going to be shorter we exclude this ripple 
            break
        ripple = sosfilt(filter, LFP[start : end])   
        #ripple =  LFP[start : end]
        ripple_list_tp.append(ripple)
    #pdb.set_trace() 
    # we compute the average accross the ripples
    TP_avg = np.mean(np.array(ripple_list_tp), axis=0)
    FP_avg = np.mean(np.array(ripple_list_fp), axis=0) 
    # Normalize the values 
    TP_avg_norm = (TP_avg - np.min(TP_avg)) / (np.max(TP_avg) - np.min(TP_avg))
    FP_avg_norm = (FP_avg - np.min(FP_avg)) / (np.max(FP_avg) - np.min(FP_avg))
        
    # and now we plot it and save it 
    time_axis = np.arange(-window_size, window_size)

    plt.figure()
    plt.plot(time_axis, TP_avg, color = 'blue', label = 'True Positive')
    plt.plot(time_axis, FP_avg, color = 'red', label = 'False Positive')
    window_size = 100  
    time_axis = np.arange(-window_size, window_size)
    #plt.plot(time_axis, average_probabilities, color = 'blue', label = 'onset')
    plt.plot(time_axis, average_probabilities_mid * np.max(TP_avg), color = 'k', label = 'prob')
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Timestamps')
    plt.ylabel('LFP')
    plt.legend()
    plt.title('MaxPowerRipple-centered LFP')
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/onset_images_avg_rip/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'prob+BP_{0}.png'.format(model))
    #plt.savefig(directory + '{0}.png'.format(model))
    plt.close()


    corr = signal.correlate(TP_avg, FP_avg, mode='same')
    time_axis = np.arange(-len(corr)/2, len(corr)/2)
    plt.figure()
    plt.plot(time_axis, corr)
    plt.axvline(x=0, color='k', linestyle='--')
    plt.xlabel('Timestamps')
    plt.ylabel('LFP')
    #plt.legend()
    plt.title('Correlation')
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/midpoint_corr_images_avg_rip/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + 'Correlation TP and FP {0}.png'.format(model))
    #plt.savefig(directory + '{0}.png'.format(model))
    plt.close()

######################
# Time analysis plot #
######################
prob_list_wrapped = [split_label_6(label) for label in model_list]
# create a model list with just the last 5 characters of each model 
model_list_filename = [model[39:60] for model in model_list]
# Ploting metrics
num_models = len(model_list)
indices = np.arange(num_models)
bar_width = 0.25
fig, ax = plt.subplots()
bars1 = ax.bar(indices, np.array(recall_prida), bar_width, label='Recall')
bars2 = ax.bar(indices + bar_width, np.array(precision_prida), bar_width, label='Precision')
bars3 = ax.bar(indices + 2 * bar_width, np.array(f1_prida), bar_width, label='F1 Score')
ax.set_xlabel('Models')
ax.set_ylabel('Scores')
ax.set_title('Performance Metrics by Model')
ax.set_xticks(indices + bar_width)
ax.set_xticklabels(prob_list_wrapped, fontsize=6)  
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.minorticks_on()
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='gray')
ax.legend(fontsize=8) 
plt.tight_layout()
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/ROC_curves/metrics_rippAI/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + '{0}.png'.format(model_list_filename))
plt.close()


pdb.set_trace()