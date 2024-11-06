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
from model.cnn_ripple_utils import get_predictions_index, get_performance, intersection_over_union

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


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





# Prida dataset
#dir = '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/'
# model_list = ['online_prob_rippAI_CNN1D_0', 
#               'online_prob_rippAI_CNN2D_0',
#                'online_prob_rippAI_LSTM_0',
#             'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00',
#              ]

# Allen dataset
dir = '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities_online/'
model_list = ['online_prob_rippAI_allen_CNN1D_0', 
              'online_prob_rippAI_allen_CNN2D_0',
               'online_prob_rippAI_allen_LSTM_0',
            'Base_online_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00_0',
             ]

fpr = []
tpr = []
thresholds = []
roc_auc = []
roc_auc_prida = []
tp_prida = []
fp_prida = []
f1 = []
f1_prida = []
F1_track = []
recall = []
recall_prida = []
precision = []
precision_prida = []
optimal_thrs = []
th_arr = np.linspace(0.1,0.9,19)
prida_label = False
for model in model_list: 
    print(model)
    if model[0:2] == 'p_': 
        dir_prob_prida = '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities_online/'
        y_probs = np.load(dir_prob_prida + 'prob_rippAI_' + model[2:] + '_0.npy')
        prida_label = True
    elif model[0:2] == 'on':
        dir = '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities_online/'
        y_probs = np.load(dir + model + '.npy')
    else:
        dir = '/cs/projects/OWVinckSWR/DL/predSWR/allen_dataset/probabilities_online/'
        y_probs = np.load(dir + 'preds_val0_' + model + '.npy')
    
    # If they don't have the same length, remove the samples from y_true that are not present in y_probs
    y_true = y_true_complete[:len(y_probs)]
    
    # F1 rippleAI-style 

    # The input should be Nx2 matrix with start and end of pred/true events [seconds]
    onset_indices_true = np.where(np.diff(y_true) == 1)[0] / fs
    offset_indices_true = np.where(np.diff(y_true) == -1)[0] / fs
    true_events = np.vstack((onset_indices_true, offset_indices_true)).T
    print('True events: ', true_events.shape)
    # Need to choose a threshold to binarize the probabilities
    # To choose the thr is better to go trough the F1s and see which one is better 
    
    val_pred = [y_probs]
    val_labels = [true_events]
    F1_val=np.zeros(shape=(len(val_datasets),len(th_arr)))
    all_pred_events = []
    for j,pred in enumerate(val_pred):
        tmp_pred = []
        for i,th in enumerate(th_arr):
            pred_val_events=get_predictions_index(pred,th)/fs
            _,_,F1_val[j,i],_,_,_=get_performance(pred_val_events,val_labels[j],exclude_matched_trues=False,verbose=False)
            tmp_pred.append(pred_val_events)
        all_pred_events.append(tmp_pred)

    # threshold = th_arr[np.nanargmax(F1_val[0])]
    
    # y_probs_aux = (y_probs >= threshold).astype(int)
    # onset_indice_pred = np.where(np.diff(y_probs_aux) == 1)[0] / fs
    # offset_indices_pred = np.where(np.diff(y_probs_aux) == -1)[0] / fs
    # pred_events = np.vstack((onset_indice_pred, offset_indices_pred)).T
    
    # [precision, recall, F1, TP, FN, IOU] = get_performance(pred_events, true_events)
    # f1_prida.append(F1)
    # tp_prida.append(TP)
    # fp_prida.append(FN)
    # recall_prida.append(recall)
    # precision_prida.append(precision)
    F1_track.append(F1_val[0])
# Loading data for M1
pdb.set_trace()
dir = '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/'
F1_M1 = np.load(dir + 'F1_track_M1.npy')
th_arr_M1 = np.load(dir + 'th_arr_M1.npy')
# ploting f1 curve with thrs for all of the models 
model_list_filename = [model[39:60] for model in model_list]
fig, ax = plt.subplots()
colors = ['lightcoral', 'firebrick', 'grey', 'green']
colors = ['lightcoral', 'grey', 'k']
names = ['1D CNN', 'TCN', 'Power Based']
names = ['1D CNN', '2D CNN', 'LSTM', 'TCN']
for i, model in zip(range(len(F1_track)), model_list):
    if model[0:2] == 'p_': ax.plot(th_arr, F1_track[i], label = '1D CNN')
    elif model[0:2] == 'on': 
        ax.plot(th_arr, F1_track[i], lw= 2,color = colors[i] , label = names[i])
    #else: ax.plot(th_arr, F1_track[i], label = split_label(model_list[i]))
    elif model == 'Base_K4_T50_D3_N32_L4_E200_B32_S50_AnchorNarrowSmooth_GloWReg_Shift00': 
        ax.plot(th_arr, F1_track[i],lw= 2, color= 'firebrick', label = 'TCN')
#ax.plot(th_arr, F1_M1, color = "grey",lw= 2, label = 'Power Based')    
ax.set_xlabel('Threshold', fontsize=16)
ax.set_ylabel('F1 Score', fontsize=16)
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_title('F1 Score by Threshold', fontsize=16)
ax.legend(fontsize=14) 
# adding a grid 
#ax.grid(True, which='both', linestyle='--', linewidth=0.5)
#ax.minorticks_on()
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/F1_curveVSThr/'
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/images_pdf/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + 'Presentation_F1_curve_No_Excluding_matches.pdf')
plt.close()

# saving which model performed better (higher F1 value)

shape = F1_track[0].shape
stacked_array = np.stack(F1_track)
max_idx = np.nanargmax(stacked_array)
row_idx, col_idx = np.unravel_index(max_idx, stacked_array.shape)
max_value = stacked_array[row_idx, col_idx]
best_model = model_list[row_idx]
save_best_model = best_model + '__' + '{0}'.format(max_value)
directory = '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/best_models/'
if not os.path.exists(directory):
    os.makedirs(directory)
np.save(directory + '{0}_{1}.py'.format(save_best_model[39:60],model_list_filename), save_best_model)


pdb.set_trace()