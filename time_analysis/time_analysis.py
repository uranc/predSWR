import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import pdb
import os

def split_label(label):
    mid1 = len(label) // 2
    mid2 = mid1 // 2
    mid3 = mid1 + mid2
    return label[:mid2] + '\n' + label[mid2:mid1] + '\n' + label[mid1:mid3] + '\n' + label[mid3:]


# Split the label in 6 parts
def split_label_6(label): 
    mid1 = len(label) // 3
    mid2 = mid1 // 2
    mid3 = mid1 + mid2
    mid4 = mid3 + mid2
    mid5 = mid4 + mid2
    return label[:mid2] + '\n' + label[mid2:mid1] + '\n' + label[mid1:mid3] + '\n' + label[mid3:mid4] + '\n' + label[mid4:mid5]+ '\n' + label[mid5:]
    


# y_true should contain the true binary labels (0 or 1)
# y_probs should contain the predicted probabilities for class 1
# prob_list = ['Average_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#                 'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#               'Average_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense',
#               'Base_K4_T50_D3_N64_L3_E200_B64_S50_Focal_GloWReg_BottleDense',
#                'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalGap_GloWReg_BottleDense',
#                'Base_K4_T50_D3_N64_L3_E200_B64_S50_FocalSmooth_GloWReg_BottleDense', 
#                  'Average_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense', 
#            'Base_K4_T50_D3_N64_L3_E200_B32_S50_Hinge_GloWReg_BottleDense']
prob_list = [ 
              'Average_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense',
              'Base_K4_T50_D3_N64_L3_E200_B32_S50_Tversky07_GloWReg_BottleDense', 
              'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorWiderGap_GloWReg_BottleDense',
             'Average_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense',
             'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense']
# prob_list = ['prob_cnn_0.npy',
#              'Base_K4_T50_D3_N64_L3_E200_B64_S50_AnchorNarrowSmoothGap_GloWReg_BottleDense']

dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/'
y_true_aux = np.load(dir + 'val_labels_0.npy')
val_datasets = np.load(dir + 'val_dataset_0.npy')
y_true = np.zeros(val_datasets.shape[0])
for lab in y_true_aux:
    y_true[int(lab[0]*1250):int(lab[1]*1250)] = 1
fpr = []
tpr = []
thresholds = []
roc_auc = []
f1 = []
recall = []
precision = []
optimal_thrs = []
dir = '/cs/projects/OWVinckSWR/DL/predSWR/probs/'
for prob in prob_list:
    y_probs = np.load(dir + 'preds_val0_' + prob + '.npy')
    #y_probs = np.load('/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/prob_cnn_0.npy')
    # If they don't have the same length, remove the samples from y_true that are not present in y_probs
    if len(y_true) != len(y_probs):
        y_true = y_true[:len(y_probs)]

    # Compute ROC curve and ROC area for each class
    fpr_aux, tpr_aux, thresholds_aux = roc_curve(y_true, y_probs)
    roc_auc_aux = auc(fpr_aux, tpr_aux)

    fpr.append(fpr_aux)
    tpr.append(tpr_aux)
    thresholds.append(thresholds_aux)
    roc_auc.append(roc_auc_aux)

    ### tpr vs fpr vs thr
    plt.figure()
    plt.plot(thresholds_aux, tpr_aux, color='blue', lw=2, label='TPR')
    plt.plot(thresholds_aux, fpr_aux, color='red', lw=2, linestyle='--', label='FPR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title('TPR and FPR vs. Threshold {0}'.format(prob), fontsize=8)
    plt.legend(loc="best")
    directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/ROC_curves/tprVsfpr/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + '{0}.png'.format(prob))
    plt.close()

    # Selecting optimal threshold based on a criterion
    optimal_idx = np.argmax(tpr_aux - fpr_aux)
    optimal_threshold = thresholds_aux[optimal_idx]
    print('Optimal Threshold: ', optimal_threshold)
    # Analyze performance at the optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    precision_aux, recall_aux, _ = precision_recall_curve(y_true, y_probs)
    f1_aux = f1_score(y_true, y_pred_optimal)
    mean_recall = np.mean(recall_aux)
    recall.append(mean_recall)
    mean_precision = np.mean(precision_aux)
    precision.append(mean_precision)
    f1.append(f1_aux)
    optimal_thrs.append(optimal_threshold)
    
prob_list_wrapped = [split_label_6(label) for label in prob_list]
# Ploting metrics
num_models = len(prob_list)
model_list_filename = [model[39:60] for model in prob_list]
indices = np.arange(num_models)
bar_width = 0.25

fig, ax = plt.subplots()
bars1 = ax.bar(indices, np.array(recall), bar_width, label='Recall')
bars2 = ax.bar(indices + bar_width, np.array(precision), bar_width, label='Precision')
bars3 = ax.bar(indices + 2 * bar_width, np.array(f1), bar_width, label='F1 Score')
for i, thr in enumerate(optimal_thrs):
    ax.axhline(y=thr, color='red', linestyle='--', linewidth=1.5, label=f'Optimal thrs {i+1}', xmin=i/len(optimal_thrs), xmax=(i+1)/len(optimal_thrs))
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
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/ROC_curves/metrics/'
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + '{0}.png'.format(model_list_filename))
plt.close()


# Plot ROC curve
prob_list_wrapped = [split_label(label) for label in prob_list]
plt.figure()
for i in range(len(prob_list)):
    plt.plot(fpr[i], tpr[i], lw=2, label='{0}'.format(prob_list_wrapped[i]))
#plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-AUC curve')
plt.legend(loc="lower right",fontsize=6)

# Saving images 
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/ROC_curves/ROC_AUC/'   
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + '{0}.png'.format(prob))
plt.close()


    




