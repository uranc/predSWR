import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import pdb
import os

def split_label(label):
    mid = len(label) // 2
    return label[:mid] + '\n' + label[mid:]

# y_true should contain the true binary labels (0 or 1)
# y_probs should contain the predicted probabilities for class 1
prob_list = ['prob_predSWR_TCN_BaseModel_k2_t40_d32_n128_e500_dilated_relu_dedilated_weightNorm_0.npy', 
             'prob_Average_K3_T200_D32_N64_B16_Le4_AnchorLossFix_0.npy', 
             'prob_Base_K3_T96_D32_N64_B32_Le4_AnchorLossFixNarrowNonZero_0.npy'
             ]

dir = '/cs/projects/OWVinckSWR/DL/predSWR/experiments/probabilities/'
y_true = np.load(dir + 'val_labels_0.npy')
fpr = []
tpr = []
thresholds = []
roc_auc = []
f1 = []
recall = []
precision = []
optimal_thrs = []
for prob in prob_list:
    y_probs = np.load(dir + prob)

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
    
prob_list_wrapped = [split_label(label) for label in prob_list]
# Ploting metrics
num_models = len(prob_list)
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
plt.savefig(directory + '{0}.png'.format(prob_list))
plt.close()


# Plot ROC curve
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
plt.legend(loc="lower right",fontsize=8)

# Saving images 
directory =  '/cs/projects/OWVinckSWR/DL/predSWR/time_analysis/ROC_curves/ROC_AUC/'   
if not os.path.exists(directory):
    os.makedirs(directory)
plt.savefig(directory + '{0}.png'.format(prob))
plt.close()


    




