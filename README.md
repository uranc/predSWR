# predSWR
- pred: TCN 
- pred2: TCN vs rippAI, Online 1D CNN
- pred3: using allen dataset online (1D CNN, 2D CNN, LSTM, TCN, M1, M2)
- pred4: compare TCNs
- online_sim: online simulation for CNN1D and LSTM
- hist: Hist_dect_ripp, Latencies, Dynamicthr_shift
- M1_M2_detection: computes times for M1 and M2
    LFP, ground truths + predictions
- f1_curves: f1 vs threshold, saving model with best F1
onset calculations: onset_images_avg_prob, onset_images_avg_rip(TPvsFP + prob), midpoint_corr_images_avg_rip (corr TPvsFP), corr_avg_LFP, metrics_rippAI(recall, precision, f1), onset_avg_prob_models, corr_avg_LFP_models (corr TPvsFP diff models), metrics_rippAI(recall, precision, f1) for diff models
- time_analysis: tprVsfpr, metrics, ROC (sklearn.metrics)