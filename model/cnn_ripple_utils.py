import scipy.io
import pandas as pd
import numpy as np
import h5py
import sys
import os
import math
import pickle
from tensorflow import keras
from keras.models import Sequential
from keras import layers, optimizers
from keras.initializers import GlorotUniform, Orthogonal
# from xgboost import XGBClassifier
# from imblearn.under_sampling import RandomUnderSampler
import pdb
from scipy.signal import decimate
from scipy.signal import butter, filtfilt

def fcn_save_pickle(name,x):
    '''
    void fcn_save_pickle(name,x) \n
    Generates a pickle file in path n, containing variable x\n
    '''
    with open(name, 'wb') as handle:
            pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def fcn_load_pickle(name):
    '''
    [x] = fcn_load_pickle(name) loads the content of the pickle file to x
    '''
    with open(name, 'rb') as handle:
            return( pickle.load(handle) )


# Estas dos funciones están normalmente en bz_load_binary, se pueden mover allá
def loadChunk(fid, nChannels, channels, nSamples, precision):
    size = int(nChannels * nSamples * precision)
    nSamples = int(nSamples)

    data = fid.read(size)

    # fromstring to read the data as int16
    # reshape to give it the appropiate shape (nSamples x nChannels)
    data = np.fromstring(data, dtype=np.int16).reshape(nSamples, len(channels))
    data = data[:, channels]

    return data

def bz_LoadBinary(filename, nChannels, channels, sampleSize, verbose=False):

    if (len(channels) > nChannels):
        print("Cannot load specified channels (listed channel IDs inconsistent with total number of channels).")
        return

    #aqui iria CdE de filename
    with open(filename, "rb") as f:
        dataOffset = 0

        # Determine total number of samples in file
        fileStart = f.tell()
        if verbose:
            print("fileStart ", fileStart)
        status = f.seek(0, 2) # Go to the end of the file
        fileStop = f.tell()
        f.seek(0, 0) # Back to the begining
        if verbose:
            print("fileStop ", fileStop)

        # (floor in case all channels do not have the same number of samples)
        maxNSamplesPerChannel = math.floor(((fileStop-fileStart)/nChannels/sampleSize))
        nSamplesPerChannel = maxNSamplesPerChannel

        # For large amounts of data, read chunk by chunk
        maxSamplesPerChunk = 10000
        nSamples = int(nSamplesPerChannel*nChannels)

        if verbose:
            print("nSamples ", nSamples)

        if nSamples <= maxNSamplesPerChannel:
            data = loadChunk(f, nChannels, channels, nSamples, sampleSize)
        else:
            # Determine chunk duration and number of chunks
            nSamplesPerChunk = math.floor(maxSamplesPerChunk/nChannels)*nChannels
            nChunks = math.floor(nSamples/nSamplesPerChunk)

            if verbose:
                print("nSamplesPerChannel ", nSamplesPerChannel)
                print("nSamplesPerChunk ", nSamplesPerChunk)

            # Preallocate memory
            data = np.zeros((nSamplesPerChannel,len(channels)), dtype=np.int16)

            if verbose:
                print("size data ", np.size(data, 0))

            # Read all chuncks
            i = 0
            for j in range(nChunks):
                d = loadChunk(f, nChannels, channels, nSamplesPerChunk/nChannels, sampleSize)
                m = np.size(d, 0)

                if m == 0:
                    break

                data[i:i+m, :] = d
                i = i+m

            # If the data size is not a multiple of the chunk size, read the remainder
            remainder = nSamples - nChunks*nSamplesPerChunk
            if remainder != 0:
                d = loadChunk(f, nChannels, channels, remainder/nChannels, sampleSize)
                m = np.size(d, 0)

                if m != 0:
                    data[i:i+m, :] = d

    return data

# Functions used to load the raw LFP, select channels, load ripples, downsample and normalize
def load_lab_data(path):

    sf, expName, ref_channels, dead_channels = load_info(path)

    if not (expName == 'ThyNpx'):
        ripples = load_ripples(path)/sf
        channels_map = load_channels_map(path)
    else:
        channels_map = np.hstack((np.expand_dims(np.arange(1,385), axis=1), np.ones((384,1)), np.zeros((384,1))))
        ripples = scipy.io.loadmat(os.path.join(path, "ripples.mat"))['ripples']

    channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)

    # Select channels to make dataset
    # pdb.set_trace()
    # channels_in_shank = list(np.where(np.array(shanks)==shank-1)[0])
    # data = data[:, channels_in_shank]
    if expName == '2021-01-22_13-08-20':
        shank = 4
        channels_in_shank = list(np.where(np.array(shanks)==shank-1)[0])
        LFP = load_raw_data(path, expName, channels, verbose=True)
        LFP = LFP[:, channels_in_shank]
    elif expName == 'ThyNpx':
        LFP = load_raw_data(path, expName, channels, verbose=True)
        # pdb.set_trace()
        channels_in_shank = np.array(ref_channels['pyr'])[::3][1:-1]
        LFP = LFP[:, channels_in_shank]
    else:
        LFP = load_raw_data(path, expName, channels, verbose=True)
    return (LFP, ripples)


def load_info(path):
    try:
        mat = scipy.io.loadmat(os.path.join(path, "info.mat"))
    except:
        print("info.mat file does not exist.")
        sys.exit()

    sf = mat["fs"][0][0]
    expName = mat["expName"][0]

    ref_channels = {}
    ref_channels["so"] = mat["so"][0]
    ref_channels["pyr"] = mat["pyr"][0]
    ref_channels["rad"] = mat["rad"][0]
    ref_channels["slm"] = mat["slm"][0]


    if len(mat["chDead"]) <= 0:
        dead_channels = []
    else:
        dead_channels = [x-1 for x in (mat["chDead"][0]).astype(int)]

    return sf, expName, ref_channels, dead_channels

def load_ripples(path, verbose=False):
    try:
        dataset = pd.read_csv(os.path.join(path,"ripples.csv"), delimiter=' ', header=0, usecols = ["ripIni","ripEnd"])# "ripMiddle", "ripEnd", "type", "shank"])
    except:
        print(path+"ripples.csv file does not exist.")
        sys.exit()

    ripples = dataset.values
    ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]
    if verbose:
        print("Loaded ripples: ", len(ripples))

    return ripples

def load_channels_map(path):
    try:
        dataset = pd.read_csv(path+"/mapsCh.csv", delimiter=' ', header=0)
    except:
        print("ripples.csv file does not exist.")
        sys.exit()

    channels_map = dataset.values

    return channels_map

def reformat_channels (channels_map, ref_channels):
    channels = np.where(np.isnan(channels_map[:, 0]) == False, channels_map[:, 0], 0)
    channels = [x-1 for x in (channels).astype(int)]

    shanks = np.where(np.isnan(channels_map[:, 1]) == False, channels_map[:, 1], 0)
    shanks = [x-1 for x in (shanks).astype(int)]

    ref_channels["so"] = np.where(np.isnan(ref_channels["so"]) == False, ref_channels["so"], 0)
    ref_channels["so"] = [x-1 for x in ref_channels["so"].astype(int)]
    ref_channels["pyr"] = np.where(np.isnan(ref_channels["pyr"]) == False, ref_channels["pyr"], 0)
    ref_channels["pyr"] = [x-1 for x in ref_channels["pyr"].astype(int)]
    ref_channels["rad"] = np.where(np.isnan(ref_channels["rad"]) == False, ref_channels["rad"], 0)
    ref_channels["rad"] = [x-1 for x in ref_channels["rad"].astype(int)]
    ref_channels["slm"] = np.where(np.isnan(ref_channels["slm"]) == False, ref_channels["slm"], 0)
    ref_channels["slm"] = [x-1 for x in ref_channels["slm"].astype(int)]

    return channels, shanks, ref_channels

def load_raw_data(path, expName, channels, verbose=False):

    # There is .dat file
    is_dat = any([file.endswith(".dat") for file in os.listdir(path)])

    # There is .eeg file
    is_eeg = any([file.endswith(".eeg") for file in os.listdir(path)])

    # There is .mat file with the name of the last folder
    is_mat = any([os.path.basename(os.path.normpath(path))+".mat" in file for file in os.listdir(path)])

    if is_dat:
        name_dat = os.listdir(path)[np.where([file.endswith(".dat") for file in os.listdir(path)])[0][0]]
        if verbose:
            print(path+"/"+name_dat)
        data = bz_LoadBinary(path+"/"+name_dat, len(channels), channels, 2, verbose)

    elif is_eeg:
        name_eeg = os.listdir(path)[np.where([file.endswith(".eeg") for file in os.listdir(path)])[0][0]]
        if verbose:
            print(path+"/"+name_eeg)
        data = bz_LoadBinary(path+"/"+name_eeg, len(channels), channels, 2, verbose)

    elif is_mat:
        folder = path + "/" + os.path.basename(os.path.normpath(path))+".mat"
        if verbose:
            print(folder)
        try:
            mat = scipy.io.loadmat(folder)
            data = mat["fil"]
        except:
            mat = h5py.File(folder, 'r')
            data = np.array(mat["fil"]).T
    else:
        print('Not data found')

    return data


def downsample_data(data, sf, downsampled_fs):

    # Dowsampling
    if sf > downsampled_fs:
        # Apply bandpass FIR filter 0.1-500 Hz
        nyquist = sf/2
        n_taps = 251  # Filter order
        fir_coeff = scipy.signal.firwin(n_taps, [0.5/nyquist, 500/nyquist], pass_zero=False, window='hamming')
        filtered_data = scipy.signal.filtfilt(fir_coeff, [1.0], data, axis=0)
        
        # Downsample 
        downsampled_data = decimate(filtered_data, int(sf/downsampled_fs), axis=0, zero_phase=True, ftype='fir', n=251)

    # Upsampling
    elif sf < downsampled_fs:
        print("Original sampling rate below 1250 Hz!")
        return None
    
    else:
        downsampled_data = data


    # Change from int16 to float16 if necessary
    # int16 ranges from -32,768 to 32,767
    # float16 has ±65,504, with precision up to 0.0000000596046
    if downsampled_data.dtype != 'float32':
        downsampled_data = np.array(downsampled_data, dtype="float32")

    return downsampled_data


def z_score_normalization(data):
    channels = range(np.shape(data)[1])

    for channel in channels:
        # Since data is in float16 type, we make it smaller to avoid overflows
        # and then we restore it.
        # Mean and std use float64 to have enough space
        # Then we convert the data back to float16
        dmax = np.amax(data[:, channel])
        dmin = abs(np.amin(data[:, channel]))
        dabs = dmax if dmax>dmin else dmin
        m = np.mean(data[:, channel] / dmax, dtype='float64') * dmax
        s = np.std(data[:, channel] / dmax, dtype='float64') * dmax
        s = 1 if s == 0 else s # If std == 0, change it to 1, so data-mean = 0
        data[:, channel] = ((data[:, channel] - m) / s).astype('float32')

    return data


def load_data_fs(path, shank, verbose=False):
    # Read info.mat
    sf, expName, ref_channels, dead_channels = load_info(path)

    #Read mapsCh.csv
    channels_map = load_channels_map(path)

    # Reformat channels into correct values
    channels, shanks, ref_channels = reformat_channels(channels_map, ref_channels)
    # Read .dat
    data = load_raw_data(path, expName, channels, verbose=verbose)

    return data, sf


def generate_overlapping_windows(data, window_size, stride, sf):
    window_pts = int(window_size * sf)
    stride_pts = int(stride * sf)
    r = range(0, data.shape[0], stride_pts)

    new_data = np.empty((len(list(r)), window_pts, data.shape[1]))

    cont = 0
    for idx in r:
        win = data[idx:idx+window_pts, :]

        if (win.shape[0] < window_pts):
            continue

        new_data[cont,:,:]  = win

        cont = cont+1

    return new_data

def filter_LFP(LFP, sf=1250, band='low'):
    """
    Filters the LFP data in a specified frequency band using FIR filter.

    Args:
        LFP: Input LFP data of shape [num_samples, num_channels].
        sf: Sampling frequency of the data.
        band: Frequency band to filter ('low' for below 100 Hz, 'high' for 100-300 Hz).

    Returns:
        Filtered LFP data.
    """
    from scipy import signal

    # Set filter parameters based on band
    if band == 'low':
        print('Filtering low band')
        lowcut = 8.0
        highcut = 40.0
    elif band == 'high':
        print('Filtering high band')
        lowcut = 120.0
        highcut = 250.0
    elif band == 'muax':
        print('Filtering muax band')
        lowcut = 300.0
        highcut = 6000.0
    else:
        raise ValueError("Invalid band. Choose 'low' or 'high'.")

    # Design and apply FIR filter
    nyquist = sf/2
    n_taps = 251  # Filter order
    fir_coeff = signal.firwin(n_taps, [lowcut/nyquist, highcut/nyquist], 
                             pass_zero=False, window='hamming')
    filtered_LFP = signal.filtfilt(fir_coeff, [1.0], LFP, axis=0)

    if band == 'muax':
        filtered_LFP = np.abs(filtered_LFP)

    return filtered_LFP

def process_LFP(LFP, ch=np.arange(0,8), sf=30000, new_sf=1250, use_zscore=True, use_band=None):
    '''
    This function processes the LFP before calling the detection algorithm.
    1. It extracts the desired channels from the original LFP, and interpolates where there is a value of -1.
    2. Downsamples the LFP to 1250 Hz.
    3. Normalizes each channel separately by z-scoring them.
    Mandatory inputs:
        LFP: 		LFP recorded data (np.array: n_samples x n_channels).
        sf: 		Original sampling frequency (in Hz).
        channels: 	channel to which compute the undersampling and z-score normalization. Counting starts in 0.
                    If channels contains any -1, interpolation will be also applied.
                    See channels of rippl_AI.predict(), or aux_fcn.interpolate_channels() for more information.
    Output:
        LFP_norm: normalized LFP (np.array: n_samples x len(channels)). It is undersampled to 1250Hz, z-scored,
                    and transformed to used the channels specified in channels.
    A Rubio, LCN 2023
    '''
    # data=interpolate_channels(LFP,channels)
    data = LFP
    if use_band is not None:
        if use_band == 'augment':
            data = filter_LFP(data, sf=sf, band=use_band)
        else:
            data = filter_LFP(data, sf=sf, band=use_band)
            
    if sf!=1250:
        print('Downsampling data at {} Hz...'.format(new_sf))
        data = downsample_data(data, sf, downsampled_fs=new_sf)
        print("Shape of downsampled data:",data.shape)
    else:
        print("Data is already sampled at 1250 Hz!")
    if use_zscore:
        print('Normalizing data...')
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
        normalized_data = data
        # normalized_data=z_score_normalization(data)
    else:
        print('Not normalized data.')
        normalized_data=data
    return normalized_data


def prediction_parser(LFP,arch='CNN1D',model_number=1,new_model=None,n_channels=None,n_timesteps=None):
    '''
    [y] = prediction_parser(LFP,model_sel)
    Computes the output of the model passed in params \n
    Inputs:
        LFP:			 [n x 8] lfp data,subsampled and z-scored
     Optional inputs:
        arch:   		 string containing the name of the architecture
        model_number:    int, if 1 the best model will be used to predict
        new_model:       keras model, retrained model, if not empty, it will be used to predict
                         IMPORTANT: make sure the new_model architecture and the 'arch' parameter match
            *If a new model is used, n_channels and n_timesteps need to be passed too*
            n_channels:  int, number of channels in new model
            n_timesteps:  int, number of timesteps per window of new model
    Output:
        y: (n) shape array with the output of the chosen model
    A Rubio, LCN 2023
    '''
    # If no new_model is passed:
    # Looks for the name of the selected model
    if new_model==None:
        for filename in os.listdir('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models/'):
            if f'{arch}_{model_number}' in filename:
                break
        print(filename)
        sp=filename.split('_')
        n_channels=int(sp[2][2])
        timesteps=int(sp[4][2:])
    else: # Manually set n_channels and timesteps to match the retrained model parameters
        n_channels=n_channels
        timesteps=n_timesteps

    #print(f'Validating arquitecture {arch} using {n_channels} channels and {timesteps} timesteps')
    # Input shape: number of channels

    input_len=LFP.shape[0]
    # Make sure the input data and the model number of
    assert n_channels==LFP.shape[1],f'The model expects {n_channels} channels and the data has {LFP.shape[1]}'
    # Input shape: timesteps
    if arch=='XGBOOST':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps*n_channels)
        y_predict= np.zeros(shape=(input_len,1,1))
        if new_model==None:
            xgb=XGBClassifier()
            xgb.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
        else:
            xgb=new_model
        windowed_signal=xgb.predict_proba(LFP)[:,1]
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
    elif arch=='SVM':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps*n_channels)
        y_predict= np.zeros(shape=(input_len,1,1))
        # model load
        if new_model==None:

            clf=fcn_load_pickle(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))#.calibrated_classifiers_[0]
        else:
            clf=new_model
        windowed_signal= clf.predict_proba(LFP)[:,1]
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
        # y_predict: after expanding the windows, to be compatible with perf array
    elif arch=='LSTM':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        # Model load
        if new_model==None:
           model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
        else:
           model = new_model
        y_predict = model.predict(LFP,verbose=1)
        y_predict=y_predict.reshape(-1,1,1)
        y_predict=np.append(y_predict,np.zeros(shape=(input_len%timesteps,1,1))) if (input_len%timesteps!=0) else y_predict
    elif arch=='CNN1D':
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename), compile=False)
        else:
            model=new_model
        model.compile(loss="binary_crossentropy", optimizer=optimizer)

        windowed_signal = model.predict(LFP, verbose=True)
        windowed_signal=windowed_signal.reshape(-1)
        y_predict=np.zeros(shape=(input_len,1,1))
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
    elif arch=='CNN2D':
        if new_model==None:
            model = keras.models.load_model(os.path.join('/mnt/hpc/projects/OWVinckSWR/Carmen/DBI2/rippl-AI/optimized_models',filename))
        else:
            model=new_model
        LFP=LFP[:len(LFP)-len(LFP)%timesteps,:].reshape(-1,timesteps,n_channels,1)
        y_predict= np.zeros(shape=(input_len,1,1))
        windowed_signal= model.predict(LFP,verbose=1)
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
    else:
        raise ValueError(f'The introduced architecture -{arch}- does not match the existing ones.')

    return(y_predict.reshape(-1))

# Selecting the index functions

def get_predictions_index(predictions,threshold=0.5):
    '''
        [pred_indexes] = get_predictions_index(predictions, thershold)\n
        Returns the begining and ending samples of the events above a given threshold\n
        Inputs:
            predictions:	X, array with the continuous output of a model (even the Gt)
            threshold:		float, signal intervals above this value will be considered events

        Output:
            pred_indexes:	Nx2, array containing the begining and ending index sample of the events
        '''
    aux=np.copy(predictions)
    aux[aux>=threshold]=1
    aux[aux<threshold]=0
    dif=np.diff(aux,axis=0)
    begin_indexes=np.where(dif==1)[0]
    end_indexes=np.where(dif==-1)[0]

    if len(begin_indexes)>len(end_indexes):
        begin_indexes=begin_indexes[:-1]
    elif len(begin_indexes)<len(end_indexes):
        end_indexes=end_indexes[1:]
    #pdb.set_trace()
    pred_indexes=np.empty(shape=(len(begin_indexes),2))
    pred_indexes[:,0]=begin_indexes
    pred_indexes[:,1]=end_indexes

    return pred_indexes

def get_predictions_indexes(data, predictions, window_size, stride, fs=1250, threshold=0.5):
    window_pts = window_size * fs
    stride_pts = stride * fs
    pred_indexes = []

    for i_pred, pred in enumerate(predictions.flatten()):
        if (pred >= threshold):
            ini = i_pred * stride_pts
            end = ini + window_pts - 1

            pred_indexes.append(np.array([ini, end]))


    return np.array(pred_indexes)

def middle_stamps(pred_ind):
    ''' [mids]=middle_stamps(pred_ind) returns the middle stamps of the events passed in pred_ind
    '''
    mids=[]
    for pred in pred_ind:
        mids.append(int((pred[1]+pred[0])//2))
    return(np.array(mids))

def get_click_th(event):
    if event.xdata<0:
        x=0
    elif event.xdata>1:
        x=1
    else:
        x=event.xdata
    return(x)

def format_predictions(path,preds,sf):
    '''
    format_predictions(path,preds,sf) writes a .txt with the initial and end times of events in seconds
    inputs:
        path       str, absolute path of the file that will be created
        preds      (2,n_events) np.array with the initial and end timestamps of the events
        sf         int, sampling frequency of the data
    A Rubio, LCN 2023
    '''
    f=open(path,'w')

    preds=preds/sf
    for pred in preds:

        f.write(str(pred[0])+' ')
        f.write(str(pred[1]))
        f.write('\n')
    f.close()
    return


# Performance (precission, recall, F1) metrics

def get_performance(pred_events, true_events, threshold=0, exclude_matched_trues=False, verbose=True):

    '''
    [precision, recall, F1, TP, FN, IOU] = get_performance(pred_events, true_events)
    Computes all these measures given a cell array with boundaries of pred_events

    Inputs:
        pred_events		 Nx2 matrix with start and end of pred events (seconds)
        true_events		 Mx2 matrix with start and end of true events (seconds)
        threshold		   Threshold to IoU. By default is 0
        exclude_matched_trues False by defaut (one true can match many predictions)

    Output:
        precision		   Metric indicating the percentage of correct
                            predictions out of total predictions
        recall			  Metric indicating the percentage of true events
                            predicted correctly
        F1				  Metric with a measure that combines precision and recall.
        TP				  Nx1 matrix indicating which pred event detected a true event, and
                            a true positive (True) or was false negative (False)
        FN				  Mx1 matrix indicating which true event has been detected (False)
                            or not (True)
        IOU				 NxM array with intersections over unions

    A Navas-Olive, LCN 2020
    '''

    # Check similarity of pred and true events by computing if intersection over union > 0
    # Compute IOU
    [IOU, IOU_pred, IOU_true] = intersection_over_union(pred_events, true_events)
    # IOU_pred-> IOU de cada evento predecido
    # IOU_true-> IOU de la GT


    # Excluye los true coincidentes
    if exclude_matched_trues:
        # Take maximal IOUs, and make the rest be zero
        pred_with_maxIOU = np.argmax(IOU, axis=0)
        IOU_pred_one_true_match = np.zeros_like(IOU)
        for itrue, ipred in enumerate(pred_with_maxIOU):
            IOU_pred_one_true_match[ipred, itrue] = IOU[ipred, itrue]
        # True positive: Predicted event that has a IoU with any true > 0
        TP = (IOU_pred_one_true_match.sum(axis=1) > threshold)
        # False negative: Predicted event that has not a IoU with any true
        FN = (IOU_true <= threshold)

    else:
        # True positive: Predicted event that has a IoU with any true > 0
        #
        TP = (IOU_pred>threshold)
        # False negative: Predicted event that has not a IoU with any true
        FN = (IOU_true<=threshold)

    # Precision and recall
    precision = np.mean(TP)      # Media de verdaderos positivos: 1 si todas las predicciones son aciertos
    recall = 1. - np.mean(FN)    # 1-media de falsos negativos: 1 si toda la GT está presente en los aciertos
    F1 = 2. * (precision * recall) / (precision + recall)
    if (precision + recall) == 0:
        F1 = 0.
    else:
        F1 = 2. * (precision * recall) / (precision + recall)

    if verbose:
        print('precision =', precision)
        print('recall =', recall)
        print('F1 =', F1)

    # Variable outputs
    return precision, recall, F1, TP, FN, IOU

def intersection_over_union(x, y):
    '''
    IOU = intersection_over_union(x, y) computes the percentage of
    intersection over their union between every two pair of intervals
    x and y.

    Inputs:
      x	 Nx2 array with beginnings and ends of 1D events
      y	 Mx2 array with beginnings and ends of 1D events

    Output:
      IOU   NxM array with intersections over unions
      IOUx  (optional) Nx1 array with indexes of y events with maximal IOU
            It's zero if IOU=0 for all y events
      IOUy  (optional) Mx1 array with indexes of x events with maximal IOU
            It's zero if IOU=0 for all x events

    A Navas-Olive, LCN 2020
    '''

    if (len(x)>0) and (len(y)>0):

        # Initialize
        Intersection = np.zeros((x.shape[0],  y.shape[0]),dtype=np.float32)
        Union = np.ones((x.shape[0],  y.shape[0]),dtype=np.float32)
        # Go through every y (beginning-end) pair
        for iy in range(y.shape[0]):
            # Intersection duration: difference between minimum end and maximum ini
            Intersection[:, iy] = np.maximum( np.minimum(x[:, 1], y[iy, 1]) - np.maximum(x[:, 0], y[iy, 0]), 0)
            # Union duration: sum of durations of both events minus its intersection
            Union[:, iy, None] = np.diff(x, axis=1) + np.diff(y[iy, :]) - Intersection[:, iy, None]

        # Compute intersection over union
        IOU = Intersection / Union

        # Compute which events from y have maximal IOU with x
        IOUx = np.max(IOU, axis=1, keepdims=True)

        # Compute which events from x have maximal IOU with y
        IOUy = np.max(IOU, axis=0, keepdims=True)

        # Optional outputs
        return IOU, IOUx, IOUy

    elif len(x)==0:

        print('x is empty. Cant perform IoU')
        return np.array([]), np.array([]), np.zeros((y.shape[0], 1))

    elif len(y)==0:

        print('y is empty. Cant perform IoU')
        return np.array([]), np.zeros((1, x.shape[0])), np.array([])


# Interpolation
def interpolate_channels(data, ch_map):

    interp_data = np.zeros((data.shape[0], len(ch_map)))
    for idx, ch in enumerate(ch_map):
        if ch>-1:
            interp_data[:,idx] = data[:,ch]
        else:
            pre_ch_idx = np.where(np.array(ch_map[:idx])>-1)[0][-1]
            pre_ch = ch_map[pre_ch_idx]
            post_ch_idx = np.where(np.array(ch_map[idx+1:])>-1)[0][0]+idx+1
            post_ch = ch_map[post_ch_idx]
            ch_dist = post_ch_idx - pre_ch_idx
            interp_data[:,idx] = data[:, pre_ch] + ((idx-pre_ch_idx)/ch_dist) * \
                (data[:, post_ch] - data[:, pre_ch])
    return interp_data

# Retraining auxiliary functions
def split_data(x,GT,window_dur=60,sf=1250,split=0.7):
    '''
    [x_test,y_test,x_train,y_train] = split_data(x,y,window_dur,sf,split)\n
    Performs the data train-test split, with the proportion specified in 'split' going to train. The data is shuffled in windows of 'window_dur' seconds\n
    Inputs:
        x:			[n X n_channels] matrix with the LFP values of the session
        GT:			[n events x 2]	initial and end times of each events
        window_dur: float, length in seconds of the chunks that will be asigned randomly to train or validation subsets
        sf:			int, sampling frequency of the passed data
        split:		float, proportion of windows that will be asigned to the train subset (the final proportion will diverge, being random)

    Output:
        x_test:			[test_samples x n_channels]: Test subset input
        y_test:			[n_test_events x 2]: Test subset output
        x_train:		[train_samples x n_channels]: training subset input
        y_train:		[n_train_events x 2,]: training subset output
    A Rubio LCN 2023
    '''
    n_samples_window=window_dur*sf
    n_windows=len(x)//n_samples_window

    y=np.zeros(shape=len(x))

    for event in GT:
        y[int(sf*event[0]):int(sf*event[1])]=1
    x_test=[]
    x_train=[]
    y_train=[]
    y_test=[]
    n_channels=x.shape[1]
    rand_arr= np.random.rand(n_windows)
    for i in range(n_windows):
        if rand_arr[i]>=split:
            x_test=np.append(x_test,np.arange(i*n_samples_window,(i+1)*n_samples_window))
            y_test=np.append(y_test,np.arange(i*n_samples_window,(i+1)*n_samples_window))
            # x_test=np.append(x_test,x[i*n_samples_window:(i+1)*n_samples_window])
            # y_test=np.append(y_test,y[i*n_samples_window:(i+1)*n_samples_window])
        else:
            x_train=np.append(x_train,np.arange(i*n_samples_window,(i+1)*n_samples_window))
            y_train=np.append(y_train,np.arange(i*n_samples_window,(i+1)*n_samples_window))
            # x_train=np.append(x_train,x[i*n_samples_window:(i+1)*n_samples_window])
            # y_train=np.append(y_train,y[i*n_samples_window:(i+1)*n_samples_window])

    x_test=x[x_test.astype(int), :]
    x_train=x[x_train.astype(int), :]
    y_test=y[y_test.astype(int)]
    y_train=y[y_train.astype(int)]
    x_test=np.reshape(x_test,(-1,n_channels))
    x_train=np.reshape(x_train,(-1,n_channels))
    events_test=get_predictions_index(y_test)/sf
    events_train=get_predictions_index(y_train)/sf
    return x_test,events_test,x_train,events_train

def retraining_parser(arch,x_train_or,events_train,x_test,events_test,params=None):
    '''
    [model,y_train,y_test] = retraining_parser(arch,x_train_or,events_train,x_test,events_test,params=None)\n
    Performs the retraining of the best model of the desired architecture  \n
    Inputs:
        arch:			string, with the desired architecture model to be retrained
        x_train_or:		[n train samples x 8], normalized LFP that will be used to retrain the model
        events_train: 	[n train events x 2], begin and end timess of the train events
        x_test_or:		[n test samples x 8], normalized LFP that will be used to retrain the model
        events_train: 	[n test events x 2], begin and end timess of the test events
        Optional inputs
            params: dictionary, with the parameters that will be use in each specific architecture retraining
            - In 'XGBOOST': not needed
            - In 'SVM':
                params['Undersampler proportion']. Any value between 0 and 1. This parameter eliminates
                                samples where no ripple is present untill the desired proportion is achieved:
                                Undersampler proportion= Positive samples/Negative samples
            - In 'LSTM', 'CNN1D' and 'CNN2D':
                params['Epochs']. The number of times the training data set will be used to train the model
                params['Training batch']. The number of windows that will be processed before updating the weights
    Output:
        model: The retrained model
        y_train_p: [n_train_samples], output of the model using the training data
        y_test_p:  [n_test_samples], output of the model using the test data
    A Rubio LCN 2023
    '''
    # Input data preparing for training
    x_train = np.copy(x_train_or)
    # Sampling frequency hard fixed to 1250
    y_train= np.zeros(shape=(len(x_train)))
    for ev in events_train:
        y_train[int(1250*ev[0]):int(1250*ev[1])]=1

    y_test= np.zeros(shape=(len(x_test)))
    for ev in events_test:
        y_test[int(1250*ev[0]):int(1250*ev[1])]=1

    x_train_len=x_train.shape[0]
    x_test_len=x_test.shape[0]

    # Automatically hard coded to input the required shape for the best model of each arch
    if arch=='XGBOOST':
        n_channels=8
        timesteps=16
        # Making the input data and expected output compatible with he resizing
        x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps*n_channels)
        y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
        y_train=rec_signal(y_train_aux)

        x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps*n_channels)
        y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)
        y_test=rec_signal(y_test_aux)
        # model load
        model=XGBClassifier()
        model.load_model(os.path.join('optimized_models','XGBOOST_1_Ch8_W60_Ts016_D7_Lr0.10_G0.25_L10_SCALE1'))

        model.fit(x_train, y_train,verbose=True,eval_set = [(x_test, y_test)])

        y_train_p=np.zeros(shape=(x_train_len,1,1))
        train_signal=model.predict_proba(x_train)[:,1]
        for i,window in enumerate(train_signal):
            y_train_p[i*timesteps:(i+1)*timesteps]=window
        y_test_p=np.zeros(shape=(x_test_len,1,1))
        test_signal=model.predict_proba(x_test)[:,1]
        for i,window in enumerate(test_signal):
            y_test_p[i*timesteps:(i+1)*timesteps]=window
    elif arch=='SVM':
        n_channels=8
        timesteps=1
        # Making the input data and expected output compatible with he resizing
        x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps*n_channels)
        y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
        y_train=rec_signal(y_train_aux)

        x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps*n_channels)
        y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)
        y_test=rec_signal(y_test_aux)

        #Under sampler: discards windows where there is no ripples untill the desired proportion between ripple/no ripple is achieved
        # If no params is provided, the defect proportion will be 0.5
        if params==None:
            us_prop=0.5
        else:
            us_prop=params['Unsersampler proportion']
        rus = RandomUnderSampler(sampling_strategy=us_prop)
        x_train_us, y_train_us = rus.fit_resample(x_train, y_train)

        print(f"Under sampling result: {x_train_us.shape}")
        # model load
        model=fcn_load_pickle(os.path.join('optimized_models','SVM_1_Ch8_W60_Ts001_Us0.05'))
        # model fit
        model=model.fit(x_train_us, y_train_us)

        y_train_p=np.zeros(shape=(x_train_len,1,1))
        train_signal=model.predict_proba(x_train)[:,1]
        for i,window in enumerate(train_signal):
            y_train_p[i*timesteps:(i+1)*timesteps]=window
        y_test_p=np.zeros(shape=(x_test_len,1,1))
        test_signal=model.predict_proba(x_test)[:,1]
        for i,window in enumerate(test_signal):
            y_test_p[i*timesteps:(i+1)*timesteps]=window
    elif arch=='LSTM':
        n_channels=8
        timesteps=32
        x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels)
        y_train=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,1)
        x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels)
        y_test=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,1)
        print("Input and output shape: ",x_train.shape,y_train.shape)
        model = keras.models.load_model(os.path.join('optimized_models','LSTM_1_Ch8_W60_Ts32_Bi0_L4_U11_E10_TB256'))
        # If no parameters are provided, 5 epochs and 32 as training batch will be used
        if params==None:
            epochs=5
            tb=32
        else:
            epochs=params['Epochs']
            tb=params['Training batch']
        model.fit(x_train, y_train, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)

        y_train_p = model.predict(x_train,verbose=1)
        y_train_p=y_train_p.reshape(-1,1,1)
        y_test_p = model.predict(x_test,verbose=1)
        y_test_p=y_test_p.reshape(-1,1,1)
    elif arch=='CNN1D':
        n_channels=8
        timesteps=16
        x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels)
        y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps)
        x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels)
        y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps)

        y_train=np.zeros(shape=[x_train.shape[0],1])
        for i in range(y_train_aux.shape[0]):
            y_train[i]=1  if any (y_train_aux[i]==1) else 0
        print("Train Input and Output dimension", x_train.shape,y_train.shape)

        y_test=np.zeros(shape=[x_test.shape[0],1])
        for i in range(y_test_aux.shape[0]):
            y_test[i]=1  if any (y_test_aux[i]==1) else 0

        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model = keras.models.load_model(os.path.join('optimized_models','CNN1D_1_Ch8_W60_Ts16_OGmodel12'), compile=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        if params==None:
            epochs=20
            tb=32
        else:
            epochs=params['Epochs']
            tb=params['Training batch']
        model.fit(x_train, y_train,shuffle=False, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)
        y_train_p=np.zeros(shape=(x_train_len,1,1))
        train_signal=model.predict(x_train)
        for i,window in enumerate(train_signal):
            y_train_p[i*timesteps:(i+1)*timesteps]=window
        y_test_p=np.zeros(shape=(x_test_len,1,1))
        test_signal=model.predict(x_test)
        for i,window in enumerate(test_signal):
            y_test_p[i*timesteps:(i+1)*timesteps]=window
    elif arch=='CNN2D':
        n_channels=8
        timesteps=40
        x_train=x_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,n_channels,1)
        y_train_aux=y_train[:x_train_len-x_train_len%timesteps].reshape(-1,timesteps,1)
        x_test=x_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,n_channels,1)
        y_test_aux=y_test[:x_test_len-x_test_len%timesteps].reshape(-1,timesteps,1)

        y_train=np.zeros(shape=[x_train.shape[0],1])
        for i in range(y_train_aux.shape[0]):
            y_train[i]=1  if any (y_train_aux[i]==1) else 0
        y_test=np.zeros(shape=[x_test.shape[0],1])
        for i in range(y_test_aux.shape[0]):
            y_test[i]=1  if any (y_test_aux[i]==1) else 0

        model = keras.models.load_model(os.path.join('optimized_models','CNN2D_1_Ch8_W60_Ts40_OgModel'))
        # If no parameters are provided, 20 epochs and 32 as training batch will be used
        if params==None:
            epochs=20
            tb=32
        else:
            epochs=params['Epochs']
            tb=params['Training batch']
        model.fit(x_train, y_train,shuffle=False, epochs=epochs,batch_size=tb,validation_data=(x_test,y_test), verbose=1)
        y_train_p=np.zeros(shape=(x_test_len,1,1))
        train_signal=model.predict(x_train)
        for i,window in enumerate(train_signal):
            y_train_p[i*timesteps:(i+1)*timesteps]=window
        y_test_p=np.zeros(shape=(x_test_len,1,1))
        test_signal=model.predict(x_test)
        for i,window in enumerate(test_signal):
            y_test_p[i*timesteps:(i+1)*timesteps]=window

    return(model,y_train_p.reshape(-1),y_test_p.reshape(-1))

def rec_signal(y):
    '''
    rec_signal compatibilizes the ground truth signal y with the expected output of XGBOOST and SVM architectures
        input:	(n_samples,n_windows) ground truth signal containing 0 or 1 indicating the presence of ripples
        outpur: (n_samples) collapsed ground truth, if any samples in the input window contains ripple, the collapsed
                signal contains a 1
    '''
    len=np.shape(y)[0]
    print(np.shape(y))
    r_signal=np.zeros(shape=(len))
    for i,w in enumerate(y):
        if any(w)==1:
            r_signal[i]=1
    return r_signal

def save_model (model,arch,path):
    '''
    Model save parser
        Input:
            model: actual model to save
            arch:   string, architecture of the model
            path:  string, path to the saved model
    A Rubio LCN 2023
    '''
    if (arch=='CNN2D' or arch=='CNN1D' or arch=='LSTM'):
        if not os.path.exists(f'{path}'):
            os.mkdir(f'{path}')
        model.save(f'{path}')
    elif arch=='XGBOOST':
        model.save_model(f'{path}')
    else:
        fcn_save_pickle(f'{path}',model)
    return


# Explore functions
def build_LSTM(input_shape,n_layers=3,layer_size=20,bidirectional=False):
    '''
    model = build_LSTM(input_shape,lr,dropout_rate,n_layers,layer_size,seed,bidirectional)
    Builds the specified LSTM model \n
    Inputs:
        input_shape:
        x:				[timesteps x n_channels] input dimensionality of the data,
        n_layers: 		int, # of LSTM layers
        layer_size: 	int, # number of LSTM units per layer
        bidirectional:	bool, True if the models processes backwards from the end of the window, and the usual forward pass from the begininning simultaneously

    Output:
        model: LSTM keras model
    '''
    keras.backend.clear_session()
    dropout_rate=0.2			# Hard fix to a standard value
    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    #LSTM layers
    if bidirectional==False:
        for i in range(n_layers):
            if i==0:
                x = keras.layers.LSTM(layer_size, return_sequences=True,
                                    kernel_initializer=GlorotUniform(),
                                    recurrent_initializer=Orthogonal(),
                                    )(inputs)
                x = keras.layers.Dropout(dropout_rate)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate)(x)
            else:
                x = keras.layers.LSTM(layer_size, return_sequences=True,
                            kernel_initializer=GlorotUniform(),
                            recurrent_initializer=Orthogonal(),
                            )(x)
                x = keras.layers.Dropout(dropout_rate)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate)(x)
    else: # Bidirectional
        for i in range(n_layers):
            if i==0:
                x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size, return_sequences=True,
                                    kernel_initializer=GlorotUniform(),
                                    recurrent_initializer=Orthogonal(),
                                    ) )(inputs)
                x = keras.layers.Dropout(dropout_rate,)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate,)(x)
            else:
                x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size, return_sequences=True,
                            kernel_initializer=GlorotUniform()),
                            recurrent_initializer=Orthogonal()
                            ) (x)
                x = keras.layers.Dropout(dropout_rate, )(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate, )(x)

    predictions = keras.layers.Dense(1, activation='sigmoid',kernel_initializer=GlorotUniform())(x)
    # Define model
    model = keras.models.Model(inputs=inputs,
                               outputs=predictions,
                               name='BCG_LSTM')

    opt = keras.optimizers.Adam(learning_rate=0.005)  # Hard fixed to 0.005
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

    return model

def build_CNN2D(conf, input_shape = (50,8,1)):
    '''
    model = build_CNN2D(conf,input_shape,learning_rate)
    Builds the specified CNN2D model \n
    Inputs:
        conf:	[n_l x 2] list. n_l is the number of 2Dconvolotional, Batch Normalization and LeakyRelu sets.
                Each row shape: [n_k k_h k_w] n_k is the number of kernels of the layer, which defines the output shape
                                                 k_h is the width of each 2D kernel
                                              k_w is the height of each 2D kernel
        input_shape:			[n x 8] lfp data,subsampled and z-scored
        learning_rate:	float, the size step multiplier for the weight updates during training

    Output:
        model: keras model with the specified parameters
    A Rubio LCN 2023
    '''
    n_max_pool_l=len(conf)//2
    assert 2**n_max_pool_l<=input_shape[1],"\nThe resulting model will decrease the input dimensionality too much.Try one of the following:\n1. Increase the number of used channels.\n2. Decrease the number of layers."
    model = Sequential()
    for i,layer_conf in enumerate (conf):
        if i==0:
            model.add(layers.Conv2D(filters = layer_conf[0], kernel_size=(layer_conf[1],layer_conf[2]), activation='relu',
                        input_shape=input_shape, padding='same', strides = (1,1)))
        else:
            model.add(layers.Conv2D(filters = layer_conf[0], kernel_size=(layer_conf[1],layer_conf[2]), activation='relu',
                        padding='same'))
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        if i<n_max_pool_l:
            model.add(layers.MaxPooling2D((2, 2)))


    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    model.build()
    # Learning rate hard fixed in 1e-5
    model.compile(
        optimizer= optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['mse']
    )
    return model

def build_CNN1D(n_channels,timesteps,conf):
    '''
        model = build_CNN1D(n_channels, timesteps, conf)\n
        Returns a 1D convolutional neural network. If the desired configuration will create problems, and exception with sugestions is thrown\n
        Inputs:
            n_channels:		int, number of used channels (1,3 or 8)
            timesteps:		int, number of timestamps that will be used to feed the model
            conf:			[n_l x 2] list. n_l is the number of 1Dconvolotional, Batch Normalization and LeakyRelu sets.
                Each row shape: [n_k k_w]. n_k is the number of kernels of the layer, which defines the output shape
                                           k_w is the width and the stride of the kernels
                For more clarifications, visit:
        Output:
            model: CNN1D model, consisting of a number of conv1D-BatchNorm-LeakyReLU sets followed by a dense layer.
                    Input: (N,timesteps,n_channels) --> Output: (N,1,1,)
    A Rubio LCN 2023
    '''
    o_shape_arr=[]
    for i,layer_conf in enumerate(conf):
        if i==0:
            o_shape_arr.append(timesteps//layer_conf[1])
        else:
            o_shape_arr.append(o_shape_arr[i-1]//layer_conf[1]) # The output shape of the layer n is the output of the n-1 layer /stride
    #assert 0 not in o_shape_arr, "\nThe chosen kernel dimensionality will cause problems.\nTry one of the following:\n1. Increment time window (timesteps) size.\n2. Decrement the 1st kernel size.\n3. Decrement the rest of the kernels size"

    assert o_shape_arr[-1]==1, '\nThe output of the model is not shaped (1,1).\nTry incrementing the kernel size of the final layers'

    keras.backend.clear_session()
    # input layer
    model = keras.models.Sequential()
    for i,layer_conf in enumerate(conf):
        if i==0: # 1st layer: special case, the input shape needs to be defined
            model.add(keras.layers.Conv1D(filters=layer_conf[0], kernel_size=layer_conf[1], strides=layer_conf[1], padding='valid', input_shape=(timesteps,n_channels)))
        else:
            model.add(keras.layers.Conv1D(filters=layer_conf[0], kernel_size=layer_conf[1], strides=layer_conf[1], padding='valid'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.LeakyReLU(alpha=0.1))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    opt = keras.optimizers.Adam(learning_rate=0.001)  # Hard fixed
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])
    return model
def real_ripple_times(times):
    # Real ripple times for CNN ripple
    if times.size == 0:
        times_masked = np.array([0., 0.])
    else:
        onset, offset = np.moveaxis(times, 1, 0)
        mask_onset = []
        mask_offset = []

        for i in range(len(times)-1):
            if (times[i][1] >= times[i+1][0] and times[i][1] <= times[i+1][1]):
                mask_offset.append(i)
                mask_onset.append(i+1)

        times_new_onset = np.delete(onset, mask_onset)
        times_new_offset = np.delete(offset, mask_offset)

        times_masked = np.column_stack((times_new_onset, times_new_offset))
    return times_masked
