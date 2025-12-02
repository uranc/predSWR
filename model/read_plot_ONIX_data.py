import numpy as np
import matplotlib.pyplot as plt

# Load data from breakout board tutorial workflow
path = '.' # Set to the directory of your data from the example workflow
suffix = '10' # Change to match file names' suffix
plt.close('all')

#%% Metadata
dt = {'names': ('time', 'acq_clk_hz', 'block_read_sz', 'block_write_sz'),
      'formats': ('datetime64[ns]', 'u4', 'u4', 'u4')}
meta = np.genfromtxt(path + '\\start-time_' + suffix + '.csv', delimiter=',', dtype=dt, skip_header=1, encoding='utf-8')
print(f"Recording was started at {meta['time']} GMT")

#%% Analog Inputs
analog_input = {}
analog_input['time'] = np.fromfile(path + '\\analog-clock_' + suffix + '.raw', dtype=np.uint64) / meta['acq_clk_hz']
analog_input['data'] = np.reshape(np.fromfile(path + '\\analog-data_' + suffix + '.raw', dtype=np.float32), (-1, 12))
n_analog = min(analog_input['time'].shape[0], analog_input['data'].shape[0])
analog_input['time'] = analog_input['time'][:n_analog]
analog_input['data'] = analog_input['data'][:n_analog, :]



from XX_load_bonsai_data import load_bonsai_data
import numpy as np
import pdb

# Block 1: Load LFP data
lfp_file_path = path + '\\rhd2164-amplifier_' + suffix + '.raw'
lfp_data = load_bonsai_data(file_path=lfp_file_path, dtype=np.uint16, channels=32)
lfp_data_microvolts = ((lfp_data.astype(np.int32) - 32768).astype(np.float32)) * 0.195 # Convert to int16

print(f"LFP data shape: {lfp_data.shape}")


# Block 1: Load LFP data
lfp_clock_file_path = path + '\\rhd2164-clock_' + suffix + '.raw'
lfp_clock_data = load_bonsai_data(file_path=lfp_clock_file_path, dtype=np.uint64, channels=1) / meta['acq_clk_hz']

n_rhd = min(lfp_clock_data.shape[0], lfp_data_microvolts.shape[0])
lfp_clock_data = lfp_clock_data[:n_rhd]
lfp_data_microvolts = lfp_data_microvolts[:n_rhd, :]

# plt.plot(lfp_clock_data[-30000*600:]*1000, np.mean(lfp_data_microvolts[-30000*600:,:], axis=1), 'k-')
ttl = analog_input['data'][:,4] > 3
onsets = np.flatnonzero(np.diff(ttl.astype(np.int8)) == 1) + 1
#onsets = onsets[400:]
# plt.plot(analog_input['time'], analog_input['data'][:,10])
# plt.show()
# pdb.set_trace()
nplot = int(np.ceil(np.sqrt(onsets.shape[0]//2)))
print(onsets.shape)
n=0
for onset in onsets:
      plt.figure(figsize=(17,8))
      n+=1
      # if n==256:
      #     break
      # plt.subplot(nplot,nplot,n)
      n_stride = 10
      pre_onset = 50000
      post_onset = 50000
      # Use np.logical_and to avoid dtype surprises
      analog_sel = np.squeeze((analog_input['time'][onset-pre_onset] < lfp_clock_data) & \
             (lfp_clock_data < analog_input['time'][onset+post_onset]))
      # plt.plot(lfp_clock_data[analog_sel]*1000, 2*lfp_data_microvolts[analog_sel,:]+np.linspace(0, 4000, 32)[None,:], 'k-')
      # analog_sel = analog_sel[::3]
      # pdb.set_trace()
      xx = lfp_clock_data[analog_sel]
      plt.plot((lfp_clock_data[analog_sel][::3]-xx[0])*1000, 2*lfp_data_microvolts[analog_sel,6:18][::3]+np.linspace(0, 5000, 12)[None,:], 'k-')

      t_win = analog_input['time'][onset - pre_onset:onset + post_onset:n_stride] * 1000 - analog_input['time'][onset - pre_onset] * 1000
      plt.plot(t_win, analog_input['data'][onset - pre_onset:onset + post_onset:n_stride, 4] / 4 * 10000) #pulsepal output
      plt.plot(t_win, np.ones(analog_input['data'][onset - pre_onset:onset + post_onset:n_stride, 8].shape) * 8500, 'k-') #Threshold
      plt.plot(t_win, analog_input['data'][onset - pre_onset:onset + post_onset:n_stride, 8] * 10000) #probability
      plt.plot(t_win, analog_input['data'][onset - pre_onset:onset + post_onset:n_stride, 11] * 10000) # signal to pulsepal 
   

      plt.title(onset)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,0]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,1]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,2]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,3]*1000/3.3)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,4]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,5]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,6]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,7]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,8]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,9]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,10]*500/4)
      # plt.plot(analog_input['time'][onset-pre_onset:onset+post_onset]*1000, analog_input['data'][onset-pre_onset:onset+post_onset,11]*500/4)
      # # plt.plot(analog_input['time'][-100000*60:]*1000, analog_input['data'][-100000*60:,5]*1000)
      # plt.xlabel("time (millisec)")
      # plt.ylabel("volts")

      # plt.legend(['Digital Out to PulsePal', 'PulsePal Stimulation Output', 'Model Prob'])
      # plt.legend(['Avg. Raw LFP', 'Pulse Pal Stimulation Output'])

      # plt.legend(['Avg. Raw LFP', 'Trigger (Model Threshold Crossing)', 'Pulse Pal Stimulation Output', 'Model Output'])

      plt.show()
pdb.set_trace()  # 
