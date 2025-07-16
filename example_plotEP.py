import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from calculate_reaction_times import calculate_reaction_times


csv_file = "CheckerboardFace_20250716_071748.csv"
prefix = csv_file.split('_')[0]

if prefix == "CheckerboardFace":
    montage = ['Fz','C3','Cz','C4','Pz','PO7','POz','PO8']
    N_ch = len(montage)
    trigger_ch = N_ch+1
    response_ch = N_ch+2
    trigger_oi = 3
    info_vec = [{"label": "Low Frequency", "ID": 1},
                {"label": "High Frequency", "ID": 2},
                {"label": "Faces", "ID": 3}]
    cmap = ['#ff8080','#ff0000','#0000ff']
    perform_car = True
elif prefix == "AEPOddball":
    montage = ['Fz','C3','Cz','C4','Pz','PO7','POz','PO8']
    N_ch = len(montage)
    trigger_ch = N_ch+1
    response_ch = N_ch+2
    trigger_oi = 1
    info_vec = [{"label": "Target", "ID": 1},
                {"label": "Nontarget", "ID": 2}]
    cmap = ['#0000ff','#ff0000']
    perform_car = False
elif prefix == "AEPSingleStim":
    montage = ['Fz','C3','Cz','C4','Pz','PO7','POz','PO8']
    N_ch = len(montage)
    trigger_ch = N_ch+1
    response_ch = None
    info_vec = [{"label": "Auditory Stimulus", "ID": 1}]
    cmap = ['#ff0000']
    perform_car = False
else:
    print("Unknown Paradigm type.")

fs = 250        # Sampling Frequency in Hz
t_pre = 0.1     # Time before stimulus in seconds
t_post = 0.7    # Time after stimulus in seconds

data = pd.read_csv(csv_file, header=0).values
trigger = np.round(data[:, trigger_ch].astype(float)).astype(int)
y = data[:, np.arange(1,N_ch+1)].astype(float)
if response_ch is not None:
    response = np.round(data[:, response_ch].astype(float)).astype(int)

# Preprocessing - Bandpass
f_lo = 1
f_hi = 30
b, a = butter(N=2,
                Wn=[f_lo / (fs / 2), f_hi / (fs / 2)],
                btype='bandpass')
y = filtfilt(b=b, a=a, x=y, axis=0)

# Preprocessing - Notch
fc = 50.0  # center frequency
bw = 2.0   # bandwidth
b, a = butter(N=2,
                Wn=[(fc - bw) / (fs / 2), (fc + bw) / (fs / 2)],
                btype='bandstop')
y = filtfilt(b=b, a=a, x=y, axis=0)

if perform_car:
    y = y - np.mean(y,axis=1, keepdims=True)    # Common Average Reference

n_vec_trig = np.arange(-int(t_pre * fs), int(t_post * fs) + 1)
t_vec_trig = n_vec_trig / fs

# Prepare figure
fig, axes = plt.subplots(int(np.ceil(N_ch/4)), 4, figsize=(14, 9), sharex=True, sharey=True)
axes = axes.flatten()

# Store one line per condition for legend
legend_lines = []
legend_labels = []
for ch_idx, ax in enumerate(axes):
    for info in info_vec:
        trig = np.where(trigger == info["ID"])[0]
        y_trig = np.array([y[idx + n_vec_trig, ch_idx]
                           for idx in trig if idx + n_vec_trig[-1] < len(y)])
        if y_trig.size == 0:
            continue
        y_mean = np.mean(y_trig, axis=0)
        # Plot line and collect one per condition
        line, = ax.plot(t_vec_trig, y_mean, linewidth=1, color=cmap[info["ID"]-1],
                        label=info["label"] if ch_idx == 0 else None)
        if ch_idx == 0:
            legend_lines.append(line)
            legend_labels.append(info["label"])
    ax.axvline(x=0, color='k', linestyle='-', linewidth=2)  # ← vertical trigger line
    ax.set_title(montage[ch_idx], fontsize=10)
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(0, t_post + 0.1, 0.1))
    ax.grid(True)
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(-12, 12)

# Shared axis labels
fig.text(0.5, 0.02, 'Time relative to stimulus onset (s)', ha='center', fontsize=10)
fig.text(0.04, 0.5, 'Amplitude (µV)', va='center', rotation='vertical', fontsize=10)

# Add legend outside plot
fig.legend(legend_lines, legend_labels, fontsize=9)

plt.tight_layout(rect=[0.05, 0.03, 0.95, 0.97])
plt.show()
