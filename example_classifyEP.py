import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
from my_trigger import my_trigger
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score


csv_file = "CheckerboardFace_20250716_071748.csv"
prefix = csv_file.split('_')[0]

if prefix == "CheckerboardFace":
    montage = ['Fz','C3','Cz','C4','Pz','PO7','POz','PO8']
    N_ch = len(montage)
    trigger_ch = N_ch+1
    response_ch = N_ch+2
    combine_trigger = [(1,[2])] # merge checkerboards
    class_vec = [1,3]
elif prefix == "AEPOddball":
    montage = ['Fz','C3','Cz','C4','Pz','PO7','POz','PO8']
    N_ch = len(montage)
    trigger_ch = N_ch+1
    response_ch = N_ch+2
    combine_trigger = []
    class_vec = [2,1]
else:
    print("Unknown Paradigm type.")

fs = 250
t_pre = 0.1
t_post = 0.7
f_band = np.array([0.5,15]) # Frequency band of interest for feature extraction

data = pd.read_csv(csv_file, header=0).values
trigger = np.round(data[:, trigger_ch].astype(float)).astype(int)
y = data[:, np.arange(1,N_ch+1)].astype(float)

if combine_trigger:
    for target, sources in combine_trigger:
        trigger[np.isin(trigger, sources)] = target

# Filter Data 
[b, a] = butter(N=4, Wn=f_band/(fs/2), btype='bandpass')
y_filt = filtfilt(b=b, a=a, x=y, axis=0)

# Trigger/Epoch Data
[Y, t_vec] = my_trigger(y=y_filt, group_id=trigger, class_vec=class_vec, fs=fs, t_pre=t_pre, t_post=t_post, spec='list')

# Downsample Data
K = int(np.floor(fs // (2*f_band[-1])))
Y_r = []
for cur_Y in Y:
    Y_r.append(cur_Y[::K,:,:])

# Create Feature matrix and Labels vector
X_parts = [] # temporary feature matrix of n, p
y_parts = [] # temp labels of n,
for i, cur_Y in enumerate(Y_r):
    # Flatten: (time × channels, trials)
    cur_Y = cur_Y.reshape(-1, cur_Y.shape[2])  # shape: (time * channels, trials)

    # Transpose to shape (n_trials, features)
    X_parts.append(cur_Y.T)

    # Create label vector for current class
    y_parts.append(np.full(cur_Y.shape[1], i))  # i = class index

# Concatenate all at once
X = np.vstack(X_parts)      # shape: (total_trials, features)
y = np.concatenate(y_parts) # shape: (total_trials,)


# Classification --------------------------------------------------------------------
# Parameters
n_repeats = 10
n_folds = 10
priors = [0.5, 0.5]
shrinkage = 0.05    # regularization

# Store accuracy and pooled predictions
accuracies = []
y_true_all = []
y_score_all = []

for repeat in range(n_repeats):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat)
    y_true_rep = []
    y_score_rep = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        mdl_LDA = LDA(solver='lsqr', shrinkage=shrinkage, priors=priors)
        mdl_LDA.fit(X_train, y_train)

        # Get decision scores (use predict_proba or decision_function)
        y_score = mdl_LDA.predict_proba(X_test)[:, 1]  # Probability of class 1
        y_score_rep.append(y_score)
        y_true_rep.append(y_test)

    # Pool predictions
    y_score_rep = np.concatenate(y_score_rep)
    y_true_rep = np.concatenate(y_true_rep)
    y_true_all.append(y_true_rep)
    y_score_all.append(y_score_rep)

    # Accuracy for this repetition
    y_pred_rep = (y_score_rep >= 0.5).astype(int)
    acc = accuracy_score(y_true_rep, y_pred_rep)
    accuracies.append(acc)

# Create the boxplot and display the individual data points as "fliers"
plt.figure(figsize=(4, 6))
plt.boxplot(accuracies, vert=True, patch_artist=True, 
            boxprops=dict(facecolor='lightblue'))

# Set limits and labels
plt.ylim(0, 1)
plt.xticks([])
plt.yticks(np.arange(0, 1 + 0.1, 0.1), fontsize=12)
plt.ylabel("Accuracy", fontsize=14)
plt.title(f"10×10 CV Accuracies\n Mean: {np.mean(accuracies):.2f}, SD: {np.std(accuracies):.2f}", fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


# Final pooled scores and labels for full AUC
y_true_all = np.concatenate(y_true_all)
y_score_all = np.concatenate(y_score_all)

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true_all, y_score_all, pos_label=1)
auc_score = roc_auc_score(y_true_all, y_score_all)

# Plot ROC AUC
plt.figure(figsize=(6, 6))  # Square aspect ratio
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}", color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray', label="Chance")

plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
plt.title("10x10 CV ROC Curve", fontsize=16)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xticks(np.arange(0, 1 + 0.1, 0.1), fontsize=12)
plt.yticks(np.arange(0, 1 + 0.1, 0.1), fontsize=12)
plt.gca().set_aspect('equal', adjustable='box')  # Force square plot
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()