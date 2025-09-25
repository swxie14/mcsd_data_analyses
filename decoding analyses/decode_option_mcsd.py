"""
created 25.6.27

decoding option during performace of the first from ephys data

"""

import numpy as np
import os
import utils as u
import statsmodels.api as sm
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def decode_option_mcsd(group='behEphys',region='ACC',subjs=[]):
    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name
    print(subjs)

    #subjs.remove('subj_059')
    #subjs.remove('subj_057')
    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)] #stores number of spikes for each patient in each cell
    options = [None] * n_subjs
    min_options = np.zeros(n_subjs)


    for s_idx in range(n_subjs):
        # load data
        beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        num_trials = beh_data.shape[0]
        choice_1 = beh_data['choice1'].values
        choice_2 = beh_data['choice2'].values


        cell_names = ephys_data.columns
        num_cells = cell_names.shape[0]
        num_cells_region = np.sum(np.array([ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region for c_idx in range(num_cells)]))

        options[s_idx] = choice_1 - 1 + choice_1 - 1 + choice_2 - 1 + 1
        if num_cells_region > 0:
            min_options[s_idx] = np.min(np.histogram(options[s_idx], bins=4)[0])
        # get task beginning and session length
        num_trials = beh_data.shape[0]
        t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1 #creates buffer of 1000 to account for beginning
        t_session = beh_data['tOutOn'].iloc[-1]  + 2000 - t_beginning #creates buffer for ending

        # make timesamp index mats for first 1s of state 1
        state_1_t = [np.array([], dtype=int) for _ in range(num_trials)]
        for trial in range(num_trials):
            diff = beh_data['tState1On'].iloc[trial] - beh_data['tFix'].iloc[trial]
            # Check if tSubgoalResp is not an empty array
            t_subgoal = beh_data['tSubgoalOn'].iloc[trial]
            if isinstance(t_subgoal, np.ndarray) and t_subgoal.size > 0:
                diff = beh_data['tSubgoalOn'].iloc[trial] - beh_data['tFix'].iloc[trial]
            x_state_1 = np.arange(0, diff)
            t_tmp = beh_data['tFix'].iloc[trial] - t_beginning
            state_1_t[trial] = x_state_1 + t_tmp + 400
            state_1_t[trial] = state_1_t[trial].astype(int)


        for c_idx in range(num_cells):
            if ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region:
                spike_times = ephys_data[cell_names[c_idx]].values[0][0][0] - t_beginning
                spikes_cell = np.zeros(int(t_session))
                spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
                spikes_cell[spike_times.astype(int)] = 1

                # Initialize epoch_spikes array
                epoch_spikes = np.zeros(num_trials)
                
                # Sum spikes during state 1 period for each trial
                for trial in range(num_trials):
                    if state_1_t[trial].size > 0:  # Check if there are time points
                        epoch_spikes[trial] = np.sum(spikes_cell[state_1_t[trial]]) / len(state_1_t[trial])
                    else:
                        epoch_spikes[trial] = 0  # should not happen if data has been preprocessed correctly

                #print(epoch_spikes)
                if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5:
                        
                    X = np.array(range(num_trials)) + 1
                    X = sm.add_constant(X) 
                    model = sm.GLM(epoch_spikes, X, family=sm.families.Poisson()).fit()
                    if model.pvalues[1] < 0.05:
                        epoch_spikes = epoch_spikes - model.fittedvalues
                        
                    # z-score firing rate
                    epoch_spikes = (epoch_spikes - np.mean(epoch_spikes)) / np.std(epoch_spikes)

                    # save spikes, choices, and options
                    if len(spikes[s_idx]) == 0:
                        spikes[s_idx] = epoch_spikes                                # initialize 2d array
                    else:
                        spikes[s_idx] = np.vstack((spikes[s_idx], epoch_spikes))    # add new rows


                    
    #resize min_options to the number of cells in the region
    subjects_with_cells = min_options > 0
    min_options = min_options[subjects_with_cells]
    options = [options[s_idx] for s_idx in range(len(options)) if subjects_with_cells[s_idx]]
    n_subjs = len(min_options)


    
    # Filter all lists to only include subjects with cells in the region
    spikes = [spikes[s_idx] for s_idx in range(len(spikes)) if subjects_with_cells[s_idx]]



    # initialize subsampling and cross-validation parameters
    n_folds = 5
    n_perms = 1000
    n_subsamp = int(np.min(min_options))
    
    choices = np.concatenate([np.ones(n_subsamp), np.ones(n_subsamp)*2, np.ones(n_subsamp)*3, np.ones(n_subsamp)*4])

    accuracies = np.zeros(n_perms)
    within_class_acc = np.zeros((n_perms, 4))  # Changed from 2 to 4 for 4 options

    for p_idx in range(n_perms):
        perm_spikes = []
        for s_idx in range(n_subjs):
            if spikes[s_idx].size > 0:  # only run if there are any cells for that subject
                
                # Ensure spikes[s_idx] is 2D
                if spikes[s_idx].ndim == 1:
                    spikes[s_idx] = spikes[s_idx].reshape(1, -1)
                
                for cI in range(spikes[s_idx].shape[0]):
                    # get indices of random trials for each option
                    perm_options = np.empty(n_subsamp*4)
                    for o_idx in range(4):
                        o_tmp = np.where(options[s_idx] == o_idx+1)[0]
                        o_tmp = np.random.permutation(o_tmp)[:n_subsamp]
                        perm_options[(o_idx)*n_subsamp:(o_idx+1)*n_subsamp] = o_tmp
                    perm_options = perm_options.astype(int)

                    # index the cell spike counts by the balanced trials
                    if len(perm_spikes) == 0:
                        perm_spikes = spikes[s_idx][cI, perm_options].reshape(-1, 1)
                    else:
                        included_spikes = spikes[s_idx][cI, perm_options].reshape(-1, 1)
                        perm_spikes = np.hstack((perm_spikes, included_spikes))
                


        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        perm_accuracies = np.zeros(n_folds)
        perm_within_class_acc = np.zeros((n_folds, 4)) 
        
        for f_idx, (train_idx, test_idx) in enumerate(skf.split(perm_spikes, choices)):
            X_train, X_test = perm_spikes[train_idx,:], perm_spikes[test_idx,:]
            y_train, y_test = choices[train_idx], choices[test_idx]

            clf = LinearSVC(penalty='l2',C=1,max_iter=10000)
            clf.fit(X_train, y_train)

            # 5. Predict and evaluate
            y_pred = clf.predict(X_test)
            perm_accuracies[f_idx] = accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            perm_within_class_acc[f_idx,:] = cm.diagonal() / cm.sum(axis=1)

        accuracies[p_idx] = np.mean(perm_accuracies)
        within_class_acc[p_idx,:] = np.mean(perm_within_class_acc, axis=0)

    print(region + ' overall accuracy = ' + str(np.mean(accuracies)))
    print(region + ' within-class accuracies = ' + str(np.mean(within_class_acc, axis=0)))

    return accuracies, within_class_acc

if __name__ == "__main__":
    regions = ["ACC", "A", "H", "OF", "PT", "CM", "SMA", "PRV"]
    for region in regions:
        decode_option_mcsd(region=region)
    #decode_option_mcsd()






