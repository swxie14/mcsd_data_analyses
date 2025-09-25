"""
created 25.7.15

testing decoding second_choice during performance of the first choice from ephys data

"""

import numpy as np
import os
import utils as u
import statsmodels.api as sm
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
from decode_second_choice_mcsd import decode_second_choice_mcsd

def test_second_choice_mcsd(group='behEphys',region='ACC',subjs=[]):
    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name
    #print(subjs)
    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)] #stores number of spikes for each patient in each cell
    choice_2 = [None] * n_subjs
    min_choices = np.zeros(n_subjs)

    shuffled_choices = [None] * n_subjs


    for s_idx in range(n_subjs):
        # load data
        beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        num_trials = beh_data.shape[0]
        choice_1 = beh_data['choice1'].values
        choice_2[s_idx] = beh_data['choice2'].values

        cell_names = ephys_data.columns
        num_cells = cell_names.shape[0]
        num_cells_region = np.sum(np.array([ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region for c_idx in range(num_cells)]))

        if num_cells_region > 0:
             min_choices[s_idx] = np.min([np.sum(choice_2[s_idx]==1), np.sum(choice_2[s_idx]==2)]) 
           

        t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1 #creates buffer of 1000 to account for beginning
        t_session = beh_data['tOutOn'].iloc[-1]  + 2000 - t_beginning #creates buffer for ending

        # make timesamp index mats for first 1s of state 1
        x_state_1 = np.arange(0, 1001) # why start with 200 and end with 1201?
        state_1_map = np.tile(x_state_1, (num_trials, 1))
        t_tmp = beh_data['tFix'].values - t_beginning
        t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_1_map.shape[1]))
        state_1_map = state_1_map + t_tmp
        state_1_map = state_1_map.astype(int) #stores timestamps of all notable time points (state1on)

        for c_idx in range(num_cells):
            if ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region:
                # make timesamp index mats for first 1s of state 1
                #for trials in range()

                spike_times = ephys_data[cell_names[c_idx]].values[0][0][0] - t_beginning
                spikes_cell = np.zeros(int(t_session))
                spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
                spikes_cell[spike_times.astype(int)] = 1

                epoch_spikes = np.sum(spikes_cell[state_1_map], axis=1) #gives number of spikes for each trial
                if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5:
                        
                    X = np.array(range(num_trials)) + 1
                    X = sm.add_constant(X) 
                    model = sm.GLM(epoch_spikes, X, family=sm.families.Poisson()).fit()
                    if model.pvalues[1] < 0.05:
                        epoch_spikes = epoch_spikes - model.fittedvalues
                        
                    # z-score firing rate
                    epoch_spikes = (epoch_spikes - np.mean(epoch_spikes)) / np.std(epoch_spikes)

                    # save spikes, choices
                    if len(spikes[s_idx]) == 0:
                        spikes[s_idx] = epoch_spikes                                # initialize 2d array
                    else:
                        spikes[s_idx] = np.vstack((spikes[s_idx], epoch_spikes))    # add new rows

                    
    #resize min_choices to the number of cells in the region
    subjects_with_cells = min_choices > 0
    min_choices = min_choices[subjects_with_cells]
    choice_2 = [choice_2[s_idx] for s_idx in range(len(choice_2)) if subjects_with_cells[s_idx]]
    n_subjs = len(min_choices)

    
    # Filter all lists to only include subjects with cells in the region
    spikes = [spikes[s_idx] for s_idx in range(len(spikes)) if subjects_with_cells[s_idx]]



    # initialize subsampling and cross-validation parameters
    n_folds = 5
    n_perms = 1000
    n_subsamp = int(np.min(min_choices))
    #print(n_subsamp)
    
    choices = np.concatenate([np.ones(n_subsamp), np.ones(n_subsamp)*2])

    null_accuracies = np.zeros(n_perms)
    null_within_class_acc = np.zeros((n_perms, 2))  

    for p_idx in range(n_perms):
        perm_spikes = []
        for s_idx in range(n_subjs):
            if spikes[s_idx].size > 0:  # only run if there are any cells for that subject
                #shuffle the choices
                shuffled_choices[s_idx] = np.random.permutation(choice_2[s_idx])

                # subsample the choices
                perm_choices = np.empty(n_subsamp*2)
                for c_idx in range(2):
                    c_tmp = np.where(shuffled_choices[s_idx] == c_idx+1)[0]
                    c_tmp = np.random.permutation(c_tmp)[:n_subsamp]
                    perm_choices[(c_idx)*n_subsamp:(c_idx+1)*n_subsamp] = c_tmp
                perm_choices = perm_choices.astype(int)


                # index the cells by the balanced trials
                if len(perm_spikes) == 0:
                    if spikes[s_idx].ndim == 1:
                        perm_spikes = spikes[s_idx][perm_choices].reshape(-1, 1)
                    else:
                        perm_spikes = spikes[s_idx][:,perm_choices].T
                else:
                    if spikes[s_idx].ndim == 1:
                        included_spikes = spikes[s_idx][perm_choices].reshape(-1, 1)  # reshape for hstack compatibility
                    else:
                        included_spikes = spikes[s_idx][:, perm_choices].T  # transpose to match hstack format
                    perm_spikes = np.hstack((perm_spikes, included_spikes))
                


        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        perm_accuracies = np.zeros(n_folds)
        perm_within_class_acc = np.zeros((n_folds, 2)) 

        
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


        null_accuracies[p_idx] = np.mean(perm_accuracies)
        null_within_class_acc[p_idx,:] = np.mean(perm_within_class_acc, axis=0)

    print(region + ' overall null accuracy = ' + str(np.mean(null_accuracies)))
    print(region + ' within-class null accuracies = ' + str(np.mean(null_within_class_acc, axis=0)))

    return null_accuracies, null_within_class_acc

if __name__ == "__main__":

    import pandas as pd
    regions = ["ACC", "A", "H", "OF", "PT", "CM", "SMA", "PRV", "IFG"]
    results = []

    for region in regions:
        null_accuracies, _ = test_second_choice_mcsd(region=region)
        real_accuracies, _ = decode_second_choice_mcsd(region=region)

        # Summary statistics
        real_mean_accuracy = np.mean(real_accuracies)
        null_mean_accuracy = np.mean(null_accuracies)
        p_value = np.mean(null_accuracies >= real_mean_accuracy)  # one-sided empirical p-value
        significance = p_value < 0.05

        # Collect row for table
        results.append([region,
                        round(real_mean_accuracy, 3),
                        round(null_mean_accuracy, 3),
                        round(p_value, 3),
                        significance])

    # ------------------------------------------------------------------
    # Pretty print results as a table
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(results, columns=[
        "Region", "Real Mean Acc.", "Null Mean Acc.", "p-value", "Significant"])

    print("\nDecoding Second Choice Results (real vs. shuffled)\n")
    print(results_df.to_string(index=False))
        
        






