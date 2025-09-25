"""
created 25.7.8

decoding option on two subgoals and testing on the third subgoal

"""

import numpy as np
import os
import utils as u
import statsmodels.api as sm
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

def option_ccgp_mcsd(group='behEphys',region='ACC',subjs=[], option_1 = 1, option_2 = 2, test_option = 3):
    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name
   # print(subjs)


    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)] #stores number of spikes for each patient in each cell
    spikes_test = [np.array([], dtype=int) for _ in range(n_subjs)] #stores number of spikes for each patient in each cell
    options = [None] * n_subjs
    options_test = [None] * n_subjs
    min_options = np.zeros(n_subjs)
    min_options_test = np.array([])



    for s_idx in range(n_subjs):
        # load data
        beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subjs[s_idx], rmv_flag=1)
        num_trials = beh_data.shape[0]
        choice_1 = beh_data['choice1'].values
        choice_2 = beh_data['choice2'].values

        train_trials = np.array([])
        test_trials = np.array([])

        for trial in range(num_trials):
            if beh_data['currSubgoal'].iloc[trial] == option_1 or beh_data['currSubgoal'].iloc[trial] == option_2:
                train_trials = np.append(train_trials, trial)
            else:
                test_trials = np.append(test_trials, trial)

        train_trials = train_trials.astype(int)
        test_trials = test_trials.astype(int)

        num_trials_train = len(train_trials)
        num_trials_test = len(test_trials)
        
        choice_1_test = choice_1[test_trials]
        choice_2_test = choice_2[test_trials]
        choice_1 = choice_1[train_trials]
        choice_2 = choice_2[train_trials]


        options[s_idx] = choice_1 - 1 + choice_1 - 1 + choice_2 - 1 + 1
        options_test[s_idx] = choice_1_test - 1 + choice_2_test - 1 + choice_2_test - 1 + 1



        cell_names = ephys_data.columns
        num_cells = cell_names.shape[0]
        num_cells_region = np.sum(np.array([ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region for c_idx in range(num_cells)]))

        
        if num_cells_region > 0:
            min_options[s_idx] = np.min(np.histogram(options[s_idx], bins=4)[0])
            if len(min_options_test) == 0:
                min_options_test = np.array([np.histogram(options_test[s_idx], bins=4)[0]])
            else:
                min_options_test = np.vstack((min_options_test, np.array([np.histogram(options_test[s_idx], bins=4)[0]])))
        #only use minimum of subjects with cells in the region
        

        # get task beginning and session length
        t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1 #creates buffer of 1000 to account for beginning
        t_session = beh_data['tOutOn'].iloc[-1]  + 2000 - t_beginning #creates buffer for ending

        # make timesamp index mats for first 1s of state 1
        state_1_t_train = [np.array([], dtype=int) for _ in range(num_trials_train)]
        state_1_t_test = [np.array([], dtype=int) for _ in range(num_trials_test)]
        for i, trial in enumerate(train_trials):
            diff = beh_data['tState1On'].iloc[trial] - beh_data['tFix'].iloc[trial]
            # Check if tSubgoalResp is not an empty array
            t_subgoal = beh_data['tSubgoalOn'].iloc[trial]
            if isinstance(t_subgoal, np.ndarray) and t_subgoal.size > 0:
                diff = beh_data['tSubgoalOn'].iloc[trial] - beh_data['tFix'].iloc[trial]
            x_state_1 = np.arange(0, diff)
            t_tmp = beh_data['tFix'].iloc[trial] - t_beginning
            state_1_t_train[i] = x_state_1 + t_tmp
            state_1_t_train[i] = state_1_t_train[i].astype(int)
        
        #do the same for test trials
        for i, trial in enumerate(test_trials):
            diff = beh_data['tState1On'].iloc[trial] - beh_data['tFix'].iloc[trial]
            # Check if tSubgoalResp is not an empty array
            t_subgoal = beh_data['tSubgoalOn'].iloc[trial]
            if isinstance(t_subgoal, np.ndarray) and t_subgoal.size > 0:
                diff = beh_data['tSubgoalOn'].iloc[trial] - beh_data['tFix'].iloc[trial]
            x_state_1 = np.arange(0, diff)
            t_tmp = beh_data['tFix'].iloc[trial] - t_beginning
            state_1_t_test[i] = x_state_1 + t_tmp
            state_1_t_test[i] = state_1_t_test[i].astype(int)

        # for each subject, creates z-scored spike count across all trials for each cell
        for c_idx in range(num_cells):
            if ephys_data[cell_names[c_idx]].values[0][0][1][0][1:] == region:
                spike_times = ephys_data[cell_names[c_idx]].values[0][0][0] - t_beginning
                spikes_cell = np.zeros(int(t_session))
                spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
                spikes_cell[spike_times.astype(int)] = 1

                # Initialize epoch_spikes array
                epoch_spikes = np.zeros(num_trials_train)
                epoch_spikes_test = np.zeros(num_trials_test)
                
                # Sum spikes during state 1 period for each trial
                for trial in range(num_trials_train):
                    epoch_spikes[trial] = np.sum(spikes_cell[state_1_t_train[trial]]) / state_1_t_train[trial].size

                for trial in range(num_trials_test):
                    epoch_spikes_test[trial] = np.sum(spikes_cell[state_1_t_test[trial]]) / state_1_t_test[trial].size
                    



                if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5:
                    X = np.array(range(num_trials_train)) + 1
                    X = sm.add_constant(X) 
                    X_t = np.array(range(num_trials_test)) + 1
                    X_t = sm.add_constant(X_t)
                    model = sm.GLM(epoch_spikes, X, family=sm.families.Poisson()).fit()
                    model_test = sm.GLM(epoch_spikes_test, X_t, family=sm.families.Poisson()).fit()
                    if model.pvalues[1] < 0.05:
                        epoch_spikes = epoch_spikes - model.fittedvalues
                    if model_test.pvalues[1] < 0.05:
                        epoch_spikes_test = epoch_spikes_test - model_test.fittedvalues
                        
                    # z-score firing rate
                    epoch_spikes = (epoch_spikes - np.mean(epoch_spikes)) / np.std(epoch_spikes)
                    epoch_spikes_test = (epoch_spikes_test - np.mean(epoch_spikes_test)) / np.std(epoch_spikes_test)

                    # save spikes, choices, and options
                    if len(spikes[s_idx]) == 0:
                        spikes[s_idx] = epoch_spikes                                # initialize 2d array
                    else:
                        spikes[s_idx] = np.vstack((spikes[s_idx], epoch_spikes))    # add new rows

                    if len(spikes_test[s_idx]) == 0:
                        spikes_test[s_idx] = epoch_spikes_test                                # initialize 2d array
                    else:
                        spikes_test[s_idx] = np.vstack((spikes_test[s_idx], epoch_spikes_test))    # add new rows

    
        
    #resize min_options to the number of cells in the region
    subjects_with_cells = min_options > 0
    # Filter all lists to only include subjects with cells in the region
    spikes = [spikes[s_idx] for s_idx in range(n_subjs) if subjects_with_cells[s_idx]]
    spikes_test = [spikes_test[s_idx] for s_idx in range(n_subjs) if subjects_with_cells[s_idx]]   

    min_options = min_options[subjects_with_cells]
    options = [options[s_idx] for s_idx in range(len(options)) if subjects_with_cells[s_idx]]
    options_test = [options_test[s_idx] for s_idx in range(len(options_test)) if subjects_with_cells[s_idx]]
    n_subjs = len(min_options)

    


    # initialize subsampling and cross-validation parameters
    n_folds = 5
    n_perms = 1000
    n_subsamp = int(np.min(min_options))
    print(n_subsamp)
    n_subsamp_test = np.concatenate((np.array([0]), np.min(min_options_test, axis=0).astype(int))) #.astype(int)  # [min_opt1, min_opt2, min_opt3, min_opt4]
    print(n_subsamp_test)
    #print(f"Test subsampling per option: {n_subsamp_test}")
    
    # Calculate minimum test trials across subjects (for balancing)
    
    choices = np.concatenate([np.ones(n_subsamp), np.ones(n_subsamp)*2, np.ones(n_subsamp)*3, np.ones(n_subsamp)*4])
    choices_test = np.concatenate([np.ones(n_subsamp_test[1]), np.ones(n_subsamp_test[2])*2, np.ones(n_subsamp_test[3])*3, np.ones(n_subsamp_test[4])*4])

    accuracies = np.zeros(n_perms)
    within_class_acc = np.zeros((n_perms, 4))  # Changed from 2 to 4 for 4 options


    for p_idx in range(n_perms):
        perm_spikes = []
        perm_spikes_test = []

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
            
            if spikes_test[s_idx].size > 0:  # only run if there are any cells for that subject
                # Ensure spikes_test[s_idx] is 2D
                if spikes_test[s_idx].ndim == 1:
                    spikes_test[s_idx] = spikes_test[s_idx].reshape(1, -1)
                
                for cI in range(spikes_test[s_idx].shape[0]):
                    perm_options = np.empty(np.sum(n_subsamp_test))
                    for o_idx in range(4):
                        start_idx = np.sum(n_subsamp_test[:o_idx+1])  # Sum up to current option
                        end_idx = np.sum(n_subsamp_test[:o_idx+2])    # Sum up to next option
                        o_tmp = np.where(options_test[s_idx] == o_idx+1)[0]
                        o_tmp = np.random.permutation(o_tmp)[:n_subsamp_test[o_idx+1]]
                        perm_options[start_idx:end_idx] = o_tmp
                    perm_options = perm_options.astype(int)
                
                    if len(perm_spikes_test) == 0:
                        perm_spikes_test = spikes_test[s_idx][cI, perm_options].reshape(-1, 1)
                    else:
                        included_spikes = spikes_test[s_idx][cI, perm_options].reshape(-1, 1)
                        perm_spikes_test = np.hstack((perm_spikes_test, included_spikes))


        X_train, y_train = perm_spikes, choices

        X_test, y_test = perm_spikes_test, choices_test

        clf = LinearSVC(penalty='l2',C=1,max_iter=10000)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        accuracies[p_idx] = accuracy_score(y_test, y_pred)

        cm = confusion_matrix(y_test, y_pred)
        within_class_acc[p_idx,:] = cm.diagonal() / cm.sum(axis=1)


    print(region + ' overall accuracy = ' + str(np.mean(accuracies)))
    print(region + ' within-class accuracies = ' + str(np.mean(within_class_acc, axis=0)))
    return accuracies, within_class_acc

if __name__ == "__main__":
    regions = ["ACC", "A", "H", "OF", "PT", "CM", "SMA", "PRV", "IFG"]
    context_combos = [
        (1, 2, 3),  # train 1 & 2, test 3
        (1, 3, 2),  # train 1 & 3, test 2
        (2, 3, 1),  # train 2 & 3, test 1
    ]

    summary_records = []

    for option_1, option_2, test_option in context_combos:
        ctx_label = f"{option_1}&{option_2}->{test_option}"
        print(f"\nRunning context {ctx_label}")
        for region in regions:
            accs, within_acc = option_ccgp_mcsd(region=region,
                                                option_1=option_1,
                                                option_2=option_2,
                                                test_option=test_option)
            summary_records.append({
                "Region": region,
                "Context": ctx_label,
                "Accuracy": np.mean(accs),
                "WithinClassAcc": np.mean(within_acc)
            })

    # build summary table
    summary_df = pd.DataFrame(summary_records)
    accuracy_table = summary_df.pivot(index="Region", columns="Context", values="Accuracy")

    print("\n==============================")
    print("Accuracy by Region and Context")
    print("==============================")
    print(accuracy_table.to_string(float_format="{:.3f}".format))






