"""
created 25.7.8

decoding option on two subgoals and testing on the third subgoal

"""

import numpy as np
from pathlib import Path
import pandas as pd

from support.beh import Behavior
from support.ephys import Spikes
from support.linear_decoder import build_perm_spikes


import statsmodels.api as sm
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix


def option_ccgp_mcsd(group='behEphys', region='ACC', subjs=None, option_1 = 1, option_2 = 2, test_option = 3):
    # set home directory
    
    if subjs is None:
        curr_dir = Path.cwd()
        data_path = curr_dir / group
        subjs = [subj.name for subj in data_path.iterdir() if 'subj' in subj.name]
   # print(subjs)


    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)]
    options = [None] * n_subjs #stores number of spikes for each patient in each cell
    min_options = np.zeros(n_subjs)

    spikes_test = [np.array([], dtype=int) for _ in range(n_subjs)] #stores number of spikes for each patient in each cell
    options_test = [None] * n_subjs
    min_options_test = np.zeros(n_subjs)



    for i, subj in enumerate(subjs):
        # load data

        subj_beh = Behavior(subj)

        choice_1 = subj_beh.get_feature_values('choice1')
        choice_2 = subj_beh.get_feature_values('choice2')

        subj_ephys = Spikes(subj, subj_beh.t_beginning, subj_beh.t_session, subj_beh.num_trials)

        currSubgoal = subj_beh.get_feature_values('currSubgoal')

        train_trials = np.where((currSubgoal == option_1) | (currSubgoal == option_2))[0]
        test_trials = np.where(currSubgoal == test_option)[0]


        options[i] = choice_1[train_trials] - 1 + choice_1[train_trials] - 1 + choice_2[train_trials] - 1 + 1
        options_test[i] = choice_1[test_trials] - 1 + choice_1[test_trials] - 1 + choice_2[test_trials] - 1 + 1

        num_cells_region = np.sum(np.array([subj_ephys.get_region(cell_name, full_region=False) == region for cell_name in subj_ephys.cell_names]))

        if num_cells_region > 0:
            min_options[i] = np.min(np.histogram(options[i], bins=4)[0])
            min_options_test[i] = np.min(np.histogram(options_test[i], bins=4)[0])
        #only use minimum of subjects with cells in the region
        

        # make timesamp index mats for first 1s of state 1
        state_train  = subj_beh.construct_nonfixed_timestep_index_maps('tFix', 'tState1On')
        state_test = subj_beh.construct_nonfixed_timestep_index_maps('tFix', 'tState1On')



        # for each subject, creates z-scored spike count across all trials for each cell
        for cell_name in subj_ephys.cell_names:
            if subj_ephys.get_region(cell_name, False)== region:
                cell_spikes = subj_ephys.z_score_cell(state_train, cell_name, fixed=False)
                cell_spikes_test = subj_ephys.z_score_cell(state_test, cell_name, fixed=False)

                if cell_spikes is not None and cell_spikes_test is not None:
                    spikes[i] = cell_spikes if len(spikes[i]) == 0 else np.vstack((spikes[i], cell_spikes))
                    spikes_test[i] = cell_spikes_test if len(spikes_test[i]) == 0 else np.vstack((spikes_test[i], cell_spikes_test))



    
        
    #resize min_options to the number of cells in the region
    subjects_with_cells = np.where(min_options > 0)[0]
    n_subsamp = int(np.min(min_options[subjects_with_cells]))
    print(n_subsamp)
    n_subsamp_test = int(np.min(min_options_test[subjects_with_cells]))
    print(n_subsamp_test)


    # initialize subsampling and cross-validation parameters
    n_perms = 100
    n_folds = 5 # n_folds must be greater than or equal to n_subsamp
    n_labels = 4

    choices = np.concatenate([np.ones(n_subsamp) * (i+1) for i in range(n_labels)])
    choices_test = np.concatenate([np.ones(n_subsamp_test) * (i+1) for i in range(n_labels)])

    accuracies = np.zeros(n_perms)
    within_class_acc = np.zeros((n_perms, n_labels))

    accuracies_test = np.zeros(n_perms)
    within_class_acc_test = np.zeros((n_perms, n_labels))

    for p_idx in range(n_perms):
        perm_spikes = None
        perm_spikes_test = None

        for subj_i in subjects_with_cells: 
            if spikes[subj_i].size > 0:
                if spikes[subj_i].ndim == 1:
                    spikes[subj_i] = spikes[subj_i].reshape(1, -1)
                    spikes_test[subj_i] = spikes_test[subj_i].reshape(1, -1)
                perm_spikes =  build_perm_spikes(n_subsamp, n_labels, options[subj_i], spikes[subj_i], perm_spikes, False)
                perm_spikes_test = build_perm_spikes(n_subsamp_test, n_labels, options_test[subj_i], spikes_test[subj_i], perm_spikes_test, False)


        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        perm_accuracies = np.zeros(n_folds)
        perm_within_class_acc = np.zeros((n_folds, n_labels))

        perm_accuracies_test = np.zeros(n_folds)
        perm_within_class_acc_test = np.zeros((n_folds, n_labels))

        for f_idx, (train_idx, test_idx) in enumerate(skf.split(perm_spikes, choices)):
            X_train, X_test = perm_spikes[train_idx, :], perm_spikes[test_idx, :]
            y_train, y_test = choices[train_idx], choices[test_idx]

            model = LinearSVC(penalty="l2", C=1, max_iter=10000)
            model = model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            perm_accuracies[f_idx] = accuracy_score(y_test, y_pred)
            
            cm = confusion_matrix(y_test, y_pred)
            perm_within_class_acc[f_idx, :] = cm.diagonal() / cm.sum(axis=1)


            # TESTING MODEL ON TEST_OPTION
            X_test, y_test = perm_spikes_test, choices_test
            y_pred = model.predict(X_test)
            
            perm_accuracies_test[f_idx] = accuracy_score(y_test, y_pred)

            cm = confusion_matrix(y_test, y_pred)
            perm_within_class_acc_test[f_idx,:] = cm.diagonal() / cm.sum(axis=1)

        accuracies[p_idx] = np.mean(perm_accuracies)
        within_class_acc[p_idx] = np.mean(perm_within_class_acc, axis=0)

        accuracies_test[p_idx] = np.mean(perm_accuracies_test)
        within_class_acc_test[p_idx] = np.mean(perm_within_class_acc_test, axis=0)

    print(region + ' overall accuracy on Train Data' + str(np.mean(accuracies)))
    print(region + ' within-class accuracies on Train Data' + str(np.mean(within_class_acc, axis=0)))

    print(region + ' overall accuracy on Test Data' + str(np.mean(accuracies_test)))
    print(region + ' within-class accuracies on Test Data' + str(np.mean(within_class_acc_test, axis=0)))
    
    return accuracies, within_class_acc







