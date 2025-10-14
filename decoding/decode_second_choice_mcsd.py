from support.beh import Behavior
from pathlib import Path
import numpy as np
from support.ephys import Spikes
from support.linear_decoder import run_linear_decoder


def decode_second_choice_mcsd(group='behEphys', region='ACC', subjs=None):
    # find folders that have 'subj' in the name
    if subjs is None:
        current_dir = Path.cwd()
        dir = current_dir / group
        subjs = [subj.name for subj in dir.iterdir() if 'subj' in subj.name]
    print(subjs)
    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)] 

    choice_2 = [None] * n_subjs #Why is this not numpy?
    min_choices = np.zeros(n_subjs)


    for i, subj in enumerate(subjs):
        subj_beh = Behavior(subj)
        subj_ephys = Spikes(subj, subj_beh.t_beginning, subj_beh.t_session, subj_beh.num_trials)

        choice_2[i] = subj_beh.get_feature_values('choice2')


        num_cells_region = np.sum(np.array([subj_ephys.get_region(cell_name, False) == region for cell_name in subj_ephys.cell_names]))
        if num_cells_region > 0:
            min_choices[i] = np.min([np.sum(choice_2[i]==1), np.sum(choice_2[i]==2)])

        state_map = subj_beh.construct_timestep_index_maps('tState1On', 1000, 200)

        for cell_name in subj_ephys.cell_names:
            if subj_ephys.get_region(cell_name, False) == region:
                cell_spikes = subj_ephys.z_score_cell(state_map, cell_name, True)
                
                if cell_spikes is not None:
                    spikes[i] = cell_spikes if len(spikes[i])==0 else np.vstack((spikes[i], cell_spikes))
            
            
    #print(spikes)
    #remove subjects with no cells in region
    subjects_with_cells = np.where(min_choices > 0)[0]
    n_subsamp = int(np.min(min_choices[subjects_with_cells]))


    n_perms = 100
    n_folds = 5
    n_labels = 2

    accuracy, within_class_acc = run_linear_decoder(n_perms, n_folds, subjects_with_cells, spikes, n_subsamp, n_labels, choice_2)

    print(f"Total Accuracy {region}: {np.mean(accuracy)}")
    print(f"Total Within Class Accuracy {region}: {np.mean(within_class_acc)}")


    return accuracy, within_class_acc