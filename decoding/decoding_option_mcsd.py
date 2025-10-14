from support.beh import Behavior
from pathlib import Path
import numpy as np
from support.ephys import Spikes
from support.linear_decoder import run_linear_decoder

def decode_option_mcsd(group='behEphys',region='ACC',subjs=None): 

    if subjs is None:
        curr_dir = Path.cwd()
        data_path = curr_dir / group
        subjs = [subj.name for subj in data_path.iterdir() if 'subj' in subj.name]

    n_subjs = len(subjs)
    spikes = [np.array([], dtype=int) for _ in range(n_subjs)]
    options = [None] * n_subjs
    min_options = np.zeros(n_subjs)

    for i, subj in enumerate(subjs):

        subj_beh = Behavior(subj)
        subj_ephys = Spikes(subj, subj_beh.t_beginning, subj_beh.t_session, subj_beh.num_trials)

        choice_1 = subj_beh.get_feature_values('choice1')
        choice_2 = subj_beh.get_feature_values('choice2')
        
        options[i]  = choice_1 -1 + choice_1 - 1 + choice_2 - 1 + 1
        num_cells_region = np.sum(np.array([subj_ephys.get_region(cell_name, full_region=False) == region for cell_name in subj_ephys.cell_names]))
        if num_cells_region > 0:
            min_options[i] = np.min(np.histogram(options[i], bins=4)[0])
        

        #state_map = subj_beh.construct_timestep_index_maps('tFix', 1000, 200)
        state_map = subj_beh.construct_nonfixed_timestep_index_maps('tFix', 'tState1On')

        for cell_name in subj_ephys.cell_names:
            if subj_ephys.get_region(cell_name, False) == region:
                cell_spikes = subj_ephys.z_score_cell(state_map, cell_name, fixed=False) # alter fixed depending on the state_map
                if cell_spikes is not None:
                    spikes[i] = cell_spikes if len(spikes[i]) == 0 else np.vstack((spikes[i], cell_spikes))
            
    

    subjects_with_cells = np.where(min_options > 0)[0]
    n_subsamp = int(np.min(min_options[subjects_with_cells]))

    n_perms = 100
    n_folds = 5
    n_labels = 4

    accuracy, within_class_acc = run_linear_decoder(n_perms, n_folds, subjects_with_cells, spikes, n_subsamp, n_labels, options)


    print(f"Total Accuracy {region}: {np.mean(accuracy)}")
    print(f"Total Within Class Accuracy {region}: {np.mean(within_class_acc, axis=0)}")


    return accuracy, within_class_acc
        
                





