"""
created 25.4.10

code to regress firing rates on options in states 1 and 2

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import statsmodels.api as sm
import statsmodels.formula.api as smf
import utils as u



def is_nonempty(val):
    if isinstance(val, np.ndarray):
        return val.size > 0
    return val is not None

def regression_options_subj_mcsd(subj=[], group='behEphys', z_flag=0, plot_flag=0):
    if not subj:
        raise ValueError("subject must be provided")
                            
    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subj, rmv_flag=1)
    #print(ephys_data.head())
    # get behavior and task data
    t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1
    # Safely handle empty tGoalOn
    if len(beh_data['tGoalOn']) > 0 and is_nonempty(beh_data['tGoalOn'].iloc[-1]) and beh_data['tGoalOn'].iloc[-1] > 0:  # find total session time
        t_session = np.max([beh_data['tOutOn'].iloc[-1], beh_data['tGoalOn'].iloc[-1]]) 
    elif len(beh_data['tOutOn']) > 0:
        t_session = beh_data['tOutOn'].iloc[-1] 
    else:
        raise ValueError("No valid tGoalOn or tOutOn data in beh_data for subject: {}".format(subj))
    t_session = t_session + 2000 - t_beginning

    num_trials = beh_data.shape[0]
    choice_1 = beh_data['choice1'].values
    choice_2 = beh_data['choice2'].values
    option = choice_1 - 1 + choice_1 - 1 + choice_2 - 1 + 1
    optionLL = option == 1 
    optionLR = option == 2
    optionRL = option == 3
    optionRR = option == 4
    num_options = 4

    all_vars = np.array([optionLL, optionLR, optionRL, optionRR])
    var_names = np.array(['optionLL', 'optionLR', 'optionRL', 'optionRR'])
    var_idx =[[0,1,3], [0,2,3]] #

    # make timesamp index mats for relevant task epochs
    x_state_1 = np.arange(1001)
    state_1_map = np.tile(x_state_1, (num_trials, 1))
    t_tmp = beh_data['tState1On'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_1_map.shape[1]))
    state_1_map = state_1_map + t_tmp
    state_1_map = state_1_map.astype(int)

    x_resp_1 = np.arange(-500, 501)
    resp_1_map = np.tile(x_resp_1, (num_trials, 1))
    t_tmp = beh_data['tResp1'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, resp_1_map.shape[1]))
    resp_1_map = resp_1_map + t_tmp
    resp_1_map = resp_1_map.astype(int)

    x_state_2 = np.arange(1001)
    state_2_map = np.tile(x_state_2, (num_trials, 1))
    t_tmp = beh_data['tState2On'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_2_map.shape[1]))
    state_2_map = state_2_map + t_tmp
    state_2_map = state_2_map.astype(int)

    x_resp_2 = np.arange(-500, 501)
    resp_2_map = np.tile(x_resp_2, (num_trials, 1))
    t_tmp = beh_data['tResp2'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, resp_2_map.shape[1]))
    resp_2_map = resp_2_map + t_tmp
    resp_2_map = resp_2_map.astype(int)

    # put maps in single array
    all_maps = np.array([state_1_map, resp_1_map, state_2_map, resp_2_map])
    epoch_names = ['state_1', 'resp_1', 'state_2', 'resp_2']
    num_epochs = all_maps.shape[0]

    cell_names = ephys_data.columns
    num_cells = cell_names.shape[0]

    # initialize output data structures
    coefs = np.full((num_epochs, num_cells, num_options), np.nan)
    p_vals = np.full((num_epochs, num_cells, num_options), np.nan)
    is_sig = np.full((num_epochs, num_cells, num_options), np.nan)
    regions = [None] * num_cells # list of regions of the brain

    # run through cells
    for c_idx in range(num_cells):
        spike_times = ephys_data[cell_names[c_idx]].values
        spike_times = spike_times[0][0][0] - t_beginning
        spikes = np.zeros(int(t_session))
        spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
        spikes[spike_times.astype(int)] = 1

        # run through task epochs
        for e_idx in range(num_epochs):
            epoch_spikes = np.sum(spikes[all_maps[e_idx, :, :]], axis=1) # sum across trials; each trial is a row in all_maps
            # use Poisson for sparser data and use z-score for high firing rate neurons
            if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5: # if less than 50% of trials are 0s, then z-score because high firing rate
                if z_flag:
                    epoch_spikes = (epoch_spikes - np.mean(epoch_spikes)) / np.std(epoch_spikes)
                
                # create DataFrame for regression
                df = pd.DataFrame()
                for vI, var_name in enumerate(var_names):
                    df[var_name] = all_vars[vI,:]
                df['spikes'] = epoch_spikes

                for glm_idx in range(2):
                    var_names_tmp = var_names[var_idx[glm_idx]]
                    #reference
                    #var_names = np.array(['optionLL', 'optionLR', 'optionRL', 'optionRR'])
                    #var_idx =[[0,1,3], [0,2,3]] 

                    # fit model
                    formula = f"spikes ~ {' + '.join(var_names_tmp)}"
                    try:
                        if z_flag:
                            mdl = smf.ols(formula, data=df).fit()
                        else:
                            mdl = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                        
                        # Store results
                        mdl_var_names = mdl.params.index
                        if glm_idx == 0:
                            for v_idx, var_name in enumerate(var_names_tmp[0:2]): # 2 is exclusive
                                vn_idx = [i for i, text in enumerate(mdl_var_names) if var_name in text]
                                #e_idx represents different states (state 1, resp 1, state 2, resp 2)
                                #c_idx represents different parts of the brain
                                #v_idx represents different options (LL, LR, RL, RR)
                                coefs[e_idx, c_idx, v_idx] = mdl.params.values[vn_idx][0]
                                p_vals[e_idx, c_idx, v_idx] = mdl.pvalues.values[vn_idx][0]
                                is_sig[e_idx, c_idx, v_idx] = int(mdl.pvalues.values[vn_idx][0] < 0.05)
                        else:
                            for v_idx, var_name in enumerate(var_names_tmp[1:]):
                                vn_idx = [i for i, text in enumerate(mdl_var_names) if var_name in text]
                                coefs[e_idx, c_idx, v_idx+2] = mdl.params.values[vn_idx][0]
                                p_vals[e_idx, c_idx, v_idx+2] = mdl.pvalues.values[vn_idx][0]
                                is_sig[e_idx, c_idx, v_idx+2] = int(mdl.pvalues.values[vn_idx][0] < 0.05)

                    except Exception as e:
                        print(f"Error fitting model for cell {c_idx}, epoch {epoch_names[e_idx]}: {str(e)}")

        regions[c_idx] = ephys_data[cell_names[c_idx]].values[0][0][1][0]

    if plot_flag:
        colors = plt.cm.cool(np.linspace(0, 1, num_epochs))
        
        plt.figure(figsize=(12, 13))
        
        for e_idx in range(num_epochs):            
            for o_idx in range(num_options):
                plt.subplot(num_epochs, num_options, (e_idx) * num_options + o_idx + 1)
                
                p_val_tmp = p_vals[e_idx, :, o_idx]
                p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp)]
                
                num_sig = np.sum(p_val_tmp < 0.05)
                
                # Create pie chart
                plt.pie([len(p_val_tmp) - num_sig, num_sig], 
                                    colors=[[0.7, 0.7, 0.7, 1], colors[e_idx]], 
                                    autopct='%1.1f%%')
                
                plt.title(f"{epoch_names[e_idx]} - {var_names[o_idx]}")
                plt.axis('equal')
        
        plt.suptitle(f"subject: {subj}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    return{
        'coefs': coefs,
        'p_vals': p_vals,
        'is_sig': is_sig,
        'cell_names': cell_names,
        'epoch_names': epoch_names,
        'var_names': var_names,
        'regions': regions,
    }
            
if __name__ == "__main__":
    subj='subj_057'
    group='behEphys'
    z_flag= 0
    plot_flag= 1

    result = regression_options_subj_mcsd(subj=subj, group=group, z_flag=z_flag, plot_flag=plot_flag)