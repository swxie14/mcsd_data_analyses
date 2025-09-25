"""
created 25.4.21

code to regress firing rates in different task epochs on latent variables from behavior models

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import utils as u
import seaborn as sns
from scipy.stats import pearsonr

def regression_values_subj_mcsd(subj=[], group='behEphys', behavior_mdl='det_hmbOforgetFixedEtaBetaExplorBias', var_names=[], z_flag=0, plot_flag=0):
    if not subj:
        raise ValueError("subject must be provided")
                            
    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    beh_data, ephys_data = u.load_ephys_data_mcsd(data_dir=data_dir, subj=subj, rmv_flag=1)
    mdl_data = u.load_behavior_mdl_mcsd(behavior_mdl=behavior_mdl)

    # get behavior and task data
    t_beginning = (beh_data['tFix'].iloc[0] - 1000) + 1
    
    # Handle potentially empty tGoalOn arrays
    goal_time = beh_data['tGoalOn'].iloc[-1]
    if isinstance(goal_time, np.ndarray) and goal_time.size == 0:
        goal_time = 0
    elif isinstance(goal_time, np.ndarray):
        goal_time = goal_time[0]
        
    if goal_time > 0:      # find total session time
        t_session = np.max([beh_data['tOutOn'].iloc[-1], goal_time]) 
    else:
        t_session = beh_data['tOutOn'].iloc[-1] 
    t_session = t_session + 2000 - t_beginning

    # get number of trials and index behavior model regressor data
    num_trials = beh_data.shape[0]
    subj_idxs = [i for i, s in enumerate(mdl_data['subjId']) if subj[-3:] in s]
    mdl_data = mdl_data.iloc[subj_idxs, :]


    # get value term variables and their names if not given as input
    if not var_names:
        var_indxs = [i for i, s in enumerate(mdl_data.columns) if '_' in s]
        num_vars = len(var_indxs)
        var_names = [None] * num_vars
        df = pd.DataFrame()
        for v_idx in range(num_vars):
            var_names[v_idx] = mdl_data.columns[var_indxs[v_idx]]
            var_values = mdl_data[var_names[v_idx]].values
            var_values = (var_values - np.mean(var_values)) / np.std(var_values)
            df[var_names[v_idx]] = var_values
    else:
        num_vars = len(var_names)
        df = pd.DataFrame()
        for v_idx in range(num_vars):
            var_values = mdl_data[var_names[v_idx]].values
            var_values = (var_values - np.mean(var_values)) / np.std(var_values)
            df[var_names[v_idx]] = var_values


    # make timestamp index mats for relevant task epochs
    x_state_1 = np.arange(200,1201)
    state_1_map = np.tile(x_state_1, (num_trials, 1))
    t_tmp = beh_data['tState1On'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_1_map.shape[1]))
    state_1_map = state_1_map + t_tmp
    state_1_map = state_1_map.astype(int)

    # x_resp_1 = np.arange(-500, 501)
    # resp_1_map = np.tile(x_resp_1, (num_trials, 1))
    # t_tmp = beh_data['tResp1'].values - t_beginning
    # t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, resp_1_map.shape[1]))
    # resp_1_map = resp_1_map + t_tmp
    # resp_1_map = resp_1_map.astype(int)

    x_state_2 = np.arange(200,1201)
    state_2_map = np.tile(x_state_2, (num_trials, 1))
    t_tmp = beh_data['tState2On'].values - t_beginning
    t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, state_2_map.shape[1]))
    state_2_map = state_2_map + t_tmp
    state_2_map = state_2_map.astype(int)

    # x_resp_2 = np.arange(-500, 501)
    # resp_2_map = np.tile(x_resp_2, (num_trials, 1))
    # t_tmp = beh_data['tResp2'].values - t_beginning
    # t_tmp = np.tile(t_tmp.reshape(-1, 1), (1, resp_2_map.shape[1]))
    # resp_2_map = resp_2_map + t_tmp
    # resp_2_map = resp_2_map.astype(int)


    # put maps in single array
    # all_maps = np.array([state_1_map, resp_1_map, state_2_map, resp_2_map])
    # epoch_names = ['state_1', 'resp_1', 'state_2', 'resp_2']
    all_maps = np.array([state_1_map, state_2_map])
    epoch_names = ['state_1', 'state_2']
    num_epochs = all_maps.shape[0]

    cell_names = ephys_data.columns
    num_cells = cell_names.shape[0]

    # initialize output data structures
    coefs = np.full((num_epochs, num_cells, len(var_names)), np.nan)
    p_vals = np.full((num_epochs, num_cells, len(var_names)), np.nan)
    is_sig = np.full((num_epochs, num_cells, len(var_names)), np.nan)
    regions = [None] * num_cells

    # run through cells
    for c_idx in range(num_cells):
        spike_times = ephys_data[cell_names[c_idx]].values
        spike_times = spike_times[0][0][0] - t_beginning
        spikes = np.zeros(int(t_session))
        spike_times = spike_times[np.logical_and(spike_times >= 0, spike_times < t_session)]
        spikes[spike_times.astype(int)] = 1

        # run through task epochs 
        for e_idx in range(num_epochs):
            epoch_spikes = np.sum(spikes[all_maps[e_idx, :, :]], axis=1)
            if np.sum(epoch_spikes == 0) / len(epoch_spikes) < 0.5:
                if z_flag:
                    epoch_spikes = (epoch_spikes - np.mean(epoch_spikes)) / np.std(epoch_spikes)
                df['spikes'] = epoch_spikes

                # fit model to each epoch x var
                for v_idx, var_name in enumerate(var_names):
                    formula = f"spikes ~ {var_name}"
                    try:
                        if z_flag:
                            mdl = smf.ols(formula, data=df).fit()
                        else:
                            mdl = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                        
                        # store results
                        coefs[e_idx, c_idx, v_idx] = mdl.params.values[1]
                        p_vals[e_idx, c_idx, v_idx] = mdl.pvalues.values[1]
                        is_sig[e_idx, c_idx, v_idx] = int(mdl.pvalues.values[1] < 0.05)
                    except Exception as e:
                        print(f"Error fitting model for cell {c_idx}, epoch {epoch_names[e_idx]}: {str(e)}")

        region_name = ephys_data[cell_names[c_idx]].values[0][0][1][0]
        
        # Standardize region names to handle inconsistencies
        if region_name == 'LOFC':
            region_name = 'LOF'  # Convert LOFC to LOF for consistency
        
        regions[c_idx] = region_name

    if plot_flag:
        frac_sig = np.zeros((num_vars, num_epochs))
        for e_idx in range(num_epochs):            
            for v_idx in range(num_vars):                
                p_val_tmp = p_vals[e_idx,:, v_idx]
                p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp)]
                frac_sig[v_idx,e_idx] = round((np.sum(p_val_tmp < 0.05) / len(p_val_tmp))*100, 1)

        cell_text = []
        for row_label, row in zip(var_names, frac_sig):
            cell_text.append(list(row))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        table = ax.table(
            cellText=cell_text,
            rowLabels=var_names,
            colLabels=epoch_names,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)
        plt.tight_layout()
        
        # Save the figure instead of showing it interactively
        plt.savefig(f'{subj}_neural_encoding_results.png', dpi=300, bbox_inches='tight')
        print(f"Results table saved as '{subj}_neural_encoding_results.png'")
        plt.close()  # Close the figure to free memory

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
    behavior_mdl='det_hmbOforgetFixedEtaBetaExplorBias'
    var_names=[]
    z_flag=0
    plot_flag=1

    result = regression_values_subj_mcsd(subj=subj, group=group, behavior_mdl=behavior_mdl, var_names=var_names, z_flag=z_flag, plot_flag=plot_flag)