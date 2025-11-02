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
from support.beh import Behavior
from support.ephys import Spikes

def regression_values_subj_mcsd(subj=None, group='behEphys', behavior_mdl='det_hmbOforgetFixedEtaBetaExplorBias', var_names=None, z_flag=0, plot_flag=0):
    
    if subj is None:
        raise ValueError("subject must be provided")
                            
    subj_beh = Behavior(subj)
    subj_ephys = Spikes(subj)
    mdl_data = u.load_behavior_mdl_mcsd(behavior_mdl=behavior_mdl)

    # Handle potentially empty tGoalOn arrays
    goal_time = subj_beh.get_feature_values('tGoalOn')
    goal_time = goal_time[-1]

    if isinstance(goal_time, np.ndarray) and goal_time.size == 0:
        goal_time = 0
    elif isinstance(goal_time, np.ndarray):
        goal_time = goal_time[0]
        
    if goal_time > 0:      # find total session time
        session = np.max([subj_beh.get_feature_values('tGoalOn')[-1], goal_time]) 
    else:
        session = subj_beh.get_feature_values('tGoalOn')[-1]
    subj_beh.t_session = session + 2000 - subj_beh.t_beginning

    # get number of trials and index behavior model regressor data
    subj_idxs = [i for i, s in enumerate(mdl_data['subjId']) if subj[-3:] in s]
    mdl_data = mdl_data.iloc[subj_idxs, :]


    # get value term variables and their names if not given as input
    df = pd.DataFrame()
    if not var_names:
        var_indxs = [i for i, s in enumerate(mdl_data.columns) if '_' in s]
        num_vars = len(var_indxs)
        var_names = [None] * num_vars
        for i, v_idx in enumerate(var_indxs):
            var_names[i] = mdl_data.columns[v_idx]
            var_values = mdl_data[v_idx].values
            var_values = (var_values - np.mean(var_values)) / np.std(var_values)
            df[v_idx] = var_values
    else:
        num_vars = len(var_names)
        for v_idx in var_indxs:
            var_values = mdl_data[v_idx].values
            var_values = (var_values - np.mean(var_values)) / np.std(var_values)
            df[v_idx] = var_values

    state_1_map = subj_beh.construct_timestep_index_maps('tState1On', 1000, 200)
    #resp_1_map = subj_beh.construct_timestep_index_maps('tResp1', 1000, -500)
    state_2_map = subj_beh.construct_timestep_index_maps('tState2On', 1000, 200)
    #resp_2_map = subj_beh.construct_timestep_index_maps('tResp2', 1000, -500)

    # all_maps = np.array([state_1_map, resp_1_map, state_2_map, resp_2_map])
    # epoch_names = ['state_1', 'resp_1', 'state_2', 'resp_2']
    all_maps = np.array([state_1_map, state_2_map])
    epoch_names = ['state_1', 'state_2']
    num_epochs = all_maps.shape[0]

    # initialize output data structures
    coefs = np.full((num_epochs, subj_ephys.num_cells, len(var_names)), np.nan)
    p_vals = np.full((num_epochs, subj_ephys.num_cells, len(var_names)), np.nan)
    is_sig = np.full((num_epochs, subj_ephys.num_cells, len(var_names)), np.nan)
    regions = [None] * subj_ephys.num_cells

    # run through cells
    for i, cell_name in enumerate(subj_ephys.cell_names):
        for e_idx in range(num_epochs):
            epoch_spikes = subj_ephys.z_score_cell(all_maps[e_idx], cell_name)
            df['spikes'] = epoch_spikes

            for v_idx, var_name in enumerate(var_names):
                formula = f"spikes ~ {var_name}"
                try:
                    if z_flag:
                        mdl = smf.ols(formula, data=df).fit()
                    else:
                        mdl = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
                    coefs[e_idx, i, v_idx] = mdl.params.values[1]
                    p_vals[e_idx, i, v_idx] = mdl.pvalues.values[1]
                    is_sig[e_idx, i, v_idx] = int(mdl.pvalues.values[1] < 0.05)
                except Exception as e:
                    print(f"Error fitting model for cell {cell_name}, epoch {epoch_names[e_idx]}: {str(e)}")

        region_name = subj_ephys.get_region(cell_name)
        # Standardize region names to handle inconsistencies
        regions[i] = 'LOF' if region_name == 'LOFC' else region_name

    return {
        'coefs': coefs,
        'p_vals': p_vals,
        'is_sig': is_sig,
        'cell_names': subj_ephys.cell_names,
        'epoch_names': epoch_names,
        'var_names': var_names,
        'regions': regions,
    }
            



# def plot():
#     if plot_flag:
#         frac_sig = np.zeros((num_vars, num_epochs))
#         for e_idx in range(num_epochs):            
#             for v_idx in range(num_vars):                
#                 p_val_tmp = p_vals[e_idx,:, v_idx]
#                 p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp)]
#                 frac_sig[v_idx,e_idx] = round((np.sum(p_val_tmp < 0.05) / len(p_val_tmp))*100, 1)

#         cell_text = []
#         for row_label, row in zip(var_names, frac_sig):
#             cell_text.append(list(row))

#         fig, ax = plt.subplots(figsize=(8, 4))
#         ax.axis('off')
#         table = ax.table(
#             cellText=cell_text,
#             rowLabels=var_names,
#             colLabels=epoch_names,
#             loc='center'
#         )
#         table.auto_set_font_size(False)
#         table.set_fontsize(12)
#         table.scale(1.2, 1.2)
#         plt.tight_layout()
        
#         # Save the figure instead of showing it interactively
#         plt.savefig(f'{subj}_neural_encoding_results.png', dpi=300, bbox_inches='tight')
#         print(f"Results table saved as '{subj}_neural_encoding_results.png'")
#         plt.close()  # Close the figure to free memory