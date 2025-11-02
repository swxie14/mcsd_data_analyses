"""
created 25.4.22

code to see percentages of neurons that encode value terms
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from regression_values_subj_mcsd import regression_values_subj_mcsd
import utils as u

def regression_values_mcsd(group='behEphys', subjs=[], behavior_mdl=[], var_names=[], z_flag=0, plot_flag=0, coef_corr_plot_flag=0, coef_diff_plot_flag=0, pie_chart_plot_flag=1):

    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/') 

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)

    for s_idx in range(n_subjs):

        subj = subjs[s_idx]
        print('running subject ' + subj + ', ' + str(s_idx+1) + ' of ' + str(n_subjs))

        out = regression_values_subj_mcsd(subj=subj, group=group, behavior_mdl=behavior_mdl, var_names=var_names, z_flag=z_flag, plot_flag=0)
        
        if s_idx == 0:
            p_vals = out['p_vals']
            coefs = out['coefs']
            is_sig = out['is_sig']
            regions = out['regions']
        else:
            p_vals = np.concatenate((p_vals, out['p_vals']), axis=1)
            coefs = np.concatenate((coefs, out['coefs']), axis=1)
            is_sig = np.concatenate((is_sig, out['is_sig']), axis=1)
            regions = regions + out['regions']   


    return{
        'coefs': coefs,
        'p_vals': p_vals,
        'is_sig': is_sig,
        'cell_names': out['cell_names'],
        'epoch_names': epoch_names,
        'var_names': var_names,
        'regions': regions,
    }

if __name__ == "__main__":
    subjs=[]
    group='behEphys'
    behavior_mdl='det_hmbOforgetFixedEtaBetaExplorBias'
    var_names=['pChoice_1', 'Qc_1', 'Uc_1']
    z_flag=0
    plot_flag=1
    coef_corr_plot_flag=0      # Turn off coefficient correlation heatmaps
    coef_diff_plot_flag=0      # Turn off coefficient difference heatmaps  
    pie_chart_plot_flag=1      # Turn on pie charts

    result = regression_values_mcsd(group=group, subjs=subjs, behavior_mdl=behavior_mdl, var_names=[], z_flag=z_flag, plot_flag=plot_flag, coef_corr_plot_flag=coef_corr_plot_flag, coef_diff_plot_flag=coef_diff_plot_flag, pie_chart_plot_flag=pie_chart_plot_flag)