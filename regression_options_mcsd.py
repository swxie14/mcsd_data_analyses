"""
created 25.4.9

code to see if the same neurons are different options

"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os
from regression_options_subj_mcsd import regression_options_subj_mcsd

def regression_options_mcsd(group='behEphys', subjs=[], z_flag=0, plot_flag=0):

    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/')

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)

    for s_idx in range(n_subjs):

        subj = subjs[s_idx]
        print('running subject ' + subj + ', ' + str(s_idx+1) + ' of ' + str(n_subjs))

        out = regression_options_subj_mcsd(subj=subj, group=group, z_flag=z_flag, plot_flag=0)
        
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


    if plot_flag:
        num_epochs = len(out['epoch_names'])
        colors = plt.cm.cool(np.linspace(0, 1, num_epochs))
        var_names = out['var_names']
        num_options = len(var_names)
        epoch_names = out['epoch_names']

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
        
        plt.suptitle(f"all subjects: {group}")
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show(block=False)


        regions = [x[1:] for x in regions]
        region_list = np.unique(regions)
        num_regions = len(region_list)

        for r_idx in range(num_regions):
            region = region_list[r_idx]
            plt.figure(figsize=(12, 13))

            for e_idx in range(num_epochs):
                for o_idx in range(num_options):
                    plt.subplot(num_epochs, num_options, (e_idx) * num_options + o_idx + 1)

                    rc_idx = [i for i, text in enumerate(regions) if region == text]
                    p_val_tmp = p_vals[e_idx, rc_idx, o_idx]
                    p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp)]

                    if len(p_val_tmp) > 0:
                        num_sig = np.sum(p_val_tmp < 0.05)

                        # Create pie chart
                        plt.pie([len(p_val_tmp) - num_sig, num_sig], 
                                        colors=[[0.7, 0.7, 0.7, 1], colors[e_idx]], 
                                        autopct='%1.1f%%')

                        plt.title(f"{epoch_names[e_idx]} - {var_names[o_idx]}")
                        plt.axis('equal')

            plt.suptitle(f"all subjects: {group}, region: {region}")
            plt.tight_layout()
            plt.subplots_adjust(top=0.95)
            plt.show(block=False)


        # plot pie chart of ratio of cells with different numbers of significant regression coefs for options
        plt.figure(figsize=(10, 5))
        for e_idx in range(num_epochs):
            plt.subplot(1, num_epochs, e_idx+1)
            rc_idx = [i for i, text in enumerate(regions) if 'A' == text]
            p_val_tmp = p_vals[e_idx, rc_idx,:]
            p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp[:,0]),:]
           # p_val_tmp = p_vals[e_idx, ~np.isnan(p_vals[e_idx, :,0]),:]
            sig_options = np.sum(p_val_tmp < 0.05, axis=1)
            sig_options = np.histogram(sig_options, 5)[0]
            plt.pie(sig_options, 
                        colors=np.vstack((np.array((0.7,0.70,0.7,0.1)), colors)), 
                        autopct='%1.1f%%',
                        labels=[str(i) for i in range(len(sig_options))])
            plt.title(f"{epoch_names[e_idx]}")
            plt.axis('equal')

        plt.suptitle(f"all subjects: {group}")
        plt.tight_layout(rect=[0, 0, 1, 0.90])
        plt.show(block=False)

        # plot heatmap of correlation between regression coefs for each option
        option_names = [s[-2:] for s in var_names]
        plt.figure(figsize=(13, 3))
        for e_idx in range(num_epochs):
            plt.subplot(1, num_epochs, e_idx+1)
            coefs_tmp = coefs[e_idx, ~np.isnan(coefs[e_idx, :,0]),:]
            coefs_tmp = coefs_tmp[~np.isnan(coefs_tmp[:,0]),:]

            coef_matrix = np.zeros((num_options, num_options))
            p_matrix = np.zeros((num_options, num_options))
            for i in range(num_options):
                for j in range(num_options):
                    r, p = pearsonr(coefs_tmp[:,i], coefs_tmp[:,j])
                    coef_matrix[i,j] = r
                    p_matrix[i,j] = p
            matrix_mask = np.tril(coef_matrix)
            matrix_mask = np.where(np.tril(np.ones_like(coef_matrix)), coef_matrix, np.nan)

            plt.imshow(matrix_mask, aspect='auto', cmap='cool', interpolation='nearest')
            plt.title(f"{epoch_names[e_idx]}")
            plt.colorbar(label='corr ceoff')
            plt.xticks(range(num_options), option_names, rotation=45)
            plt.yticks(range(num_options), option_names)

            # for i in range(num_options):
            #     for j in range(num_options):
            #         if p_matrix[i, j] < 0.05:
            #             plt.text(j, i, f"{p_matrix[i, j]:.2f}", ha='center', va='center', color='black', fontsize=8)
        
        plt.suptitle(f"all subjects: {group}")
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
            

        # plot relationship between different option encoding by region
        for r_idx in range(num_regions):
            plt.figure(figsize=(10, 5))
            region = region_list[r_idx]
            for e_idx in range(num_epochs):
                plt.subplot(1, num_epochs, e_idx+1)
                rc_idx = [i for i, text in enumerate(regions) if region == text]
                p_val_tmp = p_vals[e_idx, rc_idx,:]
                p_val_tmp = p_val_tmp[~np.isnan(p_val_tmp[:,0]), :]
                sig_options = np.sum(p_val_tmp < 0.05, axis=1)
                sig_options = np.histogram(sig_options, 5)[0]
                plt.pie(sig_options, 
                            colors=np.vstack((np.array((0.7,0.70,0.7,0.1)), colors)), 
                            autopct='%1.1f%%',
                            labels=[str(i) for i in range(len(sig_options))])
                plt.title(f"{epoch_names[e_idx]}")
                plt.axis('equal')

            plt.suptitle(f"all subjects: {group}, region: {region}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show(block=False)

        # plot heatmaps of correlation between regression coefs for each option by region
        for r_idx in range(num_regions):
            plt.figure(figsize=(13, 3))
            region = region_list[r_idx]
            for e_idx in range(num_epochs):
                plt.subplot(1, num_epochs, e_idx+1)
                rc_idx = [i for i, text in enumerate(regions) if region == text]
                coefs_tmp = coefs[e_idx, rc_idx,:]
                coefs_tmp = coefs_tmp[~np.isnan(coefs_tmp[:,0]), :]
                coef_matrix = np.corrcoef(coefs_tmp, rowvar=False)
                matrix_mask = np.tril(coef_matrix)
                matrix_mask = np.where(np.tril(np.ones_like(coef_matrix)), coef_matrix, np.nan)

                plt.imshow(matrix_mask, aspect='auto', cmap='cool', interpolation='nearest')
                plt.title(f"{epoch_names[e_idx]}")
                plt.colorbar(label='corr ceoff')
                plt.xticks(range(num_options), option_names, rotation=45)
                plt.yticks(range(num_options), option_names)
            
            plt.suptitle(f"all subjects: {group}, region: {region}")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show(block=False)

        # plot violin plot with connecting lines for correlation coeffs for single-option encoding neurons
        option_inds = np.array([[0,1,2,3], [1,0,3,2], [2,3,0,1], [3,2,1,0]])    #same first action, same second action, opposite actions
        for e_idx in range(num_epochs):
            plt.figure(figsize=(8, 4))
            for r_idx in range(num_regions):
                plt.subplot(2, int(np.ceil(num_regions/2)), r_idx+1)
                region = region_list[r_idx]
                rc_idx = [i for i, text in enumerate(regions) if region == text]
                coefs_tmp = coefs[e_idx, rc_idx,:]
                coefs_tmp = coefs_tmp[~np.isnan(coefs_tmp[:,0]), :]
                is_sig_tmp = is_sig[e_idx, rc_idx,:]
                is_sig_tmp = is_sig_tmp[~np.isnan(is_sig_tmp[:,0]), :]
                single_option_idx = np.sum(is_sig_tmp, axis=1) == 1

                if single_option_idx.any():
                    single_option_idx = np.where(single_option_idx)[0]
                    is_sig_tmp = is_sig_tmp[single_option_idx, :]
                    coefs_tmp = coefs_tmp[single_option_idx, :]

                    for c_idx in range(len(single_option_idx)):
                        o_idx = np.where(is_sig_tmp[c_idx, :] == 1)[0]
                        coefs_tmp[c_idx] = coefs_tmp[c_idx, option_inds[o_idx]]


                    plt.violinplot(coefs_tmp, showmedians=True, showextrema=False, quantiles=[[0.25, 0.75]] * 4)
                    for c_idx in range(len(single_option_idx)):
                        plt.plot(range(1, num_options + 1), coefs_tmp[c_idx], linestyle='-', color='gray')

                    plt.title(f"{region}")
                    plt.xticks(range(1, num_options + 1), ['option', 'same 1st', 'same 2nd', 'oppo.'], rotation=45)
                    plt.ylabel('Coefficient Value')
            plt.suptitle(f"epoch: {epoch_names[e_idx]}")

    return{
        'coefs': coefs,
        'p_vals': p_vals,
        'cell_names': out['cell_names'],
        'epoch_names': epoch_names,
        'var_names': var_names,
        'regions': regions,
    }

if __name__ == "__main__":
    subjs=[]
    group='behEphys'
    z_flag=0
    plot_flag=1

    result = regression_options_mcsd(group=group, subjs=subjs, z_flag=z_flag, plot_flag=plot_flag)


