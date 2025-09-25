"""
created 25.4.22

code to aggregate goal context encoding data across subjects
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from goal_context_subj import goal_context_subj
import utils as u

def goal_context_mcsd(group='behEphys', subjs=[], behavior_mdl=[], var_names=[], z_flag=0, plot_flag=0):

    # set home directory
    data_dir = ('/Users/samuelxie/Desktop/oLab/' + group + '/') 

    if not subjs:
        subjs = os.listdir(data_dir)                        # get list of subject folders
        subjs = [subj for subj in subjs if 'subj' in subj]  # find folders that have 'subj' in the name

    n_subjs = len(subjs)

    for s_idx in range(n_subjs):

        subj = subjs[s_idx]
        print('running subject ' + subj + ', ' + str(s_idx+1) + ' of ' + str(n_subjs))

        out = goal_context_subj(subj=subj, group=group, behavior_mdl=behavior_mdl, var_names=var_names, z_flag=z_flag, plot_flag=0)
        
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
        regions = [x[1:] for x in regions]
        region_list = np.unique(regions)
        num_regions = len(region_list)

        var_names = out['var_names']
        num_vars = len(var_names)
        num_epochs = len(out['epoch_names'])

        # Create separate tables for currSubgoal and goalAmt
        print("\n" + "="*60)
        print("GOAL CONTEXT ENCODING PERCENTAGES BY REGION AND EPOCH")
        print("="*60)
        
        # Calculate currSubgoal encoding percentages
        print("\nCURRSUBGOAL ENCODING:")
        print("-" * 40)
        currSubgoal_frac_sig = np.zeros((num_regions, num_epochs))
        
        for r_idx in range(num_regions):
            region = region_list[r_idx]
            region_idxs = [i for i, text in enumerate(regions) if region == text]
            
            if len(region_idxs) == 0:
                continue
                
            print(f"Region: {region}")
            for e_idx in range(num_epochs):
                # Get currSubgoal significance for this region and epoch (index 1 in 4th dimension)
                is_sig_currSubgoal = is_sig[e_idx, region_idxs, :, 1]  # All variables, currSubgoal coefficient
                is_sig_currSubgoal = is_sig_currSubgoal[~np.isnan(is_sig_currSubgoal)]
                
                if len(is_sig_currSubgoal) > 0:
                    currSubgoal_frac_sig[r_idx, e_idx] = round((np.sum(is_sig_currSubgoal == 1) / len(is_sig_currSubgoal)) * 100, 1)
                    print(f"  {out['epoch_names'][e_idx]}: {currSubgoal_frac_sig[r_idx, e_idx]}%")
                else:
                    currSubgoal_frac_sig[r_idx, e_idx] = 0
                    print(f"  {out['epoch_names'][e_idx]}: No data")

        # Calculate goalAmt encoding percentages
        print("\nGOALAMT ENCODING:")
        print("-" * 40)
        goalAmt_frac_sig = np.zeros((num_regions, num_epochs))
        
        for r_idx in range(num_regions):
            region = region_list[r_idx]
            region_idxs = [i for i, text in enumerate(regions) if region == text]
            
            if len(region_idxs) == 0:
                continue
                
            print(f"Region: {region}")
            for e_idx in range(num_epochs):
                # Get goalAmt significance for this region and epoch (index 2 in 4th dimension)
                is_sig_goalAmt = is_sig[e_idx, region_idxs, :, 2]  # All variables, goalAmt coefficient
                is_sig_goalAmt = is_sig_goalAmt[~np.isnan(is_sig_goalAmt)]
                
                if len(is_sig_goalAmt) > 0:
                    goalAmt_frac_sig[r_idx, e_idx] = round((np.sum(is_sig_goalAmt == 1) / len(is_sig_goalAmt)) * 100, 1)
                    print(f"  {out['epoch_names'][e_idx]}: {goalAmt_frac_sig[r_idx, e_idx]}%")
                else:
                    goalAmt_frac_sig[r_idx, e_idx] = 0
                    print(f"  {out['epoch_names'][e_idx]}: No data")

        # Create currSubgoal table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Prepare data for table
        cell_text_currSubgoal = []
        for r_idx in range(num_regions):
            row_data = []
            for e_idx in range(num_epochs):
                row_data.append(f"{currSubgoal_frac_sig[r_idx, e_idx]:.1f}%")
            cell_text_currSubgoal.append(row_data)
        
        table_currSubgoal = ax.table(
            cellText=cell_text_currSubgoal,
            rowLabels=region_list,
            colLabels=out['epoch_names'],
            loc='center'
        )
        table_currSubgoal.auto_set_font_size(False)
        table_currSubgoal.set_fontsize(12)
        table_currSubgoal.scale(1.2, 1.5)
        plt.suptitle(f"currSubgoal Encoding Percentages - All Subjects: {group}", fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Save currSubgoal table
        plt.savefig('currSubgoal_encoding_percentages.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\ncurrSubgoal encoding table saved as 'currSubgoal_encoding_percentages.png'")
        plt.close()

        # Create goalAmt table
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.axis('off')
        
        # Prepare data for table
        cell_text_goalAmt = []
        for r_idx in range(num_regions):
            row_data = []
            for e_idx in range(num_epochs):
                row_data.append(f"{goalAmt_frac_sig[r_idx, e_idx]:.1f}%")
            cell_text_goalAmt.append(row_data)
        
        table_goalAmt = ax.table(
            cellText=cell_text_goalAmt,
            rowLabels=region_list,
            colLabels=out['epoch_names'],
            loc='center'
        )
        table_goalAmt.auto_set_font_size(False)
        table_goalAmt.set_fontsize(12)
        table_goalAmt.scale(1.2, 1.5)
        plt.suptitle(f"goalAmt Encoding Percentages - All Subjects: {group}", fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Save goalAmt table
        plt.savefig('goalAmt_encoding_percentages.png', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"goalAmt encoding table saved as 'goalAmt_encoding_percentages.png'")
        plt.close()

    return{
        'coefs': coefs,
        'p_vals': p_vals,
        'is_sig': is_sig,
        'cell_names': out['cell_names'],
        'epoch_names': out['epoch_names'],
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

    result = goal_context_mcsd(group=group, subjs=subjs, behavior_mdl=behavior_mdl, var_names=[], z_flag=z_flag, plot_flag=plot_flag)
