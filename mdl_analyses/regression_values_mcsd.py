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

    if plot_flag:
        regions = [x[1:] for x in regions]
        region_list = np.unique(regions)
        num_regions = len(region_list)

        var_names = out['var_names']
        num_vars = len(var_names)
        num_epochs = len(out['epoch_names'])

        for r_idx in range(num_regions):
            frac_sig = np.zeros((num_vars, num_epochs))
            region = region_list[r_idx]
            region_idxs = [i for i, text in enumerate(regions) if region == text]
            for e_idx in range(num_epochs):            
                for v_idx in range(num_vars):                
                    is_sig_tmp = is_sig[e_idx,region_idxs, v_idx]
                    is_sig_tmp = is_sig_tmp[~np.isnan(is_sig_tmp)]
                    frac_sig[v_idx,e_idx] = round((np.sum(is_sig_tmp == 1) / len(is_sig_tmp))*100, 1)

            cell_text = []
            for row_label, row in zip(var_names, frac_sig):
                cell_text.append(list(row))

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.axis('off')
            table = ax.table(
                cellText=cell_text,
                rowLabels=var_names,
                colLabels=out['epoch_names'],
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 1.2)
            plt.suptitle(f"all subjects: {group}, region: {region}")
            plt.tight_layout()
            plt.show()        

        # plot heatmap of correlation between value term regressors
        mdl_data = u.load_behavior_mdl_mcsd(behavior_mdl=behavior_mdl)
        all_vars = np.full((num_vars, len(mdl_data)), np.nan)
        for v_idx in range(num_vars):
            all_vars[v_idx,:] = mdl_data[var_names[v_idx]].values
        
        coef_matrix = np.zeros((num_vars, num_vars))
        p_matrix = np.zeros((num_vars, num_vars))
        for i in range(num_vars):
            for j in range(num_vars):
                r, p = pearsonr(all_vars[:,i], all_vars[:,j])
                coef_matrix[i,j] = r
                p_matrix[i,j] = p
        matrix_mask = np.tril(coef_matrix)
        matrix_mask = np.where(np.tril(np.ones_like(coef_matrix)), coef_matrix, np.nan)

        plt.figure(figsize=(12, 12))
        plt.imshow(matrix_mask, aspect='auto', cmap='cool', interpolation='nearest')
        plt.title('correlation between value terms')
        plt.colorbar(label='corr coeff')
        plt.xticks(range(num_vars),var_names, rotation=45)
        plt.yticks(range(num_vars), var_names)
        plt.tight_layout()
        
        # Save the heatmap
        plt.savefig('behavioral_variables_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("Behavioral variables correlation heatmap saved as 'behavioral_variables_correlation_heatmap.png'")
        plt.close()

        # ------------------------------------------------------------------
        # Generate heatmaps of correlation between coefficient values
        # for each brain region and task state (state_1, state_2)
        # Only using coefficients that have significant regression coefficients
        # ------------------------------------------------------------------
        if coef_corr_plot_flag:
            heatmap_dir = 'heat maps'
            os.makedirs(heatmap_dir, exist_ok=True)

            epoch_names = out['epoch_names']
            for region in region_list:
                region_idxs = [i for i, text in enumerate(regions) if region == text]

                if len(region_idxs) == 0:
                    continue

                fig, axes = plt.subplots(1, num_epochs, figsize=(6 * num_epochs, 5))
                if num_epochs == 1:
                    axes = [axes]

                for e_idx in range(num_epochs):
                    coef_region_state = coefs[e_idx, region_idxs, :]
                    is_sig_region_state = is_sig[e_idx, region_idxs, :]

                    # Filter out rows with any NaN values
                    valid_mask = ~np.isnan(coef_region_state).any(axis=1)
                    coef_region_state = coef_region_state[valid_mask, :]
                    is_sig_region_state = is_sig_region_state[valid_mask, :]

                    # Filter to only include coefficients that are significant
                    # For each variable, only include cells where that variable has a significant coefficient
                    sig_filtered_coefs = []
                    for v_idx in range(num_vars):
                        sig_mask = is_sig_region_state[:, v_idx] == 1
                        if np.sum(sig_mask) > 0:
                            sig_filtered_coefs.append(coef_region_state[sig_mask, :])
                        else:
                            # If no significant cells for this variable, add empty array
                            sig_filtered_coefs.append(np.empty((0, num_vars)))

                    if len(sig_filtered_coefs) > 0 and any(len(coefs) > 0 for coefs in sig_filtered_coefs):
                        # Combine all significant coefficients
                        all_sig_coefs = np.vstack([coefs for coefs in sig_filtered_coefs if len(coefs) > 0])
                        
                        if all_sig_coefs.shape[0] < 2:
                            corr_matrix = np.full((num_vars, num_vars), np.nan)
                        else:
                            corr_matrix = np.corrcoef(all_sig_coefs.T)
                    else:
                        corr_matrix = np.full((num_vars, num_vars), np.nan)

                    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                    corr_masked = np.where(mask, np.nan, corr_matrix)

                    im = axes[e_idx].imshow(corr_masked, aspect='auto', cmap='cool', vmin=-1, vmax=1, interpolation='nearest')
                    axes[e_idx].set_title(f"{epoch_names[e_idx]}")
                    axes[e_idx].set_xticks(range(num_vars))
                    axes[e_idx].set_yticks(range(num_vars))
                    axes[e_idx].set_xticklabels(var_names, rotation=45, ha='right')
                    axes[e_idx].set_yticklabels(var_names)

                # Adjust layout to make space for the horizontal colorbar below the heatmaps
                plt.subplots_adjust(bottom=0.25)
                cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # x, y, width, height
                fig.colorbar(im, cax=cbar_ax, orientation='horizontal', label='corr coeff')
                fig.suptitle(f'Coefficient correlations (significant only) - Region: {region}', y=0.97)
                fig.tight_layout(rect=[0, 0.15, 1, 0.95])

                save_name = f"{region.replace(' ', '_')}_coefficient_correlation_heatmaps.png"
                fig.savefig(os.path.join(heatmap_dir, save_name), dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f'Coefficient correlation heatmaps (significant only) saved as {os.path.join(heatmap_dir, save_name)}')

        # Calculate average differences between coefficients for each variable pair
        # for each region and epoch (only for cells with significant coefficients)
        if coef_diff_plot_flag:
            print("\n" + "="*60)
            print("AVERAGE COEFFICIENT DIFFERENCES BY REGION AND EPOCH (SIGNIFICANT CELLS ONLY)")
            print("="*60)
            
            # Create folder for heatmaps
            heatmap_dir = 'coefficient_difference_heatmaps'
            os.makedirs(heatmap_dir, exist_ok=True)
            
            epoch_names = out['epoch_names']
            
            for region in region_list:
                region_idxs = [i for i, text in enumerate(regions) if region == text]
                
                if len(region_idxs) == 0:
                    continue
                    
                print(f"\nRegion: {region}")
                print("-" * 50)
                
                # Create separate heatmap for each epoch
                for e_idx in range(num_epochs):
                    coef_region_state = coefs[e_idx, region_idxs, :]
                    p_val_region_state = p_vals[e_idx, region_idxs, :]
                    is_sig_region_state = is_sig[e_idx, region_idxs, :]
                    
                    # Filter out rows with any NaN values
                    valid_mask = ~np.isnan(coef_region_state).any(axis=1)
                    coef_region_state = coef_region_state[valid_mask, :]
                    p_val_region_state = p_val_region_state[valid_mask, :]
                    is_sig_region_state = is_sig_region_state[valid_mask, :]
                    
                    if coef_region_state.shape[0] == 0:
                        # Create empty heatmap
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
                        ax.set_title(f'Average Coefficient Differences - Region: {region}, Epoch: {epoch_names[e_idx]}', fontsize=16, weight='bold')
                        ax.axis('off')
                        plt.tight_layout()
                        
                        # Save empty heatmap
                        save_name = f"{region.replace(' ', '_')}_{epoch_names[e_idx].replace(' ', '_')}_coefficient_differences.png"
                        plt.savefig(os.path.join(heatmap_dir, save_name), dpi=300, bbox_inches='tight', facecolor='white')
                        plt.close(fig)
                        print(f'Empty heatmap saved as {os.path.join(heatmap_dir, save_name)}')
                        continue
                    
                    # Calculate average differences for this region and epoch (only significant cells)
                    avg_diff_matrix = np.full((num_vars, num_vars), np.nan)
                    
                    # Debug: Print coefficient ranges for this region/epoch
                    print(f"  Coefficient ranges for {region}, {epoch_names[e_idx]}:")
                    for v_idx in range(num_vars):
                        coef_range = coef_region_state[:, v_idx]
                        coef_range = coef_range[~np.isnan(coef_range)]
                        if len(coef_range) > 0:
                            print(f"    {var_names[v_idx]}: [{coef_range.min():.4f}, {coef_range.max():.4f}]")
                    
                    for i in range(num_vars):
                        for j in range(num_vars):
                            if i != j:
                                # Only include cells where BOTH variables have significant coefficients
                                sig_mask = (is_sig_region_state[:, i] == 1) & (is_sig_region_state[:, j] == 1)
                                if np.sum(sig_mask) > 0:
                                    # Get coefficients for significant cells
                                    coef_i = coef_region_state[sig_mask, i]
                                    coef_j = coef_region_state[sig_mask, j]
                                    
                                    # Raw coefficient differences
                                    differences = coef_i - coef_j
                                    
                                    avg_diff_matrix[i, j] = np.mean(differences)
                                    
                                    # Debug: Print some statistics (only if average > 10)
                                    if np.sum(sig_mask) > 0 and abs(avg_diff_matrix[i, j]) > 5:
                                        print(f"    {var_names[i]} vs {var_names[j]}: {np.sum(sig_mask)} cells, diff range: [{differences.min():.4f}, {differences.max():.4f}], mean: {avg_diff_matrix[i, j]:.4f}")
                    
                    # Create heatmap for this epoch
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Create mask for upper triangle (set to NaN to hide them)
                    mask = np.triu(np.ones_like(avg_diff_matrix, dtype=bool), k=0)
                    masked_matrix = np.where(mask, np.nan, avg_diff_matrix)
                    
                    # Create heatmap
                    im = ax.imshow(masked_matrix, cmap='RdBu_r', aspect='auto', vmin=-np.nanmax(np.abs(masked_matrix)), vmax=np.nanmax(np.abs(masked_matrix)))
                    
                    # Set ticks and labels
                    ax.set_xticks(range(num_vars))
                    ax.set_yticks(range(num_vars))
                    ax.set_xticklabels(var_names, rotation=45, ha='right')
                    ax.set_yticklabels(var_names)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Average Coefficient Difference', rotation=270, labelpad=15)
                    
                    # Add title
                    ax.set_title(f'Average Coefficient Differences - Region: {region}, Epoch: {epoch_names[e_idx]}\n(Significant cells only)', fontsize=14, weight='bold')
                    
                    plt.tight_layout()
                    
                    # Save heatmap
                    save_name = f"{region.replace(' ', '_')}_{epoch_names[e_idx].replace(' ', '_')}_coefficient_differences.png"
                    plt.savefig(os.path.join(heatmap_dir, save_name), dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close(fig)
                    print(f'Heatmap saved as {os.path.join(heatmap_dir, save_name)}')
            
            # Also create overall average heatmap across all regions and epochs
            print("\n" + "="*60)
            print("OVERALL AVERAGE COEFFICIENT DIFFERENCES (SIGNIFICANT CELLS ONLY)")
            print("="*60)
            
            # Calculate overall averages
            diff_matrix = np.zeros((num_vars, num_vars))
            count_matrix = np.zeros((num_vars, num_vars))
            
            for region in region_list:
                region_idxs = [i for i, text in enumerate(regions) if region == text]
                if len(region_idxs) == 0:
                    continue
                    
                for e_idx in range(num_epochs):
                    coef_region = coefs[e_idx, region_idxs, :]
                    p_val_region = p_vals[e_idx, region_idxs, :]
                    is_sig_region = is_sig[e_idx, region_idxs, :]
                    valid_mask = ~np.isnan(coef_region).any(axis=1)
                    coef_region = coef_region[valid_mask, :]
                    p_val_region = p_val_region[valid_mask, :]
                    is_sig_region = is_sig_region[valid_mask, :]
                    
                    if coef_region.shape[0] == 0:
                        continue
                    
                    # Calculate differences for all variable pairs (only significant cells)
                    for i in range(num_vars):
                        for j in range(num_vars):
                            if i != j:
                                # Only include cells where BOTH variables have significant coefficients
                                sig_mask = (is_sig_region[:, i] == 1) & (is_sig_region[:, j] == 1)
                                if np.sum(sig_mask) > 0:
                                    # Get coefficients for significant cells
                                    coef_i = coef_region[sig_mask, i]
                                    coef_j = coef_region[sig_mask, j]
                                    
                                    # Raw coefficient differences
                                    differences = coef_i - coef_j
                                    
                                    diff_matrix[i, j] += np.sum(differences)
                                    count_matrix[i, j] += len(differences)
            
            # Calculate overall averages
            overall_avg_matrix = np.full((num_vars, num_vars), np.nan)
            for i in range(num_vars):
                for j in range(num_vars):
                    if count_matrix[i, j] > 0:
                        overall_avg_matrix[i, j] = diff_matrix[i, j] / count_matrix[i, j]
            
            # Create overall heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create mask for upper triangle (set to NaN to hide them)
            mask = np.triu(np.ones_like(overall_avg_matrix, dtype=bool), k=0)
            masked_matrix = np.where(mask, np.nan, overall_avg_matrix)
            
            # Create heatmap
            im = ax.imshow(masked_matrix, cmap='RdBu_r', aspect='auto', vmin=-np.nanmax(np.abs(masked_matrix)), vmax=np.nanmax(np.abs(masked_matrix)))
            
            # Set ticks and labels
            ax.set_xticks(range(num_vars))
            ax.set_yticks(range(num_vars))
            ax.set_xticklabels(var_names, rotation=45, ha='right')
            ax.set_yticklabels(var_names)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Average Coefficient Difference', rotation=270, labelpad=15)
            
            # Add title
            ax.set_title('Overall Average Coefficient Differences\n(across all regions and epochs, significant cells only)', fontsize=16, weight='bold')
            
            plt.tight_layout()
            
            # Save the overall heatmap
            overall_heatmap_path = os.path.join(heatmap_dir, 'overall_average_coefficient_differences.png')
            plt.savefig(overall_heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"Overall heatmap saved as: {overall_heatmap_path}")
        
        # ------------------------------------------------------------------
        # Generate pie charts for each region showing distribution of 
        # variables that are best encoded by cells (highest coefficient magnitude)
        # Only include cells that significantly encode at least one variable
        # ------------------------------------------------------------------
        if pie_chart_plot_flag:
            print("\n" + "="*60)
            print("PIE CHARTS: BEST ENCODED VARIABLES BY REGION")
            print("="*60)
            
            epoch_names = out['epoch_names']
            
            for region in region_list:
                region_idxs = [i for i, text in enumerate(regions) if region == text]
                
                if len(region_idxs) == 0:
                    continue
                
                print(f"\nRegion: {region}")
                
                # Combine data across all epochs for this region
                all_best_vars = []
                
                for e_idx in range(num_epochs):
                    coef_region_epoch = coefs[e_idx, region_idxs, :]
                    is_sig_region_epoch = is_sig[e_idx, region_idxs, :]
                    
                    # Filter out rows with any NaN values
                    valid_mask = ~np.isnan(coef_region_epoch).any(axis=1)
                    coef_region_epoch = coef_region_epoch[valid_mask, :]
                    is_sig_region_epoch = is_sig_region_epoch[valid_mask, :]
                    
                    if coef_region_epoch.shape[0] == 0:
                        continue
                    
                    # For each cell, find the variable with highest coefficient magnitude
                    # but only if the cell significantly encodes at least one variable
                    for cell_idx in range(coef_region_epoch.shape[0]):
                        cell_coefs = coef_region_epoch[cell_idx, :]
                        cell_is_sig = is_sig_region_epoch[cell_idx, :]
                        
                        # Check if this cell significantly encodes at least one variable
                        if np.sum(cell_is_sig == 1) > 0:
                            # Find variable with highest coefficient magnitude
                            abs_coefs = np.abs(cell_coefs)
                            best_var_idx = np.argmax(abs_coefs)
                            best_var_name = var_names[best_var_idx]
                            all_best_vars.append(best_var_name)
                
                if len(all_best_vars) == 0:
                    print(f"  No cells with significant encoding found")
                    continue
                
                # Count occurrences of each variable
                var_counts = {}
                for var in var_names:
                    var_counts[var] = all_best_vars.count(var)
                
                # Remove variables with 0 counts
                var_counts = {k: v for k, v in var_counts.items() if v > 0}
                
                if len(var_counts) == 0:
                    print(f"  No valid data for pie chart")
                    continue
                
                # Create pie chart
                fig, ax = plt.subplots(figsize=(8, 8))
                
                labels = list(var_counts.keys())
                sizes = list(var_counts.values())
                
                # Create pie chart with percentages
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                
                # Improve text formatting
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(10)
                    autotext.set_weight('bold')
                
                ax.set_title(f'Best Encoded Variables - Region: {region}\n(Total cells: {len(all_best_vars)})', 
                            fontsize=14, weight='bold', pad=20)
                
                plt.tight_layout()
                
                # Save pie chart
                save_name = f"{region.replace(' ', '_')}_best_encoded_variables_pie_chart.png"
                plt.savefig(save_name, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                print(f"  Pie chart saved as '{save_name}'")
                
                # Print summary
                print(f"  Variable counts:")
                for var, count in var_counts.items():
                    percentage = (count / len(all_best_vars)) * 100
                    print(f"    {var}: {count} cells ({percentage:.1f}%)")

                          


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