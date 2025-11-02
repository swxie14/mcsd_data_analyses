import numpy as np








def correlation_btwn_value_regressors():
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
def plot_coeff_corr():
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

        
        # ------------------------------------------------------------------
        # Generate pie charts for each region showing distribution of 
        # variables that are best encoded by cells (highest coefficient magnitude)
        # Only include cells that significantly encode at least one variable
        # ------------------------------------------------------------------
def pie_chart_plot():
    
    print("PIE CHARTS: BEST ENCODED VARIABLES BY REGION")
    
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

                       